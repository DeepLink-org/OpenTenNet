#! /usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#  - Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  - Neither the name(s) of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import torch
import torch.autograd
import numpy as np
from .binding import einsum, einsumV2, getEinsumOutputShape, TensorMg, \
                     toTensor, fromTensor, init, getOutputShapeMg, \
                     einsumMgV2, einsumV2_autotune, defaultAlgo
from ..common import normalize_subscript
import warnings

# def split_real_imag(tensor):
#     real = tensor[0]
#     imag = tensor[1]
#     return real, imag
def split_real_imag(tensor, **kwargs):
    if kwargs.get('dtype_') == "complex32Torriihalf":
        shape = tensor.shape
        view_as_real = torch.view_as_real(tensor)
        real = view_as_real.flatten()[:int(view_as_real.numel()/2)].view(shape)
        imag = view_as_real.flatten()[int(view_as_real.numel()/2):].view(shape)
    elif kwargs.get('dtype_') == "complex32Toririhalf":
        real = tensor.real
        imag = tensor.imag
    return real, imag

def swap_eq_inputs(equation):
    if '->' in equation:
        equation = equation.split('->')
        # modify input
        lhs = equation[0]
        in1, in2 = lhs.split(',')[0], lhs.split(',')[1]
        rhs = equation[1]
    return in2 + "," + in1 + "->" + rhs
 
def diff_eq(eq, **kwargs):
    chars = [chr(i) for i in range(ord('a'),ord('z')+1)] + [chr(i) for i in range(ord('A'),ord('Z')+1)] +  list("!@#$%^&*(~{}[]|:;<")
    diff = list(set(chars) - set(eq))
    return diff

def modify_eq(equation, **kwargs):
    diff = diff_eq(equation)
    dtype_ = kwargs.get("dtype_")
    if '->' in equation:
        equation = equation.split('->')
        # modify input
        lhs = equation[0]
        in1, in2 = lhs.split(',')[0], lhs.split(',')[1]
        rhs = equation[1]

    if dtype_ == "complex32Toririhalf":
        # modify input
        in1 = in1 + diff[0]
        in2 = in2 + diff[0]
        in2 = diff[1] + in2
        rhs = rhs + diff[1]
        return in1 + "," + in2 + "->" + rhs
    if dtype_ == "complex32Torriihalf":
        # modify input
        in1 = diff[0] + in1
        in2 = diff[0] + in2 
        in2 = diff[1] + in2
        rhs = diff[1] + rhs
        return in1 + "," + in2 + "->" + rhs

def fill_beffer_data(input, **kwargs):
    shape = list(input.shape); shape = [2] + shape
    if isinstance(kwargs.get('buffer_tensors'), torch.Tensor):
        # Reshape buffer tensors
        buffer_tensors = kwargs["buffer_tensors"]
        buffer_tensors = buffer_tensors.flatten()[:input.numel()*2].view(shape)
    else:
        warnings.warn("Buffer tensors not given, creating tensors")
        buffer_tensors = torch.empty(shape, dtype = torch.complex32, device = input.device)

    buffer1, buffer2 = buffer_tensors[0], buffer_tensors[1]

    if kwargs.get('dtype_') == "complex32Toririhalf":
        buffer1.real.copy_(input.real)
        buffer1.imag.copy_(-1*input.imag)
        buffer2.real.copy_(input.imag)
        buffer2.imag.copy_(input.real)

    elif kwargs.get('dtype_') == "complex32Torriihalf":
        in_real, in_imag = split_real_imag(input, **kwargs)

        buffer1r, buffer1i = split_real_imag(buffer1, **kwargs)
        buffer1r.copy_(in_real)
        buffer1i.copy_(-1*in_imag)

        buffer2r, buffer2i = split_real_imag(buffer2, **kwargs)
        buffer2r.copy_(in_imag)
        buffer2i.copy_(in_real)
            
    return buffer1, buffer2, buffer_tensors


class EinsumFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, equation, input_0, input_1=None, kwargs = {}):
        equation, isBinary = normalize_subscript(equation)
        if isBinary and input_1 is None:
            raise RuntimeError('The subscript indicates two inputs, but only one was passed')
        if not isBinary and input_1 is not None:
            raise RuntimeError('The subscript indicates one input, but two were passed')
        if input_1 is None:
            input_1 = input_0.new_empty((1,))
        output = torch.empty(getOutputShape(equation, input_0, input_1, **kwargs), dtype = input_0.dtype, device = input_0.device)

        alpha = kwargs["alpha"] if "alpha" in kwargs.keys() else 1
        beta = kwargs['beta'] if "beta" in kwargs.keys() else 0
        dtype_ = kwargs.get("dtype_")

        try:
            algo = kwargs["algos"][equation]
        except:
            algo = defaultAlgo()

        tuning = False
        if kwargs.get("autotune") == True:
            if ("algos" not in kwargs.keys()):
                warnings.warn("No algo dict given. No autotuning will be performed!")
            elif (kwargs["algos"].get(equation) is None):
                tuning = True

        ##########  modify equation and tensors ########## 
        if dtype_ == "complex32Toririhalf" or dtype_ == "complex32Torriihalf":
            eq_org = equation
            if (input_0.numel() < input_1.numel()): # buffer the smaller tensor
                equation = swap_eq_inputs(equation)
                in1_buffer1, in1_buffer2, buffer_tensors = fill_beffer_data(input_0, **kwargs)
                input_0 = input_1
            else:
                in1_buffer1, in1_buffer2, buffer_tensors = fill_beffer_data(input_1, **kwargs)
            ########## Modify equation ###########
            equation = modify_eq(equation, **kwargs)
            if tuning:
                algo = einsumV2_autotune(equation, torch.view_as_real(input_0), torch.view_as_real(buffer_tensors), torch.view_as_real(output), False, False, alpha, beta, None, None, None)
                kwargs["algos"][eq_org] = algo
            else:
                einsumV2(equation, torch.view_as_real(input_0), torch.view_as_real(buffer_tensors), torch.view_as_real(output), False, False, algo, alpha, beta, None, None, None)
        else:
            if tuning:
                algo = einsumV2_autotune(equation, input_0, input_1, output, False, False, alpha, beta, None, None, None)
                kwargs["algos"][equation] = algo
            else:
                einsumV2(equation, input_0, input_1, output, False, False, algo, alpha, beta, None, None, None)
        
        if isBinary:
            ctx.save_for_backward(input_0, input_1)

        ctx.equation = equation
        ctx.isBinary = isBinary

        return output

    @staticmethod
    def backward(ctx, grad_output):
        equation = ctx.equation
        lhs, modeC = equation.split('->')
        if ctx.isBinary:
            input_0, input_1 = ctx.saved_tensors
            conjugate = False
            if torch.is_complex(input_0) or torch.is_complex(input_1):
                conjugate = True
            modeA, modeB = lhs.split(',')
            d_input_0 = einsum(modeC + ',' + modeB + '->' + modeA, grad_output,
                               input_1, False, conjugate)
            d_input_1 = einsum(modeA + ',' + modeC + '->' + modeB, input_0,
                               grad_output, conjugate, False)
            return None, d_input_0, d_input_1
        else:
            dummy = grad_output.new_empty((1,))
            d_input = einsum(modeC + '->' + lhs, grad_output, dummy, False, False)
            return None, d_input

class Einsum(torch.nn.Module):

    def __init__(self, equation):
        super(Einsum, self).__init__()
        self.equation = equation
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, input_0, input_1):
        return EinsumFunction.apply(self.equation, input_0, input_1)


def _compute_target_tensor(in0, in1, target):
    result = ""
    for m in in0[:-1] + in1[:-1] + in1[-1] + in0[-1]:
        if m in target and not m in result:
            result += m
    # reorder target modes like target
    result = list(result)
    for i in range(len(result)):
        if result[i] not in target: continue
        for j in range(i):
            if result[j] not in target: continue
            if target.index(result[j]) > target.index(result[i]):
                result[i], result[j] = result[j], result[i]
    return ''.join(result)


def EinsumGeneral(equation, *tensors, **kwargs):
    tensors = list(tensors)

    in0 = tensors[0]
    in1 = tensors[1]
    result = EinsumFunction.apply(equation, in0, in1, kwargs)
    return result
    



def einsumForwardV2(output, equation, input_0, input_1=None, **kwargs):
    equation, isBinary = normalize_subscript(equation)
    if isBinary and input_1 is None:
        raise RuntimeError('The subscript indicates two inputs, but only one was passed')
    if not isBinary and input_1 is not None:
        raise RuntimeError('The subscript indicates one input, but two were passed')
    if input_1 is None:
        input_1 = input_0.new_empty((1,))

    alpha = kwargs["alpha"] if "alpha" in kwargs.keys() else 1
    beta = kwargs['beta'] if "beta" in kwargs.keys() else 0
    dtype_ = kwargs.get("dtype_")
    tuning = False
    if kwargs.get("autotune") == True:
        if ("algos" not in kwargs.keys()):
            warnings.warn("No algo dict given. No autotuning will be performed!")
        elif (kwargs["algos"].get(equation) is None):
            tuning = True
    try:
        algo = kwargs["algos"][equation]
    except:
        algo = defaultAlgo()
    ##########  modify equation and tensors ########## 
    if dtype_ == "complex32Toririhalf" or dtype_ == "complex32Torriihalf":
        eq_org = equation
        if (input_0.numel() < input_1.numel()): # buffer the smaller tensor
            equation = swap_eq_inputs(equation)
            in1_buffer1, in1_buffer2, buffer_tensors = fill_beffer_data(input_0, **kwargs)
            input_0 = input_1
        else:
            in1_buffer1, in1_buffer2, buffer_tensors = fill_beffer_data(input_1, **kwargs)
        ########## Modify equation ###########
        equation = modify_eq(equation, **kwargs)
        if tuning:
            algo = einsumV2_autotune(equation, torch.view_as_real(input_0), torch.view_as_real(buffer_tensors), torch.view_as_real(output), False, False, alpha, beta, None, None, None)
            kwargs["algos"][eq_org] = algo
        else:
            einsumV2(equation, torch.view_as_real(input_0), torch.view_as_real(buffer_tensors), torch.view_as_real(output), False, False, algo, alpha, beta, None, None, None)
    else:
        if tuning:
            algo = einsumV2_autotune(equation, input_0, input_1, output, False, False, alpha, beta, None, None, None)
            kwargs["algos"][equation] = algo
        else:
            einsumV2(equation, input_0, input_1, output, False, False, algo, alpha, beta, None, None, None)



def einsumForwardOutputShape(equation, input_0, input_1=None, **kwargs):
    equation, isBinary = normalize_subscript(equation)
    if isBinary and input_1 is None:
        raise RuntimeError('The subscript indicates two inputs, but only one was passed')
    if not isBinary and input_1 is not None:
        raise RuntimeError('The subscript indicates one input, but two were passed')
    if input_1 is None:
        input_1 = input_0.new_empty((1,))
    return getEinsumOutputShape(equation, input_0, input_1, False, False)

def getOutputShape(equation, *tensors, **kwargs):
    tensors = list(tensors)
    in0 = tensors[0]
    in1 = tensors[1]
    
    # split real/imag part if the data type is half
    dtype_ = kwargs.get("dtype_")
    
    # beacause einsum does not support complex32, we must view as float to calculate output shape
    if "complex32Toririhalf" in dtype_:
        output_shape = einsumForwardOutputShape(equation, in0.view(torch.float), in1.view(torch.float))
        
    else:
        output_shape = einsumForwardOutputShape(equation, in0, in1)

    return output_shape 

def EinsumGeneralV2(output, equation, *tensors, **kwargs):
    tensors = list(tensors)
    in0 = tensors[0]
    in1 = tensors[1]

    
    einsumForwardV2(output, equation, in0, in1, **kwargs)

