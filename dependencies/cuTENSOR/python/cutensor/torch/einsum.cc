/*  
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 * 
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name(s) of the copyright holder(s) nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */  
#include <optional>
#include <type_traits> 
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.hpp>
#include <cuComplex.h>

#include "../../einsum.h"
#include "../../einsum_mg.h"

template<>
struct CuTensorTypeTraits<at::Half> {
  static const cudaDataType_t cudaType = CUDA_R_16F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_32F;
  typedef float ScalarType;
};

template<>
struct CuTensorTypeTraits<at::BFloat16> {
  static const cudaDataType_t cudaType = CUDA_R_16BF;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_32F;
  typedef float ScalarType;
};

template<>
struct CuTensorTypeTraits<c10::complex<float>> {
  static const cudaDataType_t cudaType = CUDA_C_32F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_TF32;
  typedef c10::complex<float> ScalarType;
};

template<>
struct CuTensorTypeTraits<c10::complex<double>> {
  static const cudaDataType_t cudaType = CUDA_C_64F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_COMPUTE_64F;
  typedef c10::complex<double> ScalarType;
};

torch::Tensor einsum(
    std::string subscripts,
    torch::Tensor input_0,
    torch::Tensor input_1,
    bool conjA = false,
    bool conjB = false
) {
  at::Tensor output_tensor;
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_0.scalar_type(), "einsum", [&] {
    constexpr int kMaxNumModes_ = 64; // maximal number of modes supported by cuTENSOR
    cutensorOperator_t opA = conjA ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
    cutensorOperator_t opB = conjB ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
    Einsum<scalar_t, int64_t, kMaxNumModes_> myEinsum(subscripts, input_0.sizes().vec(), input_1.sizes().vec(), opA, opB);
    if (!myEinsum.isInitialized()) {
      throw std::runtime_error("cutensor: Initialization failed.");
    }
    if (input_0.device() != input_1.device()) {
      throw std::runtime_error("cutensor: input0 and inpu1 not on the same device.");
    }
    output_tensor = torch::empty(myEinsum.getOutputShape(), input_0.options());
    c10::DeviceIndex device_index = input_0.device().index();
    size_t worksize = myEinsum.getWorksize();
    // at::Tensor workspace = at::empty({static_cast<int>(worksize)}, at::CUDA(at::kByte));
    at::Tensor workspace = at::empty({static_cast<int>(worksize)}, at::device({at::kCUDA, device_index}).dtype(at::kByte));
    CHECK_MG(cudaSetDevice(device_index));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto ret = myEinsum.execute(GetCuTensorHandle(),
                                input_0.data_ptr<scalar_t>(),
                                input_1.data_ptr<scalar_t>(),
                                output_tensor.data_ptr<scalar_t>(),
                                workspace.data_ptr<uint8_t>(),
                                stream);

    if (! ret) throw std::runtime_error("cutensor: Launch failed.");
  });
  return output_tensor;
}

std::vector<int64_t> getEinsumOutputShape(
    std::string subscripts,
    torch::Tensor input_0,
    torch::Tensor input_1,
    bool conjA = false,
    bool conjB = false
) {
  std::vector<int64_t> output_shape;
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_0.scalar_type(), "einsum", [&] {
    constexpr int kMaxNumModes_ = 64; // maximal number of modes supported by cuTENSOR
    cutensorOperator_t opA = conjA ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
    cutensorOperator_t opB = conjB ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
    Einsum<scalar_t, int64_t, kMaxNumModes_> myEinsum(subscripts, input_0.sizes().vec(), input_1.sizes().vec(), opA, opB);
    if (!myEinsum.isInitialized()) {
      throw std::runtime_error("cutensor: Initialization failed.");
    }
    output_shape = myEinsum.getOutputShape();
  });
  return output_shape;
}

int einsumV2_autotune(
    std::string subscripts,
    torch::Tensor input_0,
    torch::Tensor input_1,
    torch::Tensor output_tensor,
    bool conjA = false,
    bool conjB = false, 
    float alpha = 1,
    float beta = 0,
    const std::optional<std::vector<int64_t>> A_stride = std::nullopt,
    const std::optional<std::vector<int64_t>> B_stride = std::nullopt,
    const std::optional<std::vector<int64_t>> C_stride = std::nullopt
) {
  int bestAlgo = CUTENSOR_ALGO_DEFAULT;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_0.scalar_type(), "einsum", [&] {
      constexpr int kMaxNumModes_ = 64; // maximal number of modes supported by cuTENSOR
      cutensorOperator_t opA = conjA ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
      cutensorOperator_t opB = conjB ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
      Einsum<scalar_t, int64_t, kMaxNumModes_> myEinsum(subscripts, input_0.sizes().vec(), input_1.sizes().vec(), opA, opB,
                                                      A_stride, 
                                                      B_stride,
                                                      C_stride);
      if (!myEinsum.isInitialized()) {
        throw std::runtime_error("cutensor: Initialization failed.");
      }
      if (input_0.device() != input_1.device()) {
        throw std::runtime_error("cutensor: input0 and inpu1 not on the same device.");
      }
      c10::DeviceIndex device_index = input_0.device().index();
      size_t worksize = myEinsum.getWorksize();
      at::Tensor workspace = at::empty({static_cast<int>(worksize)}, at::device({at::kCUDA, device_index}).dtype(at::kByte));
      CHECK_MG(cudaSetDevice(device_index));
      auto stream = at::cuda::getCurrentCUDAStream().stream();
      auto ret = myEinsum.autotuning(GetCuTensorHandle(),
                                  input_0.data_ptr<scalar_t>(),
                                  input_1.data_ptr<scalar_t>(),
                                  output_tensor.data_ptr<scalar_t>(),
                                  workspace.data_ptr<uint8_t>(),
                                  stream, bestAlgo, alpha, beta);

      if (!ret) throw std::runtime_error("cutensor: einsumV2_autotune launch failed.");
      
    });
  return bestAlgo;
}



bool einsumV2(
    std::string subscripts,
    torch::Tensor input_0,
    torch::Tensor input_1,
    torch::Tensor output_tensor,
    bool conjA = false,
    bool conjB = false,
    int algo = CUTENSOR_ALGO_DEFAULT,
    float alpha = 1,
    float beta = 0,
    const std::optional<std::vector<int64_t>> A_stride = std::nullopt,
    const std::optional<std::vector<int64_t>> B_stride = std::nullopt,
    const std::optional<std::vector<int64_t>> C_stride = std::nullopt
) {
  // AT_DISPATCH_COMPLEX_TYPES(input_0.scalar_type(), "einsum", [&] (){
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_0.scalar_type(), "einsum", [&] {
    constexpr int kMaxNumModes_ = 64; // maximal number of modes supported by cuTENSOR
    cutensorOperator_t opA = conjA ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
    cutensorOperator_t opB = conjB ? CUTENSOR_OP_CONJ : CUTENSOR_OP_IDENTITY;
    Einsum<scalar_t, int64_t, kMaxNumModes_> myEinsum(subscripts, input_0.sizes().vec(), input_1.sizes().vec(), opA, opB, 
                                                      A_stride, 
                                                      B_stride,
                                                      C_stride);
    if (!myEinsum.isInitialized()) {
      throw std::runtime_error("cutensor: einsumV2 Initialization failed.");
    }
    if (input_0.device() != input_1.device()) {
      throw std::runtime_error("Cutensor: input0 and inpu1 not on the same device.");
    }
    c10::DeviceIndex device_index = input_0.device().index();
    size_t worksize = myEinsum.getWorksize();
    at::Tensor workspace = at::empty({static_cast<int>(worksize)}, at::device({at::kCUDA, device_index}).dtype(at::kByte));
    CHECK_MG(cudaSetDevice(device_index));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto ret = myEinsum.execute(GetCuTensorHandle(),
                                input_0.data_ptr<scalar_t>(),
                                input_1.data_ptr<scalar_t>(),
                                output_tensor.data_ptr<scalar_t>(),
                                workspace.data_ptr<uint8_t>(),
                                stream, algo, alpha, beta);
    if (!ret) throw std::runtime_error("Cutensor:einsumV2 launch failed.");
  });
  return true;
}

bool init(const int32_t numDevices) {
  bool ret = CutensorMgConfig::Init(numDevices);
  if (! ret) throw std::runtime_error("cutensor: Init failed.");
  return true;
}

bool fromTensor(TensorMg& dst, torch::Tensor& src) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, src.scalar_type(), "fromTensor", [&] {
    bool ret = TensorMg::fromTensor<scalar_t>(dst, src);
    if (! ret) throw std::runtime_error("cutensor: failed.");
  });
  return true;
}

bool toTensor(torch::Tensor& dst, TensorMg& src) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, dst.scalar_type(), "toTensor", [&] {
    bool ret = TensorMg::toTensor<scalar_t>(dst, src);
    if (! ret) throw std::runtime_error("cutensor: Launch failed.");
  });
  return true;
}

std::vector<int64_t> getOutputShapeMg(std::string& subscripts, TensorMg& input_0, TensorMg& input_1) {
  EinsumMg myEinsumMg(subscripts, input_0, input_1);
  if (!myEinsumMg.isInitialized()) {
    throw std::runtime_error("cutensorMg: Initialization failed.");
  }
  return myEinsumMg.getOutputShape();
}

// TensorMg einsumMg(std::string& subscripts, TensorMg& input_0, TensorMg& input_1, torch::Tensor& origin) {
//   TensorMg output;
//   AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, origin.scalar_type(), "einsumMg", [&] {
//     EinsumMg myEinsumMg(subscripts, input_0, input_1);
//     if (!myEinsumMg.isInitialized()) {
//       throw std::runtime_error("cutensor: Initialization failed.");
//     }
//     std::vector<int64_t> shape = myEinsumMg.getOutputShape();
//     if (shape.size() < 1) {
//       throw std::runtime_error("cutensorMg: shape error.");
//     }
//     output.init(shape);
//     bool ret = myEinsumMg.execute<scalar_t>(input_0, input_1, output);
//     if (! ret) throw std::runtime_error("cutensor: Launch failed.");
//   });
//   return output;
// }

bool einsumMgV2(std::string& subscripts, TensorMg& input_0, TensorMg& input_1, TensorMg& output) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::ComplexFloat, "einsumMg", [&] {
    EinsumMg myEinsumMg(subscripts, input_0, input_1);
    if (!myEinsumMg.isInitialized()) {
      throw std::runtime_error("cutensor: Initialization failed.");
    }
    bool ret = myEinsumMg.execute<scalar_t>(input_0, input_1, output);
    if (! ret) throw std::runtime_error("cutensor: Launch failed.");
  });
  return true;
}
int defaultAlgo(){
  return CUTENSOR_ALGO_DEFAULT;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("einsum", &einsum, "Einsum");
  m.def("getEinsumOutputShape", &getEinsumOutputShape, "getEinsumOutputShape");
  m.def("einsumV2", &einsumV2, "EinsumV2");

  pybind11::class_<TensorMg>(m, "TensorMg")
        // .def(pybind11::init<const std::vector<int64_t> &>())
        .def(pybind11::init<const std::vector<int64_t> &,
                            const std::vector<int64_t> &,
                            const std::vector<int32_t> &>())
        .def("getNumModes", &TensorMg::getNumModes)
        .def("setNumModes", &TensorMg::setNumModes)
        .def("getBlockDevices", &TensorMg::getBlockDevices)
        .def("getDeviceCount", &TensorMg::getDeviceCount)
        .def("getExtent", &TensorMg::getExtent)
        .def("getBlockSize", &TensorMg::getBlockSize)
        .def("getData", &TensorMg::getData)
        .def("getRemainingDevices", &TensorMg::getRemainingDevices)
        .def("getShape", &TensorMg::getShape)
        .def("getTensors", &TensorMg::getTensors)
        ;
  
    m.def("init", &init, "init devices");
    m.def("toTensor", &toTensor, "toTensor");
    m.def("fromTensor", &fromTensor, "fromTensor");
    m.def("getOutputShapeMg", &getOutputShapeMg, "getOutputShapeMg");
    // m.def("einsumMg", &einsumMg, "einsumMg");
    m.def("einsumMgV2", &einsumMgV2, "einsumMgV2");
    m.def("einsumV2_autotune", &einsumV2_autotune, "einsumV2_autotune");
    m.def("defaultAlgo", &defaultAlgo, "defaultAlgo");
}
