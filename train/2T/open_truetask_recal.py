# -*- coding: utf-8 -*-
import os
import time
from math import ceil, sqrt, log
from functools import reduce
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.distributed as dist
import re
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from cutensor.torch import EinsumGeneral, EinsumGeneralV2, getOutputShape
import utils
import vec_h28
import vec_82h

args = utils.parse_args()

nodes_per_task = int(os.environ["nodes_per_task"]) # 做一个子任务需要多少个node
gpus_per_task = int(os.environ["ntasks_per_node"])
mnmodes = int(log(nodes_per_task, 2)) # modes for multi nodes for all-to-all single
mgmodes = mnmodes+int(log(gpus_per_task, 2))# modes for multi gpus, 3 是每个节点有2^3个gpu
split = 512*16
torch.random.manual_seed(0)

############# SETTING UP FOR MULTI_NODE CONFIGURATION ############################################################
utils.setup_distributed()
'''
分布式一共有三个层次
1. node_ : 节点层
2. subtask_ : 子任务层
3. world_ : 全局层
'''
world_rank = int(os.environ["RANK"]) # 全局rank编号
world_size = int(os.environ["WORLD_SIZE"])  # globalrank, 即参与计算的所有rank

node_rank = int(os.environ["LOCAL_RANK"]) #在一个node里面的rank编号
node_world_size = torch.cuda.device_count() # 一个node里有多少个rank
node_idx = world_rank // node_world_size # 该rank属于第几个node

subtask_world_size = node_world_size*nodes_per_task # 做一个子任务需要多少个rank
subtask_idx = world_rank // subtask_world_size # 该rank属于第几个 "多节点子任务"
subtask_rank = world_rank % subtask_world_size # "多节点子任务" 的rank编号
subtasks = world_size // subtask_world_size
# 定义并把tensor放置到单独的GPU上
device = torch.device("cuda", node_rank)

kwargs = {}
kwargs["world_rank"] = world_rank; kwargs["world_size"] = world_size
kwargs["node_rank"] = node_rank; kwargs["node_world_size"] = node_world_size; kwargs["node_idx"] = node_idx
kwargs["subtask_world_size"] = subtask_world_size; kwargs["subtask_idx"] = subtask_idx; kwargs["subtask_rank"] = subtask_rank; kwargs["subtasks"] = subtasks
kwargs["world_rank"] = world_rank; kwargs["mgmodes"] = mgmodes; kwargs["device"] = device

##################################################################################################################
############# SETTING UP FOR HYPER-PARAM CONFIGURATION ###########################################################

# data type of einsum calculation
typeCal = "complexFloat" if args.data_type else "complexHalf"
# data type of allToall communication
typecom = args.typeCom
kwargs["typeCom"] = typecom; kwargs["typeCal"] = typeCal; kwargs["autotune"] = args.autotune
autotune = args.autotune
scale_path, algo_path, result_path, trace_path = utils.getFilePath(args, prefix="open_", **kwargs)
autotune = args.autotune
#################################################################################################################
############# SETTING UP FOR NCCL CONFIGURATION ########################################################
job_names = ["_nnTo1n", "_pairwiseTo1n"]
reduce_job, subtask_gps, node_gps = utils.make_communicate_group(args.warmup, **kwargs)
kwargs["subtask_gps"] = subtask_gps
kwargs["node_gps"] = node_gps
##################################################################################################################
############# SETTING UP FOR TENSOR CREATION #################################################################
cont_file = 'TensorNetwork/4T/sc38_reproduce_scheme_n53_m20_ABCDCDAB_3000000_einsum_10_open.pt'
nsch = torch.load(f'TensorNetwork/4T/open_sc38_nsch_split{split}_mg{mgmodes}_splitmn_SelectStep.pt')
tensors, scheme, slicing_indices, bitstrings = torch.load(cont_file)[:4]
nsch =  utils.prepare_nsch(nsch, split, device)
slicing_edges = list(slicing_indices.keys())
stemPtr = 0
if args.data_type == 0:
    kwargs["dtype_"] = "complex32Toririhalf"
    # cutensor 不支持 complex32，需要一个tensor储存扩充后的小tensor
    kwargs["buffer_tensors"] = torch.zeros(2**27, dtype = torch.complex32, device = device)
    tensors_gpu = [tensor.to(device, dtype=torch.complex32) for tensor in tensors]
    stemtensor = [torch.empty([2**33], dtype = torch.complex32, device = device), torch.empty([2**33], dtype = torch.complex32, device = device)]
elif args.data_type == 1:
    kwargs["dtype_"] = "complex64"
    tensors_gpu = [tensor.to(device, dtype=torch.complex64) for tensor in tensors]
    stemtensor = [torch.empty([2**32], dtype = torch.cfloat, device = device), torch.empty([2**32], dtype = torch.cfloat, device = device)]

kwargs["alpha"] = 1 # cutensor contraction 的 alpha
if world_rank == 0:
    print(f"warmup: {args.warmup}\ndata type of calculation: {typeCal}\nis_scale: {args.is_scale}\ndata type of allToall communication: {typecom}\nautotune: {args.autotune}\nntask: {args.ntask}\n", flush = True)
###################################################################################################################
class MgTensor:  
    @property
    def curtensor(self):
        nelem = torch.tensor(self.shape).prod().item()
        return self.stemtensor[self.pcurtensor][:nelem].view(self.shape)
    
    @property
    def nexttensor(self):
        ptr = 1 - self.pcurtensor
        return self.stemtensor[ptr]
    
    def setnewtensor(self,newshape):
        self.pcurtensor = 1 - self.pcurtensor
        self.shape = newshape
        global stemPtr
        stemPtr = 1 - stemPtr

    
    def __init__(self, stemtensor, sgtensor, mgmodes, convert = False, pow_idx = 5., scale = 1):
        self.pcurtensor = 0
        self.node_rank = node_rank
        self.node_world_size = node_world_size
        self.subtask_rank = subtask_rank
        self.subtask_world_size = subtask_world_size
        self.stemtensor = stemtensor
        
        assert type(sgtensor) == torch.Tensor
        if sgtensor.dtype != torch.int8:
            if convert:
                self.shape = sgtensor.shape
                self.curtensor[:] = sgtensor
            else:
                sgtensor = sgtensor.flatten(end_dim = mgmodes)
                sgtensor = torch.chunk(sgtensor, 2**mgmodes)[self.subtask_rank]
                self.shape = sgtensor.shape
                self.curtensor[:] = sgtensor
        else:
            if convert:
                numel = torch.tensor(sgtensor.shape).prod().item()
                n = int(log(numel, 2))
                self.shape = [2]*(n-1)
                vec_82h.int82half(sgtensor.view(torch.int8).view(-1), torch.view_as_real(self.curtensor[:]).view(-1), pow_idx, scale)
            else:
                print(f"ERROR, not implemented for this type", flush=True)
                
    
    def einsum(self, nstep, ein, insg2, task_id, **kwargs):
        typeCom = kwargs["typeCom"]
        ein_list = re.split('->|,', ein)
        mgchar = list(ein_list[2][:mgmodes])
        # ein_new = re.sub('|'.join(mgchar), '', ein)
        ein_new = ein_list[0].replace("".join(mgchar), "") + "," + ein_list[1] + "->" + ein_list[2][mgmodes:]
        ein_new = re.sub('|'.join(mgchar), '', ein_new)
        # if world_rank == 0:
        #     print(f"nstep {nstep}", flush=True)
        #     print(f"ein {ein}", flush=True)
        #     print(f"ein_new {ein_new}", flush=True)
        if ein_list[0][:mnmodes] != ein_list[2][:mnmodes]:
            # if world_rank == 0:
            #     print(f"nstep {nstep}，节点间 ein {ein}", flush = True)
            group = subtask_gps[subtask_idx]
            if "int8" in typeCom:
                int_com.int8_communicate(task_id, nstep, self, group, **kwargs)
            elif typeCom == "int4kernel":
                groupsize = 128
                if world_rank==0:
                    print(f"group size {groupsize}", flush=True)
                int_com.int4_communicate(task_id, nstep, self, group, groupsize, **kwargs)
            else:
                rawtensor = self.curtensor.flatten(end_dim = mgmodes-1)
                newmgtensor = self.nexttensor
                dist.all_to_all_single(newmgtensor, rawtensor, group = group)
                self.setnewtensor(self.shape)
        elif ein_list[0][mnmodes:mgmodes] != ein_list[2][mnmodes:mgmodes]:
            # if world_rank == 0:
            #     print(f"nstep {nstep}，节点内 ein {ein}", flush = True)
            group = node_gps[node_idx]
            rawtensor = self.curtensor.flatten(end_dim = mgmodes-mnmodes-1)
            newmgtensor = self.nexttensor
            dist.all_to_all_single(newmgtensor, rawtensor, group = group)
            self.setnewtensor(self.shape)
        # if world_rank == 0:
        #     print(f"nstep {nstep}, mgmodes {mgmodes}, ein {ein}, ein_new {ein_new}", flush = True)
        EinsumGeneralV2_choose_method(nstep, self, ein_new, insg2, **kwargs)
    
    def flatsplit(self, flat, split, chunks = split // subtask_world_size, mgmodes = mgmodes):
        # Here I assumed that the tensors won't be larger than the splited tensor
        # It may won't work in other tensor network
        mgtensorsplit = utils.MgTensorSplit(self.curtensor, self.nexttensor.flatten(), flat, chunks, mgmodes)
        return mgtensorsplit

    def abs_max(self):
        max0 = torch.view_as_real(self.curtensor).max().abs()
        min0 = torch.view_as_real(self.curtensor).min().abs()
        maxi = max(max0, min0)
        return maxi
            
def EinsumGeneralV2_choose_method(nstep, mgtensor, ein, tensor_j, **kwargs):
    ein_list = re.split('->|,', ein)
    ein_0, ein_1, _ = utils.remove_common_suffixes(ein_list[0], ein_list[1])
    ein_mm = ein_list[0] + "," + ein_list[1] + "->" + ein_0 + ein_1
    ein_permute = ein_0 + ein_1 + "->" + ein_list[2]

    newshape = getOutputShape(ein, mgtensor.curtensor, tensor_j, **kwargs)
    EinsumGeneralV2(mgtensor.nexttensor, ein, mgtensor.curtensor, tensor_j, **kwargs)
    mgtensor.setnewtensor(newshape)


   
def cont_nsch_split(tensors, tensors_p2, nsch, task_id, mgmodes = mgmodes, **kwargs):
    for nstep, step in enumerate(nsch[:329]):
        with record_function(f"step{nstep}"):
            i, j = step['index']
            if scale_class.is_scale:
                kwargs["alpha"] = scale_class.get_scale(task_id, nstep, tensors[i], tensors[j], **kwargs)
            if nstep < 329:
                if step['type'] == 1:
                    flati, flatj = step['flat']
                    # print(step['flat'])
                    if flati > 1:
                        assert type(tensors[i]) == MgTensor    
                        tensors[i] = tensors[i].flatsplit(flati, split) # convert MgTensor -> utils.MgTensorSplit
                    if flatj > 1:
                        lenj = len(tensors[j].shape)
                        tensors[j] = tensors[j].reshape([-1] + [2] * (lenj - flatj))
                    chubi, chubj = step['chunk_batch']
                    
                    for chuindex in range(tensors[i].chunks):
                        pi = tensors[i].curtensors[chuindex][chubi[chuindex + split // subtask_world_size * subtask_rank]]
                        pj = tensors[j][chubj[chuindex + split // subtask_world_size * subtask_rank]]
                        
                        newshape = getOutputShape(step['ein_2'], pi, pj, **kwargs)
                        tensors[i].setnewtensor(chuindex, newshape)
                        EinsumGeneralV2(tensors[i].nexttensors[chuindex], step['ein_2'], pi, pj, **kwargs)

                        del pi
                        del pj

                    tensors[i].swap_tensors()  
                    tensors[j] = []         
                    
                elif step['type'] == 2 or step['type'] == 3:
                    if type(tensors[i]) == utils.MgTensorSplit:
                        for chuindex in range(tensors[i].chunks):
                            # tensors[i][x] = EinsumGeneral(step['ein_2'], tensors[i][x], tensors[j])
                            pi = tensors[i].curtensors[chuindex]
                            pj = tensors[j]
                            newshape = getOutputShape(step['ein_2'], pi, pj, **kwargs)
                            tensors[i].setnewtensor(chuindex, newshape)
                            EinsumGeneralV2(tensors[i].nexttensors[chuindex], step['ein_2'], pi, pj, **kwargs)
                        tensors[i].swap_tensors()   
                    elif 'reorder_ein' in step:
                        if type(tensors[i]) == torch.Tensor:
                            tensors[i] = EinsumGeneral(step['reorder_ein'], tensors[i], tensors[j], **kwargs)
                            tensors[i] = MgTensor(stemtensor, tensors[i], mgmodes)
                        else:
                            assert type(tensors[i]) == MgTensor
                            tensors[i].einsum(nstep, step['reorder_ein'], tensors[j], task_id, mnmodes = mnmodes, mgmodes = mgmodes, **kwargs)

                            # max0 = torch.view_as_real(tensors[i].curtensor).max().abs()
                            # min0 = torch.view_as_real(tensors[i].curtensor).min().abs()
                            # maxi = max(max0, min0)
                            # dist.all_reduce(maxi, dist.ReduceOp.MAX, group = subtask_gps[subtask_idx])
                            # if world_rank == 0:
                            #     print(f"nstep {nstep}, 65000/maxi.item() {65000/maxi.item()}", flush=True)
                    else:
                        # torch.cuda.empty_cache()
                        tensors[i] = EinsumGeneral(step['ein_2'], tensors[i], tensors[j], **kwargs)
                    tensors[j] = []
                tensors_p2[i] = tensors[i]
                
    nstep = 329
    step = nsch[329]
    i, j = step['index']
    if scale_class.is_scale:
        kwargs["alpha"] = scale_class.get_scale(task_id, nstep, tensors[i], tensors[j], **kwargs)

    group = node_gps[node_idx]
    rawtensor = tensors[i].curtensor.flatten(end_dim = mgmodes-mnmodes-1)
    newmgtensor = tensors[i].nexttensor
    dist.all_to_all_single(newmgtensor, rawtensor, group = group)
    tensors[i].setnewtensor(tensors[i].shape)

     # 确保每张卡上 input的第一个字母和output的第一个字母一样
    # 不一样就要做permute
    ein = step['reorder_ein']
    ein_list = re.split('->|,', ein)
    mgchar = list(ein_list[2][:mgmodes])
    ein_new = re.sub('|'.join(mgchar), '', ein)

    ein_reorder = re.split('->|,', ein_new)
    permute_eq = [i for i in range(len(ein_reorder[0]))]
    char_out0 = ein_reorder[0].find(ein_reorder[2][0])
    assert char_out0 in permute_eq
    if char_out0 != 0:
        permute_eq[0], permute_eq[char_out0] = char_out0, 0
        tensors[i].nexttensor[:tensors[i].curtensor.numel()].view(tensors[i].shape).copy_(tensors[i].curtensor.permute(permute_eq))
        tensors[i].setnewtensor(tensors[i].shape)
        letters = [letter for letter in ein_reorder[0]]
        ein_reorder[0] = "".join([letters[idx] for idx in permute_eq])
    # 重计算，每个卡上的计算数据量减半, part2的数据用int8存起来
    max_p2 = max(torch.view_as_real(tensors[i].curtensor[1]).max().abs(), torch.view_as_real(tensors[i].curtensor[1]).min().abs())
    max_p2.pow_(1./5.)
    scale_p2 = 127./max_p2.item()
    tensori_p2tmp = tensors[i].curtensor[1]
    tensori_p2 = torch.empty(tensors[i].curtensor.shape[2:], dtype = torch.complex32, device = device)
    tensori_p2 = tensori_p2.view(torch.int8)
    vec_h28.half2int8(torch.view_as_real(tensori_p2tmp).view(-1), tensori_p2.view(-1), 5., scale_p2)

    shape = tensors[i].curtensor[0].shape
    # tensori_p2 = torch.clone(tensors[i].curtensor[1])
    # 做完之后把数据分成两份 [p1, p2] ，p1继续张量计算，p2存起来(等p1算完了再继续) 
    tensors[i].setnewtensor(shape)
    tensors[i].setnewtensor(shape)
    # p1继续张量计算
    mgmodes_recal = mgmodes +1
    ein_new = ein_reorder[0][1:] + "," + ein_reorder[1] + "->" + ein_reorder[2][1:]
    ein_step329 = ein_new
    # kwargs["alpha"] = scale_class.set_scale(task_id, nstep, 10**6)
    EinsumGeneralV2_choose_method(nstep, tensors[i], ein_new, tensors[j], **kwargs)
    # kwargs["alpha"] = 1

    part = 0
    for nstep, step in enumerate(nsch):
        if nstep < 330:
            continue
        with record_function(f"step{nstep}"):
            i, j = step['index']
            if scale_class.is_scale:
                kwargs["alpha"] = scale_class.get_scale(task_id, nstep, tensors[i], tensors[j], **kwargs)

            if step['type'] == 1:
                flati, flatj = step['flat']
                # print(step['flat'])
                if flati > 1:
                    assert type(tensors[i]) == MgTensor    
                    chunks = split // subtask_world_size // 2
                    # chunks =  split // subtask_world_size
                    tensors[i] = tensors[i].flatsplit(flati, split, chunks, mgmodes_recal) # convert MgTensor -> utils.MgTensorSplit
                if flatj > 1:
                    lenj = len(tensors[j].shape)
                    tensors[j] = tensors[j].reshape([-1] + [2] * (lenj - flatj))
                chubi, chubj = step['chunk_batch']
                
                for chuindex in range(tensors[i].chunks):
                    pi = tensors[i].curtensors[chuindex][chubi[chuindex + split // subtask_world_size * subtask_rank]]
                    pj = tensors[j][chubj[chuindex + split // subtask_world_size * subtask_rank]]

                    # if chuindex == 0:
                    #     print(f"tensors[i].curtensors[{chuindex}] {tensors[i].curtensors[chuindex].shape}, chubi[chuindex + split // subtask_world_size * subtask_rank].max {chubi[chuindex + split // subtask_world_size * subtask_rank].max()}", flush = True)
                    # 分成两个 parts， 但是第一个mode不会消失，只是extent减半
                    newshape = getOutputShape(step['ein_2'], pi, pj, **kwargs)
                    tensors[i].setnewtensor(chuindex, newshape)
                    EinsumGeneralV2(tensors[i].nexttensors[chuindex], step['ein_2'], pi, pj, **kwargs)

                    del pi
                    del pj
                    
                # if world_rank == 0:
                #     print(f"nstep {nstep}, cur ptr[-1] {tensors[i].pct[-1]},  next ptr[-1] {tensors[i].pnt[-1]}", flush=True)     
                
                tensors[i].swap_tensors()  
                tensors[j] = []         
                
            elif step['type'] == 2 or step['type'] == 3:
                if type(tensors[i]) == utils.MgTensorSplit:
                    for chuindex in range(tensors[i].chunks):
                        # tensors[i][x] = EinsumGeneral(step['ein_2'], tensors[i][x], tensors[j])
                        pi = tensors[i].curtensors[chuindex]
                        pj = tensors[j]
                        newshape = getOutputShape(step['ein_2'], pi, pj, **kwargs)
                        tensors[i].setnewtensor(chuindex, newshape)
                        EinsumGeneralV2(tensors[i].nexttensors[chuindex], step['ein_2'], pi, pj, **kwargs)
                    tensors[i].swap_tensors()   
                elif 'reorder_ein' in step:
                    if type(tensors[i]) == torch.Tensor:
                        tensors[i] = EinsumGeneral(step['reorder_ein'], tensors[i], tensors[j], **kwargs)
                        tensors[i] = MgTensor(stemtensor, tensors[i], mgmodes)
                    else:
                        assert type(tensors[i]) == MgTensor
                        tensors[i].einsum(nstep, step['reorder_ein'], tensors[j], task_id, mnmodes=mnmodes, mgmodes = mgmodes_recal, **kwargs)
                else:
                    if nstep == 433:
                        nextstem = 0
                        newshape = getOutputShape(step['ein_2'], tensors[i], tensors[j], **kwargs)
                        nelem = torch.tensor(newshape).prod().item()
                        outTensor = stemtensor[nextstem].flatten()[-nelem:].view(newshape)
                        EinsumGeneralV2(outTensor, step['ein_2'], tensors[i], tensors[j], **kwargs)
                        tensors[i] = outTensor
                    else:
                        tensors[i] = EinsumGeneral(step['ein_2'], tensors[i], tensors[j], **kwargs)

                tensors[j] = []

    if type(tensors[i]) == utils.MgTensorSplit:
        res_p0 = torch.cat([x for x in tensors[i].curtensors])
    else:
        res_p0 = torch.cat([x.curtensor for x in tensors[i]])


    part = 1
    tensors = tensors_p2
    for nstep, step in enumerate(nsch):
        i, j = step['index']
        if scale_class.is_scale:
            kwargs["alpha"] = scale_class.get_scale(task_id, nstep, tensors[i], tensors[j], **kwargs)

        if nstep < 329:
            continue

        else:
            if nstep == 329:
                # 重计算 part2
                ein_new = ein_step329
                # # 反量化
                tensors[i] = MgTensor(stemtensor, tensori_p2, mgmodes, convert = True, pow_idx = 5., scale = scale_p2)

                # 继续算p2 (等p1算完了再继续) 
                # tensors[i] = MgTensor(stemtensor, tensori_p2, mgmodes, convert = True)

                EinsumGeneralV2_choose_method(nstep, tensors[i], ein_new, tensors[j], **kwargs)
                
                del tensori_p2
                continue

            if step['type'] == 1:
                flati, flatj = step['flat']
                # print(step['flat'])
                if flati > 1:
                    assert type(tensors[i]) == MgTensor    
                    chunks =  split // subtask_world_size // 2
                    tensors[i] = tensors[i].flatsplit(flati, split, chunks, mgmodes_recal) # convert MgTensor -> utils.MgTensorSplit
                if flatj > 1:
                    lenj = len(tensors[j].shape)
                    tensors[j] = tensors[j].reshape([-1] + [2] * (lenj - flatj))
                chubi, chubj = step['chunk_batch']
                
                for chuindex in range(tensors[i].chunks):
                    
                    pi = tensors[i].curtensors[chuindex][chubi[chuindex + tensors[i].chunks + split // subtask_world_size * subtask_rank]]
                    pj = tensors[j][chubj[chuindex + tensors[i].chunks + split // subtask_world_size * subtask_rank]]

                    newshape = getOutputShape(step['ein_2'], pi, pj, **kwargs)
                    tensors[i].setnewtensor(chuindex, newshape)
                    EinsumGeneralV2(tensors[i].nexttensors[chuindex], step['ein_2'], pi, pj, **kwargs)

                    del pi
                    del pj

                tensors[i].swap_tensors()  
                tensors[j] = []    
                
            elif step['type'] == 2 or step['type'] == 3:
                if type(tensors[i]) == utils.MgTensorSplit:
                    for chuindex in range(tensors[i].chunks):
                        # tensors[i][x] = EinsumGeneral(step['ein_2'], tensors[i][x], tensors[j])
                        pi = tensors[i].curtensors[chuindex]
                        pj = tensors[j]
                        newshape = getOutputShape(step['ein_2'], pi, pj, **kwargs)
                        tensors[i].setnewtensor(chuindex, newshape)
                        EinsumGeneralV2(tensors[i].nexttensors[chuindex], step['ein_2'], pi, pj, **kwargs)
                    tensors[i].swap_tensors()   
                elif 'reorder_ein' in step:
                    if type(tensors[i]) == torch.Tensor:
                        tensors[i] = EinsumGeneral(step['reorder_ein'], tensors[i], tensors[j], **kwargs)
                        tensors[i] = MgTensor(stemtensor, tensors[i], mgmodes)
                    else:
                        assert type(tensors[i]) == MgTensor
                        tensors[i].einsum(nstep, step['reorder_ein'], tensors[j], task_id, mnmodes = mnmodes, mgmodes = mgmodes_recal, **kwargs)
                else:
                    if nstep == 433:
                        nextstem = 1
                        newshape = getOutputShape(step['ein_2'], tensors[i], tensors[j], **kwargs)
                        nelem = torch.tensor(newshape).prod().item()
                        outTensor = stemtensor[nextstem].flatten()[-nelem:].view(newshape)
                        EinsumGeneralV2(outTensor, step['ein_2'], tensors[i], tensors[j], **kwargs)
                        tensors[i] = outTensor
                    else:
                        tensors[i] = EinsumGeneral(step['ein_2'], tensors[i], tensors[j], **kwargs)
                tensors[j] = []


    if type(tensors[i]) == utils.MgTensorSplit:
        res_p1 = torch.cat([x for x in tensors[i].curtensors])
    else:
        res_p1 =  torch.cat([x.curtensor for x in tensors[i]])
    return torch.cat([res_p0, res_p1])


def calc_task(s, **kwargs):
    configs = list(map(int, np.binary_repr(s, len(slicing_edges))))
    sliced_tensors = tensors_gpu.copy()
    for x in range(len(slicing_edges)):
        m, n = slicing_edges[x]
        idxm_n, idxn_m = slicing_indices[(m, n)]
        sliced_tensors[m] = sliced_tensors[m].select(idxm_n, configs[x]).clone()
        sliced_tensors[n] = sliced_tensors[n].select(idxn_m, configs[x]).clone()
    sliced_tensors_p2 = sliced_tensors.copy()
    ans = cont_nsch_split(sliced_tensors, sliced_tensors_p2, nsch, s, **kwargs)
    # time_end = time.time()
    # tottime = time_end - time_begin
    tottime = None
    return ans, tottime

################### Tuning for the best cutensor algos and scales ###################
total_tasks = subtasks*args.ntask
total_steps = len(nsch)
scale_class = utils.Scale_Class(args.is_scale, total_tasks, total_steps, device, f"{scale_path}/scale.pt")
int_com = utils.Communicate_quant(subtask_rank, total_tasks, total_steps, device, f"{scale_path}/Int8_scale.pt")
if os.path.exists(algo_path) and kwargs["autotune"] == True:
    algo_dict = torch.load(algo_path)
    kwargs["algos"] = algo_dict
else:
    kwargs["algos"] = {}
    
dist.barrier()
torch.cuda.synchronize()
time_begin = time.time()
task_id = 0 + subtask_idx * args.ntask
ans = scale_class.rescale(task_id, calc_task(task_id, **kwargs)[0].to(dtype=torch.complex64),  **kwargs)
torch.cuda.synchronize()
time_end = time.time()
if world_rank == 0:
    print(f"warmup {world_rank} used time {round(time_end-time_begin, 3)}s", flush = True)
kwargs["autotune"] = False # remember to close autotune
################### Save training data ################### 
if world_rank == 0 and autotune:
    torch.save(kwargs["algos"], algo_path)
    print(f"Tuning algos saved to {algo_path}", flush=True)

################### Performe TrueTask ##############################
ntask = args.ntask
for s in range(ntask):
    if world_rank == 0 and not os.path.exists(f"{result_path}/ntask{ntask*subtasks-1}"):
        os.makedirs(f"{result_path}/ntask{ntask*subtasks-1}")
torch.cuda.empty_cache()
dist.barrier()

torch.cuda.synchronize()
time_begin = time.time()
for s in range(ntask):
    task_id = s + subtask_idx * ntask
    if s == 0:
        ans = scale_class.rescale(task_id, calc_task(task_id, **kwargs)[0].to(dtype=torch.complex64), **kwargs)
    else:
        ans += scale_class.rescale(task_id, calc_task(task_id, **kwargs)[0].to(dtype=torch.complex64), **kwargs)

torch.cuda.synchronize()
time_end = time.time()

if world_rank == 0:
    # print(f"Scale saved to {scale_class.output_path}", flush = True)
    # print(f"Scale for int8 communication saved to {int_com.output_path}", flush = True)
    print(f"rank {world_rank} used time {round(time_end-time_begin, 3)}s")

################### Save trained data ###############################
scale_class.save_scale(reduce_job, **kwargs)
int_com.save_scale(reduce_job, **kwargs)
    
######################### Reduce answer ########################################
del stemtensor; torch.cuda.empty_cache()
torch.cuda.synchronize()
time_begin = time.time()
ans = utils.reduceAns(ans, reduce_job, **kwargs)
torch.cuda.synchronize()
time_end = time.time()
if world_rank == 0:
    print(f"Reduce answer used time {round(time_end-time_begin, 3)}s", flush = True)
    print(f"save result in {result_path}/ntask{ntask*subtasks-1}/", flush = True)
if subtask_idx == 0:
    torch.save(ans.cpu(), f"{result_path}/ntask{ntask*subtasks-1}/rank{subtask_rank}.pt")
