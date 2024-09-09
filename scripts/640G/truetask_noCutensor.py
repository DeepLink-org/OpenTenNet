# -*- coding: utf-8 -*-
import os
import time
from math import ceil, sqrt, log
from functools import reduce

import torch
import torch.distributed as dist
import re
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from cutensor.torch import EinsumGeneral, EinsumGeneralV2, getOutputShape
import utils
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

args = utils.parse_args()

nodes_per_task = int(os.environ["nodes_per_task"]) # 做一个子任务需要多少个node
gpus_per_task = int(os.environ["ntasks_per_node"])
mnmodes = int(log(nodes_per_task, 2)) # modes for multi nodes for all-to-all single
mgmodes = mnmodes+int(log(gpus_per_task, 2))# modes for multi gpus, 3 是每个节点有2^3个gpu
split = 512
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
node_world_size = int(os.environ["ntasks_per_node"]) # 一个node里有多少个rank
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
scale_path, algo_path, result_path, trace_path = utils.getFilePath(args, **kwargs)
autotune = args.autotune
#################################################################################################################
############# SETTING UP FOR NCCL CONFIGURATION ########################################################
job_names = ["_nnTo1n", "_pairwiseTo1n"]
reduce_job, subtask_gps, node_gps = utils.make_communicate_group(args.warmup, **kwargs)
kwargs["subtask_gps"] = subtask_gps
kwargs["node_gps"] = node_gps
##################################################################################################################
############# SETTING UP FOR TENSOR CREATION #################################################################
cont_file = 'TensorNetwork/640G/reproduce_scheme_n53_m20_ABCDCDAB_3000000_einsum_10_open_shortver.pt'
nsch = torch.load(f'TensorNetwork/640G/640G_rep_nsch640_split{split}_mg{mgmodes}_splitmn.pt')

tensors, scheme, slicing_indices = torch.load(cont_file)[:3]
nsch =  utils.prepare_nsch(nsch, split, device)
slicing_edges = list(slicing_indices.keys())
stemPtr = 0
if args.data_type == 0:
    kwargs["dtype_"] = "complex32Toririhalf"
    # cutensor 不支持 complex32，需要一个tensor储存扩充后的小tensor
    # kwargs["buffer_tensors"] = torch.zeros(2**27, dtype = torch.complex32, device = device)
    tensors_gpu = [tensor.to(device, dtype=torch.complex32) for tensor in tensors]
    stemtensor = [None, None]
elif args.data_type == 1:
    kwargs["dtype_"] = "complex64"
    tensors_gpu = [tensor.to(device, dtype=torch.complex64) for tensor in tensors]
    stemtensor = [torch.empty([2**32], dtype = torch.cfloat, device = device), torch.empty([2**32], dtype = torch.cfloat, device = device)]

kwargs["alpha"] = 1 # cutensor contraction 的 alpha
if world_rank == 0:
    print(f"warmup: {args.warmup}\ndata type of calculation: {typeCal}\nis_scale: {args.is_scale}\ndata type of allToall communication: {typecom}\nautotune: {args.autotune}\nntask: {args.ntask}\n", flush = True)

###################################################################################################################
class MgTensor:  
    def __init__(self, sgtensor, mgmodes, convert = False):
        self.pcurtensor = 0
        self.node_rank = node_rank
        self.node_world_size = node_world_size
        self.subtask_rank = subtask_rank
        self.subtask_world_size = subtask_world_size
        
        assert type(sgtensor) == torch.Tensor
        if convert:
            self.shape = sgtensor.shape
            self.curtensor = sgtensor
        else:
            sgtensor = sgtensor.flatten(end_dim = mgmodes)
            sgtensor = torch.chunk(sgtensor, 2**mgmodes)[self.subtask_rank]
            self.shape = sgtensor.shape
            self.curtensor = sgtensor
    
    def einsum(self, nstep, ein, insg2, task_id, mnmodes, mgmodes, **kwargs):
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
            if "half" in typeCom:
                int_com.half_communicate(task_id, nstep, self, group, **kwargs)
            elif "int8" in typeCom:
                int_com.int8_communicate(task_id, nstep, self, group, **kwargs)
            elif typeCom == "int4kernel":
                groupsize = args.int4group
                if world_rank==0:
                    print(f"group size {groupsize}", flush=True)
                int_com.int4_communicate(task_id, nstep, self, group, groupsize, **kwargs)
            else:
                self.shape = self.curtensor.shape
                rawtensor = self.curtensor.flatten(end_dim = mgmodes-1)
                newmgtensor = torch.empty_like(rawtensor)
                dist.all_to_all_single(newmgtensor, rawtensor, group = group)
                self.curtensor = newmgtensor.view(self.shape)

        elif ein_list[0][mnmodes:mgmodes] != ein_list[2][mnmodes:mgmodes]:
            # if world_rank == 0:
            #     print(f"nstep {nstep}，节点内 ein {ein}", flush = True)
            group = node_gps[node_idx]
            rawtensor = self.curtensor.flatten(end_dim = mgmodes-mnmodes-1)
            newmgtensor = torch.empty_like(rawtensor)
            dist.all_to_all_single(newmgtensor, rawtensor, group = group)
            self.curtensor = newmgtensor.view(self.shape)
        # if world_rank == 0:
        #     print(f"nstep {nstep}, mgmodes {mgmodes}, ein {ein}, ein_new {ein_new}", flush = True)
        EinsumGeneralV2_choose_method(nstep, self, ein_new, insg2, **kwargs)
    
    def flatsplit(self, flat, split, chunks = split // subtask_world_size, mgmodes = mgmodes):
        # Here I assumed that the tensors won't be larger than the splited tensor
        # It may won't work in other tensor network
        torch.cuda.empty_cache()
        self.nexttensor = torch.empty_like(self.curtensor)
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
    # torch.cuda.empty_cache()
    mgtensor.curtensor = utils.torch_einsum(ein, mgtensor.curtensor, tensor_j, **kwargs)



def cont_nsch_split(tensors, nsch, task_id, **kwargs):
    for nstep, step in enumerate(nsch):
        # if world_rank == 0:
        #     print(f"nstep {nstep}", flush = True)
        with record_function(f"step{nstep}"):
            i, j = step['index']

            if scale_class.is_scale:
                kwargs["alpha"] = scale_class.get_scale(task_id, nstep, tensors[i], tensors[j], **kwargs)

            if step['type'] == 1:
                flati, flatj = step['flat']
                # print(step['flat'])
                if flati > 1:
                    assert type(tensors[i]) == MgTensor    
                    tensors[i] = tensors[i].flatsplit(flati, split) # convert MgTensor -> MgTensorSplit
                if flatj > 1:
                    lenj = len(tensors[j].shape)
                    tensors[j] = tensors[j].view([-1] + [2] * (lenj - flatj))
                chubi, chubj = step['chunk_batch']
                
                for chuindex in range(tensors[i].chunks):
                    pi = tensors[i].curtensors[chuindex][chubi[chuindex + split // subtask_world_size * subtask_rank]]
                    pj = tensors[j][chubj[chuindex + split //subtask_world_size * subtask_rank]]

                    out_tensor = utils.torch_einsum(step['ein_2'], pi, pj, **kwargs)
                    newshape = out_tensor.shape
                    tensors[i].setnewtensor(chuindex, newshape)
                    tensors[i].nexttensors[chuindex].view(-1)[:out_tensor.numel()].copy_(out_tensor.view(-1)) 

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
                        out_tensor = utils.torch_einsum(step['ein_2'], pi, pj, **kwargs)
                        newshape = out_tensor.shape
                        tensors[i].setnewtensor(chuindex, newshape)
                        tensors[i].nexttensors[chuindex].view(-1)[:out_tensor.numel()].copy_(out_tensor.view(-1))
                    tensors[i].swap_tensors()   
                elif 'reorder_ein' in step:
                    if type(tensors[i]) == torch.Tensor:
                        tensors[i] = utils.torch_einsum(step['reorder_ein'], tensors[i], tensors[j], **kwargs)
                        tensors[i] = MgTensor(tensors[i], mgmodes)
                    else:
                        assert type(tensors[i]) == MgTensor
                        tensors[i].einsum(nstep, step['reorder_ein'], tensors[j], task_id, args.use_int8, **kwargs)
                else:
                    # torch.cuda.empty_cache()
                    tensors[i] = utils.torch_einsum(step['ein_2'],tensors[i], tensors[j], **kwargs)
                    
                tensors[j] = []
            if world_rank == 0:
                if kwargs['alpha'] != 1:
                    print(f"kwargs['alpha'] {kwargs['alpha']}", flush=True)

    if type(tensors[i]) == utils.MgTensorSplit:
        return torch.cat([x for x in tensors[i].curtensors])
    else:
        return torch.cat([x.curtensor for x in tensors[i]])


def calc_task(s, **kwargs):
    configs = list(map(int, np.binary_repr(s, len(slicing_edges))))
    sliced_tensors = tensors_gpu.copy()
    for x in range(len(slicing_edges)):
        m, n = slicing_edges[x]
        idxm_n, idxn_m = slicing_indices[(m, n)]
        sliced_tensors[m] = sliced_tensors[m].select(idxm_n, configs[x]).clone()
        sliced_tensors[n] = sliced_tensors[n].select(idxn_m, configs[x]).clone()
    
    ans = cont_nsch_split(sliced_tensors, nsch, s, **kwargs)
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
ans = calc_task(task_id, **kwargs)[0]
torch.cuda.synchronize()
time_end = time.time()
if world_rank == 0:
    print(f"warm up used time {round(time_end-time_begin, 3)}s", flush = True)
kwargs["autotune"] = False # remember to close autotune
del ans; torch.cuda.empty_cache()
############## Only profile one task #######
ntask = args.ntask
with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,],
        ) as prof:
    torch.cuda.synchronize()
    task_id = 0 + subtask_idx * ntask
    ans = calc_task(task_id, **kwargs)
    torch.cuda.synchronize()
del ans; torch.cuda.empty_cache()
################### Save profiler #####################
if world_rank == 0:
    prof.export_chrome_trace(f"{trace_path}/ntask{ntask*subtasks-1}_CAL{typeCal}_COM{typecom}_TUNE{args.autotune}_Nodes{int(os.environ['nnodes'])}.json")
    
################### PERFORM TrueTask #########################################
# Supervise energy consumption
import multiprocessing
from utils import monitor_gpu_power

if world_rank == 0 and not os.path.exists(f"{result_path}/ntask{ntask*subtasks-1}"):
    os.makedirs(f"{result_path}/ntask{ntask*subtasks-1}")
dist.barrier()

torch.cuda.synchronize()
time_begin = time.time()
stop_event = multiprocessing.Value('b', False)
process = multiprocessing.Process(target=monitor_gpu_power, args=(stop_event, node_idx, node_rank, trace_path))
process.start()
for s in range(ntask):
    task_id = s + subtask_idx * ntask
    if s == 0:
        ans = calc_task(task_id, **kwargs)[0]
    else:
        ans += calc_task(task_id, **kwargs)[0]
stop_event.value = True
process.join()
torch.cuda.synchronize()
time_end = time.time()
dist.barrier()
total_time = torch.tensor([time_end-time_begin]).to(device)
dist.all_reduce(total_time, dist.ReduceOp.MAX)
################### Calculate energy coonsumption ###############################
energy = utils.cal_energy(world_size//node_world_size, node_world_size, trace_path)
if world_rank == 0:
    print(f"\n\n\n\n***************** Results *******************", flush = True)
    print(f"Profile saved to {trace_path}/ntask{ntask*subtasks-1}_CAL{typeCal}_COM{typecom}_TUNE{args.autotune}_Nodes{int(os.environ['nnodes'])}.json", flush = True)
    print(f"energy information saved to {trace_path}/energy/", flush=True)
    print(f"total consumption {energy} kwh", flush=True)
    print(f"Truetask used time {round(total_time[0].item(), 3)} s", flush = True)
    print(f"torch.memory.allocated {torch.cuda.memory_allocated()/2**30} G, torch.memory.reserved {torch.cuda.memory_reserved()/2**30} G", flush = True)

######################### Reduce answer ########################################
del stemtensor; torch.cuda.empty_cache()
ans = scale_class.rescale(task_id, ans.to(dtype=torch.complex64),  **kwargs)
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
