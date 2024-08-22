import os
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
import re
import argparse
import time
import subprocess
import math
import Quant
# int4 sacle zeros
def FHtoInt4(mgtensor,k,groupsize):
    in_tensor = mgtensor.curtensor
    n = int(torch.log2(torch.tensor(in_tensor.numel())).item()) + 1
    nscales = 2**(n-k)
    nzeros = 2**(n-k)
    nint8 = 2**(n-1)
    nexttensor = torch.view_as_real(mgtensor.nexttensor).flatten()
    scales = nexttensor[:nscales]
    zeros = nexttensor[nscales:nzeros+nscales]
    uint8_out = nexttensor[nzeros+nscales:].view(torch.uint8)[: nint8]
    Quant.FHtoInt4(torch.view_as_real(in_tensor).view(-1), n, scales, zeros, uint8_out, groupsize)
    scales.view([2]*(n-k))
    zeros.view([2]*(n-k))
    uint8_out.view([2]*(n-1))
    return scales, zeros, uint8_out, n

def Int4ToFH(scales, zeros, uint8_out, n, mgtensor,groupsize):
    scales.view(-1)
    zeros.view(-1)
    uint8_out.view(-1)
    nexttensor = torch.view_as_real(mgtensor.nexttensor)
    out_tensor = nexttensor.flatten()[:2**n]
    Quant.Int4ToFH(scales, zeros, uint8_out, n, out_tensor,groupsize)
    out_tensor = torch.view_as_complex(out_tensor.view(-1,2)).view([2]*(n-1))
    return out_tensor
# int4 scale zeros

class _pairwiseTo1n():
    def __init__(self):
        self.groups = None
        self.nnodes = int(os.environ["nnodes"])
        self.num = int(math.log(self.nnodes, 2))
        self.gpus = torch.cuda.device_count()

    def make_group(self):
        self.groups = {}
        self.group_lists = {}
        pairwise_groups = self.nnodes
        for iter in range(self.num):
            pairwise_groups = int(pairwise_groups/2)
            group_list = []; group = []
            for n in range(pairwise_groups ):
                for gpu in range(self.gpus):
                    group_list.append([n* self.gpus + gpu, (n+pairwise_groups)*self.gpus + gpu])
                    group.append(dist.new_group([n* self.gpus + gpu, (n+pairwise_groups)*self.gpus + gpu]))
            self.group_lists[iter] = group_list
            self.groups[iter] = group

    def warmup(self, tensor_warm, rank, num_loop=1):
        for _ in range(num_loop):
            pairwise_groups = self.nnodes
            for iter in range(self.num):
                if rank < pairwise_groups*self.gpus:
                    dist.reduce(tensor_warm, rank % int(pairwise_groups*self.gpus/2), op=ReduceOp.SUM, \
                    group=self.groups[iter][rank % int(pairwise_groups*self.gpus/2)])
                pairwise_groups = int(pairwise_groups/2)
    
    def cal(self, tensor, rank):
        pairwise_groups = self.nnodes
        for iter in range(self.num):
            if rank < pairwise_groups*self.gpus:
                dist.reduce(tensor, rank % int(pairwise_groups*self.gpus/2), op=ReduceOp.SUM, \
                group=self.groups[iter][rank % int(pairwise_groups*self.gpus/2)])
            pairwise_groups = int(pairwise_groups/2)


class _nnTo1n():
    def __init__(self, subtask_gps, subtask_world_size):
        self.groups = None
        self.subtask_gps = subtask_gps
        self.swz = subtask_world_size

    def make_group(self):
        self.groups = []
        self.group_lists = []
        for i in range(self.swz):
            self.groups.append(dist.new_group([index for index in range(i, self.subtask_gps*self.swz, self.swz)]))
            self.group_lists.append([index for index in range(i, self.subtask_gps*self.swz, self.swz)])

    def warmup(self, tensor_warm, rank, num_loop=1):
        for _ in range(num_loop):
            self.cal(tensor_warm, rank)
    
    def cal(self, tensor, rank, op = ReduceOp.SUM):
        dist.reduce(tensor, rank%self.swz, op=op, group=self.groups[rank%self.swz])

        
class Communicate_quant:
    def __init__(self, subtask_rank, total_tasks, steps, device, outpath):
        self.data_path = "data"
        self.subtask_rank = subtask_rank
        self.file_exist = None      # predict whether the .pt is loaded or the .pt is exist
        self.new_info = True # whether new information is added
        self.output_path = outpath
        self.scales = torch.zeros([total_tasks, steps]).to(device)
        if os.path.exists(self.output_path):
            self.file_exist = True   # is loaded
            self.scales_prev = torch.load(self.output_path).to(device)
            prev_tasks = self.scales_prev.shape[0]
            if prev_tasks > total_tasks: # padding scales
                self.new_info = False
                self.scales = self.scales_prev
            

    def int4_communicate(self, task_id, nstep, mgtensor, group, groupsize, **kwargs):
        # int4 scale zeros
        k = int(math.log2(groupsize))   
        inshape = mgtensor.curtensor.shape
        scales_, zeros, uint8_out, n = FHtoInt4(mgtensor,k,groupsize) # nexttensor
        # should change curtensor and nexttensor
        mgtensor.setnewtensor(mgtensor.shape) # now, scales,zeros and uint8_out are in curtensor
        nscales = 2**(n-k)
        nzeros = 2**(n-k)
        nint8 = 2**(n-1)
        nexttensor = torch.view_as_real(mgtensor.nexttensor)
        scales_out = nexttensor.flatten()[:nscales]
        zeros_out = nexttensor.flatten()[nscales:nzeros+nscales]
        uint8_out_ = nexttensor[nzeros+nscales:].view(torch.uint8)[: nint8]
        # if kwargs["world_rank"] == 0:
        #     print(f"int4_communicate", flush = True)
        # all2all
        dist.all_to_all_single(scales_out, scales_, group = group)
        dist.all_to_all_single(zeros_out, zeros, group = group)
        dist.all_to_all_single(uint8_out_, uint8_out, group = group)
        # all2all
        mgtensor.setnewtensor(mgtensor.shape) # now, xxx_out are in curtensor
        out_tensor = Int4ToFH(scales_out, zeros_out, uint8_out_, n, mgtensor,groupsize)
        # mgtensor.nexttensor.view(-1)[:out_tensor.numel()].copy_(out_tensor.view(-1)) 
        mgtensor.setnewtensor(inshape)


    def half_communicate(self, task_id, nstep, mgtensor, group, **kwargs):
        typeCom = kwargs["typeCom"]
        pow_idx = 0
        rawtensor = mgtensor.curtensor
            
        if self.scales[task_id, nstep] == 0:
            max0 = max(torch.view_as_real(rawtensor).max().abs(), torch.view_as_real(rawtensor).min().abs())
            # max0.pow_(1./pow_idx)
            dist.all_reduce(max0, dist.ReduceOp.MAX, group = group)
            self.scales[task_id, nstep] = 65000./max0.item()
            # if kwargs["world_rank"] == 0:
            #     print(f"calculate half scale for communication", flush = True)

        scale_ = self.scales[task_id, nstep]
        nelem = mgtensor.curtensor.numel() # num elem
        if typeCom == "HalfKernel":
            Quant.Float2Half(torch.view_as_real(mgtensor.curtensor).view(-1), mgtensor.nexttensor.view(torch.half)[: nelem*2], pow_idx, scale_)
            
            dist.all_to_all_single(mgtensor.curtensor.view(torch.half).view(-1)[: nelem*2], mgtensor.nexttensor.view(torch.half)[: nelem*2], group = group)      
            
            mgtensor.setnewtensor(mgtensor.shape)
            Quant.Half2Float(mgtensor.nexttensor.view(torch.half)[: nelem*2], torch.view_as_real(mgtensor.curtensor), pow_idx, scale_)
            
        else:
            # rawtensor.pow_(1./pow_idx)
            rawtensor.mul_(scale_)
            mgtensor.nexttensor.view(torch.half)[: nelem*2].copy_(torch.view_as_real(rawtensor).view(-1))
            dist.all_to_all_single(mgtensor.curtensor.view(torch.half).view(-1)[: nelem*2], mgtensor.nexttensor.view(torch.half)[: nelem*2], group = group)
            
            torch.view_as_real(mgtensor.nexttensor).view(-1)[: nelem*2].copy_(mgtensor.curtensor.view(torch.half).view(-1)[: nelem*2])
            mgtensor.setnewtensor(mgtensor.shape)
            newtensor = mgtensor.curtensor
            newtensor.mul_(1./(scale_))
            # newtensor.pow_(pow_idx)
        
    def int8_communicate(self, task_id, nstep, mgtensor, group, **kwargs):
        
        typeCom = kwargs["typeCom"]
        pow_idx = 3.
        rawtensor = mgtensor.curtensor
            
        if self.scales[task_id, nstep] == 0:
            max0 = max(torch.view_as_real(rawtensor).max().abs(), torch.view_as_real(rawtensor).min().abs())
            max0.pow_(1./pow_idx)
            dist.all_reduce(max0, dist.ReduceOp.MAX, group = group)
            self.scales[task_id, nstep] = 127./max0.item()
            # if kwargs["world_rank"] == 0:
            #     print(f"calculate int8 scale", flush = True)

        scale_ = self.scales[task_id, nstep]
        nelem = mgtensor.curtensor.numel() # num elem

        if typeCom == "int8kernel":
            # mgtensor.nexttensor.view(torch.int8)[: nelem*2] # todo
            Quant.FHtoInt8(torch.view_as_real(mgtensor.curtensor).view(-1), mgtensor.nexttensor.view(torch.int8)[: nelem*2], pow_idx, scale_)
            
            dist.all_to_all_single(mgtensor.curtensor.view(torch.int8).view(-1)[: nelem*2], mgtensor.nexttensor.view(torch.int8)[: nelem*2], group = group)      
            
            mgtensor.setnewtensor(mgtensor.shape)
            Quant.Int8ToFH(mgtensor.nexttensor.view(torch.int8)[: nelem*2], torch.view_as_real(mgtensor.curtensor), pow_idx, scale_)
            
        else:
            rawtensor.pow_(1./pow_idx)
            rawtensor.mul_(scale_)
            mgtensor.nexttensor.view(torch.int8)[: nelem*2].copy_(torch.view_as_real(rawtensor).view(-1))
            dist.all_to_all_single(mgtensor.curtensor.view(torch.int8).view(-1)[: nelem*2], mgtensor.nexttensor.view(torch.int8)[: nelem*2], group = group)
            
            torch.view_as_real(mgtensor.nexttensor).view(-1)[: nelem*2].copy_(mgtensor.curtensor.view(torch.int8).view(-1)[: nelem*2])
            mgtensor.setnewtensor(mgtensor.shape)
            newtensor = mgtensor.curtensor
            newtensor.mul_(1./(scale_))
            newtensor.pow_(pow_idx)
            
    # To do: 需要保存每个子任务对应的scale，所以在保存时需要把所有子任务的scale gather 到一起再保存。
    def save_scale(self, reduce_job, **kwargs):
        subtask_rank = kwargs["subtask_rank"]
        typeCom = kwargs["typeCom"]
        if typeCom == "int8kernel":
            reduce_job.cal(self.scales, subtask_rank, op = dist.ReduceOp.SUM)
            dist.barrier()
            torch.cuda.synchronize()
            if kwargs["world_rank"] == 0 and self.new_info:   # in rank 0, if the .pt isn't loaded or isn't exist, we'll save the new .pt
                print(f"there is new info for Communicate_quant", flush = True)
                print(f"Scale for int8 communication saved to {self.output_path}", flush = True)
                torch.save(self.scales, self.output_path)

def get_MaterAddr(file_path, timeout = 60):
    start_time = time.time()  
    while not os.path.exists(file_path):  
        if time.time() - start_time > timeout:  
            break
        time.sleep(1) 
    if os.path.isfile(file_path):  
        with open(file_path, 'r') as f:
            os.environ["MASTER_ADDR"] = f.readline().strip()
    else:  
        raise ValueError("%s isn't a file! Please write your master address correctly!" % file_path)

def setup_distributed(backend="nccl", port=None):
    """
    Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    """
    num_gpus = torch.cuda.device_count()
    
    if "SLURM_JOB_ID" in os.environ:
        # if split the project in multiple srun
        if "num_srun" in os.environ and int(os.environ["num_srun"]) > 1:
            # rank = int(os.environ["SLURM_PROCID"])
            rank = int(os.environ["SLURM_PROCID"]) + int(os.environ["RANK_BASE"])
            world_size = int(os.environ["WORLD_SIZE"])
            node_list = os.environ["SLURM_NODELIST"]
            addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
            # specify master port
            if port is not None:
                os.environ["MASTER_PORT"] = str(port)
            elif "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29512"

            if "MASTER_ADDR" not in os.environ and os.environ["RANK_BASE"] == "0":
                os.environ["MASTER_ADDR"] = addr
                if rank == 0:
                    # 将MASTER_ADDR写入文件
                    with open(f"log/MASTER_ADDR_{os.environ['time']}.txt", 'w') as f:
                        f.write(addr)
            if "MASTER_ADDR" not in os.environ:
                file_path = f"log/MASTER_ADDR_{os.environ['time']}.txt"
                get_MaterAddr(file_path, timeout = 600)
        else: 
            rank = int(os.environ["SLURM_PROCID"])
            world_size = int(os.environ["SLURM_NTASKS"])
            node_list = os.environ["SLURM_NODELIST"]
            addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
            # specify master port
            if port is not None:
                os.environ["MASTER_PORT"] = str(port)
            elif "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29512"
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = addr
        
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
        os.environ["nnodes"] = str(int(world_size/torch.cuda.device_count()))

    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        os.environ["nnodes"] = str(int(world_size/torch.cuda.device_count()))

    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )

def remove_common_suffixes(s1, s2):
    index = 0
    while index < len(s1) and index < len(s2) and s1[-index-1] == s2[-index-1]:
        index += 1
    return s1[:-index], s2[:-index], index        

def make_communicate_group(warmup = True, **kwargs):
    world_size = kwargs["world_size"]
    subtask_world_size = kwargs["subtask_world_size"]
    node_world_size = kwargs["node_world_size"]
    device = kwargs["device"]
    subtask_idx = kwargs["subtask_idx"]
    node_idx = kwargs["node_idx"]
    subtasks = kwargs["subtasks"]
    node_rank = kwargs["node_rank"]
    world_rank = kwargs["world_rank"]
    reduce_job = _nnTo1n(subtasks, subtask_world_size)
    reduce_job.make_group()
    
    subtask_gps = {} # 子任务层的通讯组
    for st_idx in range(world_size // subtask_world_size): # 一共有多少个
        subtask_gps[st_idx] = dist.new_group([st_idx * subtask_world_size + i for i in range(subtask_world_size)])

    node_gps = {} # 节点层的通讯组
    for n_idx in range(world_size // node_world_size):
        node_gps[n_idx] = dist.new_group([n_idx * node_world_size + i for i in range(node_world_size)])

    # warm up all2all single
    if warmup:
        tensor_warm = (node_rank * torch.ones((1), dtype=torch.int8)).to(device)
        reduce_job.warmup(tensor_warm, world_rank)
        del tensor_warm

        input_warm = torch.arange(subtask_world_size).to(device); output_warm = torch.empty([subtask_world_size], dtype=torch.int64).to(device)
        # if int(os.environ["RANK"]) == 0:
        
        dist.all_to_all_single(output_warm, input_warm, group = subtask_gps[subtask_idx])
        dist.all_to_all_single(output_warm, input_warm, group = node_gps[node_idx])
        del input_warm, output_warm

    return reduce_job, subtask_gps, node_gps
