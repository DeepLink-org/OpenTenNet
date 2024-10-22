import os
import torch
import torch.distributed as dist
import re
from cutensor.torch import EinsumGeneralV2, getOutputShape
from .unit_functions import remove_common_suffixes



class MgTensorSplit:
    '''
    correct order of operations:
    init() -> # split currtensor evenly
    [curtensor() -> setnewtensor() -> nexttensor() ->] # repeat times of 'chunks'
    swap_tensors()
    '''
    def __init__(self, curbuffer, nextbubber, flat, chunks, mgmodes):
        self.chunks = chunks

        self.curbuffer = curbuffer # one of stemtensors, 32G
        self.nextbubber = nextbubber # the other of stemtensors, 32G

        self.curtensors = list(torch.chunk(
                            self.curbuffer.flatten(end_dim=flat-mgmodes-1),
                            chunks)) # initialize evenly
        self.nexttensors = [None]*chunks # initialize as unkonwn amount

        self.pcurbuffer = 0 # pointer of buffers

        numels = [t.numel() for t in self.curtensors]
        self.pct = [0]+list(map(lambda i: sum(numels[:i+1]), range(len(numels)))) # pointor of current tensor. pct[i] is the begin pos of current tensor, pct[i+1] is the end pos of the current tensor
        self.pnt = [0]*(chunks+1) # pointor of next tensor. pnt[i] is the begin pos of the next tensor, pnt[i+1] is the end pos of the next tensor

        self.curshapes = [t.shape for t in self.curtensors] # shapes of current tensor 
        self.nextshapes = [None]*(chunks) # shapes of next tensor 
    # @property
    def curtensor(self, chunidx):
        nelem = torch.tensor(self.curshapes[chunidx]).prod().item()
        shape = self.curshapes[chunidx]
        return self.curtensors[chunidx].flatten()[:nelem].view(shape)
    
    # @property
    def nexttensor(self, chuindex):
        assert type(self.nexttensors[chuindex]) == torch.Tensor
        return self.nexttensors[chuindex]
    
    def swap_tensors(self):
        self.curbuffer, self.nextbubber = self.nextbubber, self.curbuffer
        self.curtensors, self.nexttensors = self.nexttensors, [None]*self.chunks
        self.pcurbuffer = 1-self.pcurbuffer
        self.pct, self.pnt = self.pnt, [0]*(self.chunks+1)
        self.curshapes, self.nextshapes = self.nextshapes, [None]*(self.chunks)

    def setnewtensor(self, chunidx, newshape):
        self.nextshapes[chunidx] = newshape
        numel = torch.tensor(newshape).prod().item()
        ptr1 = self.pnt[chunidx]
        ptr2 = ptr1 + numel
        self.pnt[chunidx+1] = ptr2
        self.nexttensors[chunidx] = self.nextbubber.flatten()[ptr1:ptr2].view(newshape)
    
    def abs_max(self):
        max0 = max([torch.view_as_real(t).max().abs() for t in self.curtensors])
        min0 = min([torch.view_as_real(t).min().abs() for t in self.curtensors])
        maxi = max(max0, min0)
        return maxi
    
class Scale_Class:
    def __init__(self, is_scale, total_tasks, steps, device, outpath):
        self.is_scale = is_scale
        self.file_exist = None  # predict whether the .pt is loaded or the .pt is exist
        self.new_info = False # whether new information is added
        self.output_path = outpath

        if os.path.exists(self.output_path):
            self.file_exist = True   # is loaded
            self.scales = torch.load(self.output_path).to(device)
        else:
            self.scales = torch.zeros([1, steps]).to(device)
                
                
    def calculate_scale(self, task_id, nstep, tensor_i, tensor_j, **kwargs):
        '''
        calculate only if current scale does not exist in self.scales
        '''
        subtask_gps = kwargs["subtask_gps"]
        subtask_idx = kwargs['subtask_idx']
        subtask_world_size = kwargs["subtask_world_size"]
        alpha = 1
        if isinstance(tensor_i, torch.Tensor):
            max0 = torch.view_as_real(tensor_i).max().abs()
            min0 = torch.view_as_real(tensor_i).min().abs()
            maxi = max(max0, min0)
    
        else:
            maxi = tensor_i.abs_max()

        dist.all_reduce(maxi, dist.ReduceOp.MAX, group = subtask_gps[subtask_idx])

        if subtask_world_size == 8:
            if maxi.item() < 10**-4:
                alpha = torch.ones_like(maxi)*10000
        else:
            if maxi.item() < 10**-3:
                alpha = 10**7

        self.scales[0, nstep] = alpha
        # if alpha != 1 and kwargs["subtask_rank"] == 0:
        #     print(f"maxi.item() {maxi.item()}, task_id {task_id}, subtask_idx {subtask_idx}", flush=True)

        return alpha

    def get_scale(self, task_id, nstep, tensor_i, tensor_j, **kwargs):
        if self.scales[0, nstep] != 0: # 第task_id 子任务的第nstep 的scale信息是否存在
            alpha = self.scales[0, nstep]
        else:
            self.new_info = True
            alpha = self.calculate_scale(task_id, nstep, tensor_i, tensor_j, **kwargs)
        return alpha

    def rescale(self, task_id, ans, **kwargs):
        if self.is_scale:
            total_scale = torch.prod(self.scales[0])
            ans.mul_(1./total_scale)
        return ans
    # To do: 需要保存每个子任务对应的scale，所以在保存时需要把所有子任务的scale gather 到一起再保存
    def save_scale(self, reduce_job, **kwargs):
        subtask_rank = kwargs["subtask_rank"]
        world_rank = kwargs["world_rank"]

        if self.is_scale and not self.file_exist and world_rank == 0:
            print(f"There is new info for Scale_Class", flush=True)
            print(f"Scale saved to {self.output_path}", flush = True)
            torch.save(self.scales, self.output_path)

def reduceAndCat(ans, reduce_job, **kwargs):
    subtask_rank = kwargs["subtask_rank"]
    subtask_world_size = kwargs['subtask_world_size']
    device = kwargs["device"]
    subtask_gps = kwargs["subtask_gps"]
    subtask_idx = kwargs["subtask_idx"]

    tensor_real = torch.view_as_real(ans)
    reduce_job.cal(tensor_real, subtask_rank)
    res_shapes = torch.zeros([subtask_world_size], dtype = int, device = device)
    res_shapes[subtask_rank] = tensor_real.numel()
    dist.all_reduce(res_shapes, dist.ReduceOp.SUM, group = subtask_gps[subtask_idx])

    # move all results to subtask_rank = 0
    cat_res = [torch.zeros(n.item(), device = device, dtype = tensor_real.dtype) for n in res_shapes]
    cat_res[subtask_rank] = tensor_real.flatten()
    cat_res = torch.cat(cat_res, dim = 0)
    dist.all_reduce(cat_res, dist.ReduceOp.SUM, group = subtask_gps[subtask_idx])
    cat_res = torch.view_as_complex(cat_res.view(-1, 2))
    return cat_res

def reduceAns(ans, reduce_job, **kwargs):
    subtask_rank = kwargs["subtask_rank"]

    tensor_real = torch.view_as_real(ans)
    reduce_job.cal(tensor_real, subtask_rank)
    ans = torch.view_as_complex(tensor_real)
    return ans