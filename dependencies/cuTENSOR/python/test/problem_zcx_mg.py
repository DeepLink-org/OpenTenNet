import torch
import numpy as np
from copy import deepcopy
from os.path import exists, dirname, abspath
from os import makedirs
import multiprocessing as mp
from math import ceil
import time
from cutensor.torch import EinsumGeneral, TensorMg, toTensor, fromTensor, init, getOutputShapeMg, einsumMgV2

def tensor_to_mg(tensor:torch.Tensor):
    t_mg = TensorMg(tensor.shape)
    assert fromTensor(t_mg, tensor)
    return t_mg

def tensormg_to_Tensor(t_mg:TensorMg, dtype=torch.complex64):
    res = torch.empty(t_mg.getShape(), dtype=dtype)
    assert toTensor(res, t_mg)
    return res

def reshape_mg(t_mg:TensorMg, shape):
    t_cpu = tensormg_to_Tensor(t_mg)
    return tensor_to_mg(t_cpu.reshape(shape))

def my_einsum_mg(subscripts:str, input_0:TensorMg, input_1:TensorMg, origin:torch.Tensor):
    shape = getOutputShapeMg(subscripts, input_0, input_1)
    ans_mg = TensorMg(shape)
    assert einsumMgV2(subscripts, input_0, input_1, ans_mg, origin)
    return ans_mg

def tensor_contraction_sparse(tensors, contraction_scheme, use_cutensor=True) -> torch.Tensor:
    '''
    contraction the tensor network according to contraction scheme

    :param tensors: numerical tensors of the tensor network
    :param contraction_scheme: 
        list of contraction step, defintion of entries in each step:
        step[0]: locations of tensors to be contracted
        step[1]: einsum equation of this tensor contraction
        step[2]: batch dimension of the contraction
        step[3]: optional, if the second tensor has batch dimension, 
            then here is the reshape sequence
        step[4]: optional, if the second tensor has batch dimension, 
            then here is the correct reshape sequence for validation

    :return tensors[i]: the final resulting amplitudes
    '''
    origin = tensors[0]
    tensors_mg = [tensor_to_mg(tensor) for tensor in tensors]
    tensors = tensors_mg
    if use_cutensor:
        # einsum_func = EinsumGeneral
        einsum_func = my_einsum_mg
    else:
        einsum_func = torch.einsum

    for step in contraction_scheme:
        i, j = step[0]
        batch_i, batch_j = step[2]
        if len(batch_i) > 1:
            tensors[i] = [tensors[i]]
            for k in range(len(batch_i)-1, -1, -1):
               
                # if torch.is_tensor(batch_i[k]) and batch_i[k].device.type == 'cpu':
                #     batch_i[k] = batch_i[k].to('cuda:0')
                t_cpu = tensormg_to_Tensor(tensors[i][0])
                param1 = tensor_to_mg(torch.index_select(t_cpu, 0, batch_i[k]))
                t_cpu = tensormg_to_Tensor(tensors[j])
                param2 = tensor_to_mg(t_cpu[batch_j[k]])
                
                if k != 0:
                    if step[3]:
                        ans_mg = einsum_func(step[1], param1, param2, origin)
                        tensors[i].insert(
                            1, 
                            reshape_mg(ans_mg, step[3])
                        )
                    else:
                        ans = einsum_func(
                                step[1],
                                param1,
                                param2, 
                                origin)
                        tensors[i].insert(
                            1, 
                            ans
                        )
                else:
                    if step[3]:
                     
                        ans_mg = einsum_func(
                            step[1],
                            param1, 
                            param2, origin
                        )
                        tensors[i][0] = reshape_mg(ans_mg, step[3])
                    else:
                        tensors[i][0] = einsum_func(
                            step[1],
                            param1, 
                            param2, origin
                        )
                     
                # del param1
                # del param2
            tensors[j] = []
            temp_tensor_cpu = [tensormg_to_Tensor(tensor_mg) for tensor_mg in tensors[i]]
            tensors[i] = tensor_to_mg(torch.cat(temp_tensor_cpu, dim=0))
            # tensors[i] = torch.cat(tensors[i], dim=0)
        elif len(step) > 3 and len(batch_i) == len(batch_j) == 1:
            t_i_cpu = tensormg_to_Tensor(tensors[i])
            tensors[i] = tensor_to_mg(t_i_cpu[batch_i[0]])

            t_j_cpu = tensormg_to_Tensor(tensors[j])
            tensors[j] = tensor_to_mg(t_j_cpu[batch_j[0]])
            
            # tensors[i] = tensors[i][batch_i[0]]
            # tensors[j] = tensors[j][batch_j[0]]
            tensors[i] = einsum_func(step[1], tensors[i], tensors[j], origin)
        elif len(step) > 3:
            temp_tensor_mg = einsum_func(
                step[1],
                tensors[i],
                tensors[j], origin
            )
            tensors[i] = reshape_mg(temp_tensor_mg, step[3])
            if len(batch_i) == 1:
                tensor_cpu = tensormg_to_Tensor(tensors[i])
                # tensors[i] = tensors[i][batch_i[0]]
                tensors[i] = tensor_to_mg(torch.index_select(tensor_cpu, 0, batch_i[0]))
            tensors[j] = []
        else:
            # shape = getOutputShape(step[1], tensors[i], tensors[j])
            # ans = torch.empty(shape, device='cuda:0', dtype=tensors[i].dtype)
            # torch.cuda.synchronize()
            # start = time.perf_counter()
            # EinsumGeneralV2(ans, step[1], tensors[i], tensors[j])
            # torch.cuda.synchronize()
            # end = time.perf_counter()
            # increase_total_get_tensor_time(end - start)
            # print(tensors[i].dtype)
            # print(tensors[i].numel())
            # print(tensors[i].shape)
            # print(tensors[i].element_size())
            # quit()
            
                # input_i = tensors[i].to('cpu')
                # input_j = tensors[j].to('cpu')
                # mg_i = TensorMg(input_i.shape)
                # if not fromTensor(mg_i, input_i):
                #     print("input_i error")
                # mg_j = TensorMg(input_j.shape)
                # if not fromTensor(mg_j, input_j):
                #     print("input_j error")
                
                # mg_ans = einsumMg(step[1], mg_i, mg_j, input_i)
                # y = torch.empty(shape, dtype=input_i.dtype)
                # if not toTensor(y, mg_ans):
                #     print("toTensor error")
                #     quit()
                # if not torch.allclose(y, ans.to('cpu'), atol=1e-05, rtol=1e-02):
                #     print("failed-----")
                #     print(f"step[1] = {step[1]}")
                #     print(f"tensors[i] = {tensors[i].shape}")
                #     print(f"tensors[j] = {tensors[j].shape}")
                #     print(f"y = {y.shape}")
                #     print(f"ans = {ans.shape}")
                #     quit()
                # else:
                #     print("einsumMg success.")
                #     quit()
                # del mg_i
                # del mg_j
                # del mg_ans
                # del y
            # torch.cuda.synchronize()
            # start = time.perf_counter()
            # print(type())
            ans = einsum_func(step[1], tensors[i], tensors[j], origin)
            # torch.cuda.synchronize()
            # end = time.perf_counter()
            # incrase_total_einsum_time(end - start)

            # del ans
            # del myOutput
            tensors[i] = ans
            tensors[j] = []

    return tensormg_to_Tensor(tensors[i])


def contraction_single_task(
        tensors:list, scheme:list, slicing_indices:dict, 
        task_id:int, device='cuda:0'
    ):
    store_path = abspath(dirname(__file__)) + '/results/'
    if not exists(store_path):
        try:
            makedirs(store_path)
        except:
            pass
    file_path = store_path + f'partial_contraction_results_{task_id}.pt'
    time_path = store_path + f'result_time.txt'
    n_sub_task = 2 ** 1 # subtask number of each task
    if not exists(file_path) or True:
        t0 = time.perf_counter()
        slicing_edges = list(slicing_indices.keys())
        # tensors_gpu = [tensor.to(device) for tensor in tensors]
        for s in range(task_id * n_sub_task, (task_id + 1) * n_sub_task):
            configs = list(map(int, np.binary_repr(s, len(slicing_edges))))
            sliced_tensors = tensors.copy()
            for x in range(len(slicing_edges)):
                m, n = slicing_edges[x]
                idxm_n, idxn_m = slicing_indices[(m, n)]
                sliced_tensors[m] = sliced_tensors[m].select(idxm_n, configs[x]).clone()
                sliced_tensors[n] = sliced_tensors[n].select(idxn_m, configs[x]).clone()
            if s == task_id * n_sub_task:
                collect_tensor = tensor_contraction_sparse(sliced_tensors, scheme)
            else:
                collect_tensor += tensor_contraction_sparse(sliced_tensors, scheme)
        t1 = time.perf_counter()
        torch.save(collect_tensor.cpu(), file_path)
        with open(time_path, 'a') as f:
            f.write(f'task id {task_id} running time: {t1-t0:.4f} seconds\n')
        print(f'subtask {task_id} done, the partial result file has been written into results/partial_contraction_results_{task_id}.pt')
    else:
        print(f'subtask {task_id} has already been calculated, skip to another one.')


def collect_results(task_num):
    for task_id in range(task_num):
        file_path = abspath(dirname(__file__)) + f'/results/partial_contraction_results_{task_id}.pt'
        if task_id == 0:
            collect_result = torch.load(file_path)
        else:
            collect_result += torch.load(file_path)
    
    return collect_result


def write_result(bitstrings, results):
    amplitude_filename = abspath(dirname(__file__)) + f'/results/result_amplitudes.txt'
    xeb_filename = abspath(dirname(__file__)) + f'/results/result_xeb.txt'
    time_filename = abspath(dirname(__file__)) + f'/results/result_time.txt'
    with open(amplitude_filename, 'w') as f:
        for bitstring, amplitude in zip(bitstrings, results):
            f.write(f'{bitstring} {np.real(amplitude)} {np.imag(amplitude)}j\n')
    with open(xeb_filename, 'w') as f:
        f.write(f'{results.abs().square().mean().item() * 2 ** 53 - 1:.4f}')
    with open(time_filename, 'r') as f:
        lines = f.readlines()
    time_all = sum([float(line.split()[5]) for line in lines])
    with open(time_filename, 'a') as f:
        f.write(f'overall running time: {time_all:.2f} seconds.\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        "-taskid", type=int, default=0, 
        help="tensor network contraction id"
    )
    parser.add_argument(
        "-device", type=int, default=0, 
        help="using which device, -1 for cpu, otherwise use cuda:device"
    )
    args = parser.parse_args()
    assert args.device >= -1
    
    init(torch.cuda.device_count())
   



    args.device = 'cuda:0'
    contraction_filename = abspath(dirname(__file__)) + '/80G_scheme_n53_m20.pt'
    if not exists(contraction_filename):
        assert ValueError('No contraction data!')

    """
    There will be four objects in the contraction scheme:
        tensors: Numerical tensors in the tensor network
        scheme: Contraction scheme to guide the contraction of the tensor network
        slicing_indices: Indices to be sliced, the whole tensor network will be
            divided into 2**(num_slicing_indices) sub-pieces and the contraction of
            all of them returns the overall result. The indices is sliced to avoid
            large intermediate tensors during the contraction.
        bitstrings: bitstrings of interest, the contraction result will be amplitudes
            of these bitstrings
    """
    tensors, scheme, slicing_indices, bitstrings = torch.load(contraction_filename)
    task_num = 1
    
    contraction_single_task(tensors, scheme, slicing_indices, args.taskid, args.device)

    # file_exist_flag = True
    # for i in range(task_num):
    #     if not exists(abspath(dirname(__file__)) + f'/results/partial_contraction_results_{i}.pt'):
    #         file_exist_flag = False
    # if file_exist_flag:
    #     print('collecting results, results will be written into results/result_*.txt')
    #     results = collect_results(task_num)
    #     write_result(bitstrings, results)
