import torch
import numpy as np
from copy import deepcopy
from os.path import exists, dirname, abspath
from os import makedirs
# import multiprocessing as mp
# from math import ceil
import time
# import nvtx
from cutensor.torch import EinsumGeneral, EinsumGeneralV2, getOutputShape, \
                           TensorMg, toTensor, fromTensor, init, getOutputShapeMg, einsumMgV2
# from torch.profiler import profile, record_function, ProfilerActivity
# torch.set_printoptions(edgeitems=5)
# torch.set_printoptions(profile="default")

total_get_tensor_time = 0

def zero_total_get_tensor_time():
    global total_get_tensor_time
    total_get_tensor_time = 0

def increase_total_get_tensor_time(t):
    global total_get_tensor_time
    total_get_tensor_time += t

get_tensor_cnt = 0
def zero_get_tensor_cnt():
    global get_tensor_cnt
    get_tensor_cnt = 0

def increase_get_tensor_cnt():
    global get_tensor_cnt
    get_tensor_cnt += 1

total_einsum_time = 0
def zero_total_einsum_time():
    global total_einsum_time
    total_einsum_time = 0

def incrase_total_einsum_time(t):
    global total_einsum_time
    total_einsum_time += t

def tensor_to_mg(tensor:torch.Tensor):
    t_mg = TensorMg(tensor.shape)
    assert fromTensor(t_mg, tensor)
    return t_mg

def tensormg_to_Tensor(t_mg:TensorMg, dtype=torch.complex64):
    res = torch.empty(t_mg.getShape(), dtype=dtype, device=torch.device("cuda:0"))
    assert toTensor(res, t_mg)
    return res

def reshape_mg(t_mg:TensorMg, shape):
    t_cpu = tensormg_to_Tensor(t_mg)
    return tensor_to_mg(t_cpu.reshape(shape))

def my_einsum_mg(subscripts:str, input_0:TensorMg, input_1:TensorMg):
    shape = getOutputShapeMg(subscripts, input_0, input_1)
    ans_mg = TensorMg(shape)
    assert einsumMgV2(subscripts, input_0, input_1, ans_mg)
    return ans_mg

def tensor_contraction_sparse(tensors, contraction_scheme, use_cutensor=True):
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

    if use_cutensor:
        einsum_func = EinsumGeneral
    else:
        einsum_func = torch.einsum

    for step in contraction_scheme:
        i, j = step[0]
        batch_i, batch_j = step[2]
        if len(batch_i) > 1:
            tensors[i] = [tensors[i]]
            for k in range(len(batch_i)-1, -1, -1):
                start = time.perf_counter()
                if torch.is_tensor(batch_i[k]) and batch_i[k].device.type == 'cpu':
                    batch_i[k] = batch_i[k].to('cuda:0')
                
                param1 = torch.index_select(tensors[i][0], 0, batch_i[k])
                # param1 = tensors[i][0][batch_i[k]]
                # end = time.perf_counter()
                # increase_total_get_tensor_time(end - start)
                param2 = tensors[j][batch_j[k]]
                end = time.perf_counter()
                # increase_total_get_tensor_time(end - start)
                
                if k != 0:
                    if step[3]:
                        s0 = time.perf_counter()
                        ans = einsum_func(
                                step[1],
                                param1, 
                                param2, 
                            )
                        print("test EinsumGeneralV2")
                        shape = []
                        getOutputShape(shape, step[1], param1, param2)
                        myOutput = torch.empty(shape, param1.options())
                        EinsumGeneralV2(myOutput, step[1], param1, param2)
                        assert torch.allclose(ans, myOutput)
                        
                        tensors[i].insert(
                            1, 
                            ans.reshape(step[3])
                        )
                        # incrase_total_einsum_time(time.perf_counter() - s0)
                    else:
                        s0 = time.perf_counter()
                        ans = einsum_func(
                                step[1],
                                param1,
                                param2, 
                                )
                        
                        tensors[i].insert(
                            1, 
                            ans
                        )
                        # incrase_total_einsum_time(time.perf_counter() - s0)
                else:
                    if step[3]:
                        s0 = time.perf_counter()
                        ans = einsum_func(
                            step[1],
                            param1, 
                            param2, 
                        )
                        
                        tensors[i][0] = ans.reshape(step[3])
                        # incrase_total_einsum_time(time.perf_counter() - s0)
                    else:
                        s0 = time.perf_counter()
                        tensors[i][0] = einsum_func(
                            step[1],
                            param1, 
                            param2, 
                        )
                        # incrase_total_einsum_time(time.perf_counter() - s0)
                del param1
                del param2
            tensors[j] = []
            tensors[i] = torch.cat(tensors[i], dim=0)
        elif len(step) > 3 and len(batch_i) == len(batch_j) == 1:
            # start = time.perf_counter()
            tensors[i] = tensors[i][batch_i[0]]
            tensors[j] = tensors[j][batch_j[0]]
            # end = time.perf_counter()
            # increase_total_get_tensor_time(end - start)
            s0 = time.perf_counter()
            tensors[i] = einsum_func(step[1], tensors[i], tensors[j])
            # incrase_total_einsum_time(time.perf_counter() - s0)
        elif len(step) > 3:
            s0 = time.perf_counter()
            tensors[i] = einsum_func(
                step[1],
                tensors[i],
                tensors[j],
            ).reshape(step[3])
            # incrase_total_einsum_time(time.perf_counter() - s0)
            if len(batch_i) == 1:
                
                if torch.is_tensor(batch_i[0]) and batch_i[0].device.type == 'cpu':
                    batch_i[0] = batch_i[0].to('cuda:0')
                
                
                # tensors[i] = tensors[i][batch_i[0]]
                tensors[i] = torch.index_select(tensors[i], 0, batch_i[0])
               
            tensors[j] = []
        else:
            num = 120000
            shape = getOutputShape(step[1], tensors[i], tensors[j])
            ans = torch.empty(shape, device='cuda:0', dtype=tensors[i].dtype)
            if tensors[i].numel() > num and tensors[j].numel() > num:
                torch.cuda.synchronize()
                start = time.perf_counter()
            EinsumGeneralV2(ans, step[1], tensors[i], tensors[j])
            if tensors[i].numel() > num and tensors[j].numel() > num:
                torch.cuda.synchronize()
                end = time.perf_counter()
                print(f"single gpu cost:{end-start}")
                # increase_total_get_tensor_time(end - start)
            # print(tensors[i].dtype)
            # print(tensors[i].numel())
            # print(tensors[i].shape)
            # print(tensors[i].element_size())
            # quit()
            if tensors[i].numel() > num and tensors[j].numel() > num:
                print(step[1])
                print(tensors[i].shape)
                print(tensors[j].shape)
                input_i = tensors[i].to('cpu')
                input_j = tensors[j].to('cpu')
                mg_i = TensorMg(input_i.shape)
                if not fromTensor(mg_i, input_i):
                    print("input_i error")
                mg_j = TensorMg(input_j.shape)
                if not fromTensor(mg_j, input_j):
                    print("input_j error")
                torch.cuda.synchronize()
                start = time.perf_counter()
                mg_ans = my_einsum_mg(step[1], mg_i, mg_j, input_i)
                torch.cuda.synchronize()
                end = time.perf_counter()
                print(f"multi gpu cost:{end-start}")

                # incrase_total_einsum_time(end - start)
                y = torch.empty(shape, dtype=input_i.dtype)
                if not toTensor(y, mg_ans):
                    print("toTensor error")
                    quit()
                if not torch.allclose(y, ans.to('cpu'), atol=1e-05, rtol=1e-02):
                    print("failed-----")
                    print(f"step[1] = {step[1]}")
                    print(f"tensors[i] = {tensors[i].shape}")
                    print(f"tensors[j] = {tensors[j].shape}")
                    print(f"y = {y.shape}")
                    print(f"ans = {ans.shape}")
                    quit()
                else:
                    print("einsumMg success.")
                    quit()
                del mg_i
                del mg_j
                del mg_ans
                del y
            # torch.cuda.synchronize()
            # start = time.perf_counter()
            # ans = einsum_func(step[1], tensors[i], tensors[j])
            # torch.cuda.synchronize()
            # end = time.perf_counter()
            # incrase_total_einsum_time(end - start)

            # del ans
            # del myOutput
            tensors[i] = ans
            tensors[j] = []

    return tensors[i]


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
    n_sub_task = 2 ** 2 # subtask number of each task
    if not exists(file_path) or True:
        t0 = time.perf_counter()
        slicing_edges = list(slicing_indices.keys())
        tensors_gpu = [tensor.to(device) for tensor in tensors]
        for s in range(task_id * n_sub_task, (task_id + 1) * n_sub_task):
            print(f"taskid:{s} begin -------------------")
            
            configs = list(map(int, np.binary_repr(s, len(slicing_edges))))
            
            sliced_tensors = tensors_gpu.copy()
            

            
            for x in range(len(slicing_edges)):
                m, n = slicing_edges[x]
                idxm_n, idxn_m = slicing_indices[(m, n)]
                sliced_tensors[m] = sliced_tensors[m].select(idxm_n, configs[x]).clone()
                sliced_tensors[n] = sliced_tensors[n].select(idxn_m, configs[x]).clone()
            
            if s == task_id * n_sub_task:
                collect_tensor = tensor_contraction_sparse(sliced_tensors, scheme)
            else:
                # torch.cuda.synchronize()
                # begin = time.perf_counter()
                # with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                
                collect_tensor += tensor_contraction_sparse(sliced_tensors, scheme)
                # prof.export_chrome_trace('trace.json')
                # print(prof.key_averages(group_by_input_shape=True).table(
                    # sort_by="self_cuda_memory_usage", max_name_column_width=50, row_limit=20))
                # torch.cuda.synchronize()
                # end = time.perf_counter()
                # print(f"taskid:{s} end -------------------{end - begin}s")
                # break
            
        t1 = time.perf_counter()
        print(f'task id {task_id} running time: {t1-t0:.4f} seconds')
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
    
    # device = 'cuda:0'
    init(torch.cuda.device_count())
    # torch.set_printoptions(edgeitems=3)
    # torch.set_printoptions(threshold=102)

    x_cpu = torch.randn([2, 2, 2], dtype=torch.complex64, device=torch.device('cuda:0'))
    blockSize =   [1, 2, 2]
    deviceCount = [3 - b for b in blockSize]
    shape = torch.tensor(x_cpu.shape)
    blockSize = shape.clone()
    blockSize[:1] = 1
    deviceCount = 3 - blockSize
    x_mg = TensorMg(shape.tolist(), blockSize.tolist(), deviceCount.tolist())
    assert fromTensor(x_mg, x_cpu)
    ts = x_mg.getTensors()
    print(x_cpu)
    for t in ts:
        print(t)
    x_res = torch.randn([2, 2, 2], dtype=torch.complex64, device=torch.device('cuda:0'))
    toTensor(x_res, x_mg)
    print(torch.allclose(x_cpu, x_res))
    quit()
    
    x_cpu = torch.randn([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=torch.complex64)
    y_cpu = torch.randn([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=torch.complex64)
    
    equation = "XNPGeRAbKndLfMBhTQmcDiYEkHSVC,XQlaFgIJGHjUPWOTL->NeRAbKndfMBhmcDiYEklaFgIJjUWOSVC"
    # x_cpu = torch.randn([1024, 1024, 1024], dtype=torch.complex64)
    # y_cpu = torch.randn([1024, 1024, 1024], dtype=torch.complex64)
    # equation = "abj,bax->jx"
    
    print("test begin")

    for _ in range(1):
        print(f"3", flush=True)

        blockSize =   [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1]
        deviceCount = [3 - b for b in blockSize]
        x_mg = TensorMg(x_cpu.shape, blockSize, deviceCount)
        assert fromTensor(x_mg, x_cpu)
        blockSize =   [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        deviceCount = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        y_mg = TensorMg(y_cpu.shape, blockSize, deviceCount)
        assert fromTensor(y_mg, y_cpu)

        y1_cpu = tensormg_to_Tensor(y_mg)
        print(torch.allclose(y_cpu, y1_cpu))
        
        # x_mg = tensor_to_mg(x_cpu)
        # y_mg = tensor_to_mg(y_cpu)
        print(f"1", flush=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        shape = getOutputShapeMg(equation, x_mg, y_mg)
        blockSize =   [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        deviceCount = [3 - b for b in blockSize]
        ans_mg = TensorMg(shape, blockSize, deviceCount)
        assert fromTensor(ans_mg, torch.randn(shape, dtype=torch.complex64))
        assert einsumMgV2(equation, x_mg, y_mg, ans_mg)

        # ans_mg = my_einsum_mg(equation, x_mg, y_mg, x_cpu)
        torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"multi gpu cost:{end-start}", flush=True)
        mg_result_cpu = tensormg_to_Tensor(ans_mg)
        # del ans_mg
        del x_mg
        del y_mg

        x_cuda = x_cpu.to('cuda:1')
        y_cuda = y_cpu.to('cuda:1')
    # for _ in range(2):
        print(f"3", flush=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        ans = EinsumGeneral(equation, x_cuda, y_cuda)
        torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"single gpu cost:{end-start}")
        del x_cuda
        del y_cuda
    # del ans
        ans_cpu = ans.to('cpu')
        if not torch.allclose(mg_result_cpu, ans_cpu, atol=1e-03, rtol=1e-01):
            print(f"step[1] = {equation}")
            print(f"x = {x_cpu.shape}")
            print(f"y = {y_cpu.shape}")
            print(f"ans = {ans_cpu.shape}")
            print(f"mg_result_cpu = {mg_result_cpu.shape}")
            print(ans_cpu.reshape([2**32]))
            print(mg_result_cpu.reshape([2**32]))
            print("failed-----")
        else:
            print("einsumMg success.")
    quit()
    # x = torch.randn([40, 50, 60, 70], dtype=torch.complex64)
    # mg = TensorMg(x.shape)
    # fromTensor(mg, x)
    # y = torch.empty(x.shape, dtype=x.dtype)
    # toTensor(y, mg)
    # print(x)
    # print(y)
    # print(torch.allclose(x, y))
    # quit()
    # indices = torch.tensor([0, 2]).to(device)
    # res1 = torch.index_select(x, 0, indices)
    # res2 = x[indices]
    # print(res1)
    # print(res2)
    # assert torch.allclose(res1, res2)
    # quit()


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
    args.device = 'cuda:0'
    contraction_filename = abspath(dirname(__file__)) + '/80G_scheme_n53_m20.pt'
    if not exists(contraction_filename):
        assert ValueError('No contraction data!')

    zero_total_get_tensor_time()
    zero_total_einsum_time()
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
    # print(scheme)
    task_num = 1
    # for step in scheme:
    #     batch_i, batch_j = step[2]
    #     if len(batch_i) > 1:
    #         pass
    #     elif len(step) > 3 and len(batch_i) == len(batch_j) == 1:
    #         pass
    #     elif len(step) > 3:
    #         if len(batch_i) == 1:
    #             print(type(batch_i[0]))
    #     else:
    #         pass


    #     # for idx in range(len(batch_i)):
    #     print(type(batch_i[0]))
    # quit()
    contraction_single_task(tensors, scheme, slicing_indices, args.taskid, args.device)

    
    print(f"total_get_tensor_time = {total_get_tensor_time}")
    print(f"total_einsum_time = {total_einsum_time}")
    # file_exist_flag = True
    # for i in range(task_num):
    #     if not exists(abspath(dirname(__file__)) + f'/results/partial_contraction_results_{i}.pt'):
    #         file_exist_flag = False
    # if file_exist_flag:
    #     print('collecting results, results will be written into results/result_*.txt')
    #     results = collect_results(task_num)
    #     write_result(bitstrings, results)
