import torch
import argparse
import os

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
    chars = [chr(i) for i in range(ord('a'),ord('z')+1)] + [chr(i) for i in range(ord('A'),ord('Z')+1)]
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
        # warnings.warn("Buffer tensors not given, creating tensors")
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



def torch_einsum(equation, input_0, input_1, **kwargs):
    alpha = kwargs["alpha"] if "alpha" in kwargs.keys() else 1
    beta = kwargs['beta'] if "beta" in kwargs.keys() else 0
    dtype_ = kwargs.get("dtype_")

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
        output = torch.einsum(equation, torch.view_as_real(input_0), torch.view_as_real(buffer_tensors))
        output.mul_(alpha)
        
    else:
        output = torch.einsum(equation, input_0, input_1)
        output.mul_(alpha)
    
    return torch.view_as_complex(output).contiguous()


def remove_common_suffixes(s1, s2):
    index = 0
    while index < len(s1) and index < len(s2) and s1[-index-1] == s2[-index-1]:
        index += 1
    return s1[:-index], s2[:-index], index  

def prepare_nsch(nsch, split, device):
    for i in range(len(nsch)):
        if 'chunk_batch' in nsch[i].keys():
            assert len(nsch[i]['chunk_batch'][0]) == split
            assert len(nsch[i]['chunk_batch'][1]) == split
            for t in range(split):
                nsch[i]['chunk_batch'][0][t] = nsch[i]['chunk_batch'][0][t].to(device)
                nsch[i]['chunk_batch'][1][t] = nsch[i]['chunk_batch'][1][t].to(device)
    return nsch

def parse_args():
    # read the argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_select", type=int, default=0, help="0: _nnTo1n, 1: _pairwiseTo1n")
    parser.add_argument("--warmup", type=int, default=1, help="whether to warm up for nccl, 0: False, 1: True")
    parser.add_argument("--data_type", type=int, default=0, help="0: complex32, 1: complex64")
    parser.add_argument("--is_scale", type=int, default=1, help="0: not scale, 1: scale, when complex32, it'd be 1 to guarantee precision")
    parser.add_argument("--use_int8", type=int, default=1, help="0: False, 1: True")
    parser.add_argument("--autotune", type=int, default=1, help="0: False, 1: True")
    parser.add_argument("--ntask", type=int, default=1)
    parser.add_argument("--use_int8kernel", type=int, default=1)
    parser.add_argument("--train_com", type=int, default=0, help="0: truetask, 1: trainning algos and scales")
    parser.add_argument("--tensorNetSize", type=str, default="640G")
    parser.add_argument("--typeCom", type=str, default="complex32", help="optionals: int8, int8kernel, int4kernel, complex32")
    parser.add_argument("--int4group", type=int, default=128)
    args = parser.parse_args()
    return args

def getFilePath(args, prefix = "", **kwargs):
    typeCal = kwargs["typeCal"]
    typecom = kwargs["typeCom"]
    world_rank = kwargs["world_rank"]
    trace_root = f"prof_dir"
    path_stem = f"{args.tensorNetSize}/{prefix}CAL{typeCal}_COM{typecom}_TUNE{args.autotune}"

    trace_path = f"{trace_root}/{path_stem}/Nodes{int(os.environ['nnodes'])}/{os.environ['time']}" 
    if world_rank == 0:
        if not os.path.exists(trace_path):
            os.makedirs(trace_path)

    result_path = f"results/{path_stem}/Nodes{int(os.environ['nnodes'])}"
    data_root = "train/data"
    algo_path = f'{data_root}/{args.tensorNetSize}/{prefix}algo_dict_{typeCal}.pt'
    scale_path = f"{data_root}/{path_stem}"
    if world_rank == 0 and not os.path.exists(scale_path):
        os.makedirs(scale_path)

    # energy information
    node_rank = kwargs["node_rank"]
    node_idx = kwargs["node_idx"]
    if node_rank == 0 and not os.path.exists(f"{trace_path}/energy/node{node_idx}"):
        os.makedirs(f"{trace_path}/energy/node{node_idx}")
    
    return scale_path, algo_path, result_path, trace_path

def compareWithBenchmark(cat_res, args, ntask, bitstrings = None, fakebenchmark = None, **kwargs):
    device = kwargs["device"]
    world_rank = kwargs["world_rank"]
    subtask_rank = kwargs["subtask_rank"]
    subtask_idx = kwargs["subtask_idx"]
    subtasks = kwargs["subtasks"]

    if fakebenchmark is not None:
        groundTruth = torch.load(fakebenchmark).to(device).view(-1)
        fidelity = ((groundTruth.conj() @ cat_res.reshape(-1)).abs() / (groundTruth.abs().square().sum().sqrt() * cat_res.abs().square().sum().sqrt())).square().item()
        if world_rank == 0:
            print(f"Compared with complex64              : {round(fidelity, 8)}", flush=True)
        return

    if bitstrings is not None:
        res_keys = [int(b,2) for b in bitstrings]
        res_keys = torch.tensor(res_keys, dtype=torch.int64).to(device)
        res_keys_sorted, res_idx = torch.sort(res_keys)

    if args.tensorNetSize == "640G":
        benckmark = f"results/benchmark/{args.tensorNetSize}/rank{subtask_rank}_tune_False_ein_old_ntask_{ntask*subtasks}.pt"
        groundTruth = torch.load(benckmark).to(device).view(-1)
        cat_res = cat_res.view(-1)
        fidelity = ((groundTruth.conj() @ cat_res.reshape(-1)).abs() / (groundTruth.abs().square().sum().sqrt() * cat_res.abs().square().sum().sqrt())).square().item()
        if subtask_idx== 0:
            print(f"fidelity of task {ntask*subtasks-1} on rank{subtask_rank}: {round(fidelity, 6)}", flush = True)

    elif args.tensorNetSize == "2T":
        # compare with benchmark
        benckmark = f"results/benchmark/4T_gtdata_sorted.pt"
        groundTruth = torch.load(benckmark).to(device).view(-1)
        cat_res = cat_res[res_idx].view(-1)
        fidelity = ((groundTruth.conj() @ cat_res.reshape(-1)).abs() / (groundTruth.abs().square().sum().sqrt() * cat_res.abs().square().sum().sqrt())).square().item()
        if world_rank == 0:
            expected = 0.002
            print(f"fidelity of 4T               : {round(fidelity, 8)}", flush = True)
            print(f"expected fidelity(0.002)     : {round(expected, 8)}", flush = True)
            print(f"fidelity / expected          : {round(fidelity/expected, 5)}", flush = True)

    elif args.tensorNetSize == "16T":
        benckmark = f"results/benchmark/{args.tensorNetSize}/gtdata_sorted.pt"
        groundTruth = torch.load(benckmark).to(device).view(-1)
        cat_res = cat_res[res_idx].view(-1)
        fidelity = ((groundTruth.conj() @ cat_res.reshape(-1)).abs() / (groundTruth.abs().square().sum().sqrt() * cat_res.abs().square().sum().sqrt())).square().item()
        if world_rank == 0:
            expected = 0.002
            print(f"fidelity of 2T               : {round(fidelity, 8)}")
            print(f"expected fidelity(0.002)  : {round(expected, 8)}")
            print(f"fidelity/expected            : {round(fidelity/expected, 8)}")