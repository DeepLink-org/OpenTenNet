import torch
import os

import argparse
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
parser.add_argument("--tensorNetSize", type=str, default="640G")
args = parser.parse_args()

# data type of einsum calculation
typeCal = "complexFloat" if args.data_type else "complexHalf"
# data type of allToall communication
typecom = "int8" if args.use_int8 else typeCal
path_stem = f"{args.tensorNetSize}/CAL{typeCal}_COM{typecom}_TUNE{args.autotune}"
nnodes = 2
nodes_per_task = 2
result_path = f"results/{path_stem}/Nodes{nnodes}"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for ntask in [2]:
    ranks = 8*nodes_per_task
    pt = torch.load(f"{result_path}/ntask{ntask-1}/rank0.pt")
    # pt = torch.load("{}/rank0.pt".format(pt_source))
    res = pt.to(device)
    for i in range(1, ranks):
        print(f"read pt_source {result_path}/ntask{ntask-1}/rank{i}.pt")
        pt = torch.load(f"{result_path}/ntask{ntask-1}/rank{i}.pt").to(device)
        # pt = torch.load("{}/rank{}.pt".format(pt_source, i)).to(device)
        res = torch.cat((res, pt), dim = 0)

    benckmark = f"results/benchmark/{args.tensorNetSize}/ntask{ntask - 1}.pt"
    groundTruth = torch.load(benckmark).to(device)

    ###############################################################################
    ############# fidelity ########################################################

    fidelity = (
        (groundTruth.conj() @ res.reshape(-1)).abs() /
        (groundTruth.abs().square().sum().sqrt() * res.abs().square().sum().sqrt())
    ).square().item()
    print(f"fidelity of task {ntask}: {round(fidelity, 8)}, diff max {(groundTruth-res).abs().max()}")
# print(f"groundTruth.abs().square().sum() {groundTruth.abs().square().sum()}")
# print(f"res.abs().square().sum() {res.abs().square().sum()}")

