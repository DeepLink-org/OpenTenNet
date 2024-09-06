import torch
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--use_int8", type=int, default=1, help="0: False, 1: True")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for ntask in range(76,77):
    suzhongling_pt_source = "results/benchmark/2T/ntask{}".format(ntask - 1)
    pt = torch.load("{}/rank0.pt".format(suzhongling_pt_source))
    groundTruth = pt.to(device)
    for i in range(1, 128):
        pt = torch.load("{}/rank{}.pt".format(suzhongling_pt_source, i)).to(device)
        groundTruth = torch.cat((groundTruth, pt), dim = 0)
    torch.save(groundTruth.cpu(), f"results/benchmark/2T/ntask{ntask - 1}.pt")

