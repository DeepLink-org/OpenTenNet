import torch
import os

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
benckmark = f"results/16T/CALcomplexHalf_COMcomplex64_TUNE1/Nodes2/ntask0/cat_res.pt"
groundTruth = torch.load(benckmark).to(device)

result_path = f"results/16T/CALcomplexHalf_COMint4kernel_TUNE1/Nodes2/ntask0/cat_res.pt"
res = torch.load(result_path).to(device)


fidelity = (
    (groundTruth.conj() @ res.reshape(-1)).abs() /
    (groundTruth.abs().square().sum().sqrt() * res.abs().square().sum().sqrt())
).square().item()

print(f"similarity with benchmark: {round(fidelity, 8)}, diff max {(groundTruth-res).abs().max()}")

