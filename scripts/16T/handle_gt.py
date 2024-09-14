import re
import torch
import torch.distributed as dist
import csv
device = torch.device("cpu")

#### DEAL WITH truetask ######
cont_file = 'TensorNetwork/32T/sc41_scheme_n53_m20_1646.pt'
tuples = torch.load(cont_file)

bitstrings = tuples[3]
res_keys = [int(b,2) for b in bitstrings]
res_keys = torch.tensor(res_keys, dtype=torch.int64).to(device)

#### DEAL WITH ground truth ######
gtFile = "results/benchmark/amps3M_all.txt"
gt_keys = torch.zeros_like(res_keys)
gt_data = torch.empty(res_keys.shape, device = device, dtype = torch.complex64)
with open(gtFile, 'r') as file:
    i = 0
    for line in file:
        row = line.split("\t")
        gt_keys[i] = int(row[0],2)
        data = row[1].split(",")
        gt_data[i].real = float(data[0][1:])
        gt_data[i].imag = float(data[1][:-1])
        i += 1

gtkeys_sorted, gt_idx = torch.sort(gt_keys)
gtdata_sorted = gt_data[gt_idx]
gtdata_sorted = torch.save(gtdata_sorted, "results/benchmark/32T_gtdata_sorted.pt")