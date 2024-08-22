import torch
import numpy as np
import math
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


if __name__ == '__main__':
    device1 = 'cuda:0'
    device2 = 'cuda:1'
    tensor1 = torch.zeros([0]*10, dtype = torch.complex64, device = device1)
    tensor2 = []
    torch.cuda.synchronize()
    st= time.time()
    for i in range(2):
        tensor2.append( tensor1[0].to(device2))
    tensor2 = torch.cat(tensor2)
    torch.cuda.synchronize()
    print(time.time() - st)
    
    