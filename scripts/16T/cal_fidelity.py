import torch
import numpy as np
import pandas as pd

prefix = 'results/16T/open_CALcomplexHalf_COMint4kernel_TUNE1/Nodes32/ntask0/'
data2 = torch.cat([torch.load(prefix + f'/rank{rank}.pt') for rank in range(256)])

bitstringopen = torch.load('TensorNetwork/sc41_reproduce_scheme_n53_m20_ABCDCDAB_3000000_einsum_13_open.pt')[-1]

permute_idx = [7, 3, 12, 0, 4, 5, 6, 10, 8, 11, 9, 1, 2] # this is the permutation index of open result
open_qubits = sorted([11, 12, 13, 19, 20, 22, 23, 28, 33, 37, 38, 40, 44])

close_qubits = list(set(range(53)) - set(open_qubits))

def find_corresponding_indices(dig, dig_raw_pre):
    mapping = {}
    for index, num in enumerate(dig):
        mapping[num] = index

    result = []
    for num in dig_raw_pre:
        assert num in mapping
        result.append(mapping[num])

    return result

# 读取文件，设定分隔符为 "\t"
df = pd.read_csv("results/benchmark/16T/amps3M_all.txt", sep="\t", header=None)

def parse_complex(string):
    string = string.strip('()')  # remove parentheses
    real, imag = map(float, string.split(','))  # split on comma and convert to float
    return complex(real, imag)  # return complex number

# 将第二列的字符串转换成复数
df[1] = df[1].apply(lambda x: parse_complex(x))

# 为列命名
df.columns = ["Binary", "Complex", "Float"]

df["BinaryClose"] = df["Binary"].apply(lambda x: "".join(np.array(list(x))[close_qubits]))

df["Closeidx"] = df["Binary"].apply(lambda x: 
                                   np.frombuffer(bytes(x,'ascii'), dtype=np.uint8)[open_qubits][permute_idx]
                                             - ord('0'))

corresponding_indices = find_corresponding_indices(bitstringopen,df["BinaryClose"])

amp_exact = df["Complex"].values

index_array = np.array(corresponding_indices).reshape(-1)
closeidx = (np.stack(df["Closeidx"].values)*np.array([2**(len(open_qubits)-x-1) for x in range(len(open_qubits))])).sum(axis=-1)
index_array = index_array*(2**len(open_qubits)) + closeidx

amp_sel = data2.flatten().index_select(0, torch.from_numpy(index_array))

amplitude_exact = torch.tensor(amp_exact.copy(),dtype=torch.complex64)[:]
amplitude_appro = amp_sel.clone().to(torch.complex64)[:]
fidelity = (
    (amplitude_exact.conj() @ amplitude_appro.reshape(-1)).abs() /
    (amplitude_exact.abs().square().sum().sqrt() * amplitude_appro.abs().square().sum().sqrt())
).square().item()
fidelity *= np.log(2**13)


expected=0.002
print(f"fidelity of 16T             : {round(fidelity, 8)}")
print(f"expected fidelity(0.002) : {round(expected, 8)}")
print(f"fidelity/expected           : {round(fidelity/expected, 4)}")