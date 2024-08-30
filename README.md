# Achieving Energetic Superiority Through System-Level Quantum Circuit Simulation

## Installation

We tested on a server configured with Linux, cutensor 1.7.0, cuda 12.1, Pytorch 2.1. Other similar configurations should also work, but we have not verified each one individually.
### 1. Clone this repo:

```
git clone https://github.com/DeepLink-org/OpenTenNet.git --recursive
cd OpenTenNet
```

### 2. Install dependencies
#### 2.1 Install conda enviroment
```
conda env create --file environment.yml
conda activate OpenTenNet
```
#### 2.2 Install Cutensor Python extension
##### 2.2.1 Download and extract the cuTENSOR library
https://developer.nvidia.com/cutensor-1-7-0-download-archive

##### 2.2.2 Set the CUTENSOR ROOT environment variable appropriately.
```
export CUTENSOR_ROOT={YOUR libcutensor PATH}
```
##### 2.2.3 Install the customized cutensor python API
```
cd dependencies/cuTENSOR/python
pip install . 
```
#### 2.3 Install quantization dependencies
```
cd $OpenTenNet_PATH
pip install submodule/python
pip install submodule/cuda
```
### 3. Uncompress 80G open9.pt.tar.gz
```
tar -zxvf 80G open9.pt.tar.gz 80G open9.pt
```

## Excecution
```
cd $OpenTenNet_PATH
chmod +x torchrun.sh
./torchrun.sh
```
## Explanation of the bash script
The "torchrun.sh" looks like
```
export time=
export nodes_per_task=1
export ntasks_per_node=1

torchrun --nnodes=1 --nproc-per-node=${ntasks_per_node} \
scripts/faketask.py \
--data_type 1 --ntask 8 --tensorNetSize faketask
```
Among all these parameters and variables, two particula arguments command our attention, data type, and ntasks per node:

1.data type: This parameter indicates the computational data type to be used; if set to "0", the system will use complex half for computation, whereas assigning it value "1" will prompt computation of complex float.

2. ntasks per node: It signifies how many GPUs are engaged within a node, enabling the scalability within a node. Feel free to toggle its settings from "1" to "2", "4" then "8".

## Analysis
The output looks like:
```
=====================================================
===================== RESULT ========================
Profile saved to prof_dir/faketask/CALcomplexFloat_COMcomplex32_TUNE1/Nodes0//ntask7_CALcomplexFloat_COMcomplex32_TUNE1_Nodes0.json
Truetask used 14.522 s
torch.memory.allocated 9.765924453735352 G, torch.memory.reserved 19.091796875 G
energy information saved to prof_dir/faketask/CALcomplexFloat_COMcomplex32_TUNE1/Nodes0//energy/
total consumption 0.0010177632230456961 kwh
```
After the execution, you will find the time-to-solution, a profile path, energy consumption, and the energy log in the output. To assess the time taken by computation and communication, you can visit https://pytorch.org/docs/stable/profiler.html and load the profile json file. If you want to analyze detailed energy information instead of just the total consumption, you can delve into the energy log.
