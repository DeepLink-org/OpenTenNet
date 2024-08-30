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
### 3. 4T TensorNetwork
#### 3.1 Uncompress
```
unxz example.txt.xz
```
#### 3.2 Generate tensor network
To grant execution permissions to the script `open_pre4T.sh` within the `TensorNetwork` directory, use the following command:
```
chmod +x TensorNetwork/open_pre4T.sh
```
The TensorNetwork/open_pre4T.sh looks like
```
export nodes_per_task=2
export ntasks_per_node=8
python TensorNetwork/open_pre4T.py
```
Here, `nodes_per_task` represents the number of nodes required for a multi-node level task, while `ntasks_per_node` denotes the number of GPUs per node. Please remember to adjust the values of `nodes_per_task` and `ntasks_per_node` according to the specific node configuration you intend to utilize.

To initiate the tensor network generation process, execute the script with the command:
```
./TensorNetwork/open_pre4T.sh
```

#### 3.3 Excecution
```
chmod +x run_open_4T.sh
./run_open_4T.sh
```

#### 3.4 Explanation of the bash script
The "run_open_4T.sh" looks like
```
#### Default configuration ####
export ntasks_per_node=8
export time=$(date +%m-%d-%H_%M_%S)

export nnodes=2 # 全局需要多少个node
export WORLD_SIZE=$((${nnodes}*${ntasks_per_node}))
export nodes_per_task=2 # 做一个子任务需要多少个node
srun -p {YOUR PARTITION} --quotatype=spot --cpus-per-task=8 \
--nodes=${nnodes} --ntasks=$((${nnodes}*${ntasks_per_node})) \
--ntasks-per-node=${ntasks_per_node} \
--gres=gpu:${ntasks_per_node} \
python scripts/2T/open_truetask.py \
--job_select 0 --warmup 1 --data_type 0 --is_scale 1 --autotune 1 \
--ntask 84  --tensorNetSize 2T --typeCom int4kernel
```

The parameters associated with the three-level parallel scheme in the bash script are recognized as follows:

- `nnodes`: The total number of nodes used globally. This corresponds to the global level of our three-level scheme, and this value must be an integer multiple of `nodes per task`.
- `nodes per task`: The number of nodes required for a multi-node level task. It must be consistent with the `nodes per task` parameter in the preprocessing file (see Table I).
- `WORLD_SIZE`: The total number of GPUs required globally, corresponding to the device level of the three-level scheme.

The primary function is crafted using the Python language. It accommodates multiple arguments that enable the specification of the techniques to be employed. Below is an explanation and basic usage of our parameters:

- `warmup`: Whether the GPU warms up before the official operation. `0` represents no warm-up, and `1` represents warm-up.
- `data_type`: The calculation mode used in this process is detailed in Table I.
- `is_scale`: Calculate the scaling factor for the `complex32` calculation mode, with a default value of `1`.
- `autotune`: Tuning for the best algorithms for `einsum` calculation, with a default value of `1`.
- `ntask`: We execute the number of multi-node level tasks, where the number of global-level tasks is equal to `ntask` × (`nnodes` / `nodes per task`).
- `tensorNetSize`: The scale of tensor networks.
- `typeCom`: The parameter selection for inter-node data quantization communication is detailed in the table below.


