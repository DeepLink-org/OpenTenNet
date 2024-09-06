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

### 3. 640G TensorNetwork (Optional)
#### 3.1 Uncompress tensor network configuration
```
unxz TensorNetwork/640G/reproduce_scheme_n53_m20_ABCDCDAB_3000000_einsum_10_open.pt.xz
```
#### 3.2 Generate tensor network
To grant execution permissions to the script `open_pre640G.sh` within the `TensorNetwork` directory, use the following command:
```
chmod +x TensorNetwork/640G/open_pre640G.sh
```

<details>
<summary><span style="font-weight: bold;"> Explanation of open_pre640G.sh <span></summary>

  The bash script looks like:
  ```
  export nodes_per_task=1
  export ntasks_per_node=8
  
  python TensorNetwork/640G/open_pre640G.py
  ```
  Here, `nodes_per_task` represents the number of nodes required for a multi-node level task, while `ntasks_per_node` denotes the number of GPUs per node. Please remember to adjust the values of `nodes_per_task` and `ntasks_per_node` according to the specific node configuration you intend to utilize.
</details>
<br>

To initiate the tensor network generation process, execute the script with the command:
```
./TensorNetwork/640G/open_pre640G.sh
```

#### 3.3 Excecution
```
chmod +x run_640G.sh
./run_640G.sh
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for "run_640G"</span></summary>
  
  #### nnodes
  The total number of nodes used globally. This corresponds to the global level of our three-level scheme, and this value must be an integer multiple of ```nodes_per_task```.
  
  #### nodes_per_task
  The number of nodes required for a multi-node level task.
  
  #### WORLD_SIZE
  The total number of GPUs required globally, corresponds to the device level of the three-level scheme.
  
  #### --warmup
  Whether the GPU warms up before the official operation. ```0``` represents no warm-up, and ```1``` represents warm-up.
  #### --data_type
  The calculation type. ```0``` represents complexHalf, ```1``` represent complexFloat.
  
  #### --is_scale
  Calculate the scaling factor for the `complex32` calculation mode, with a default value of `1`.
  
  #### --autotune
  Tuning for the best algorithms for einsum calculation, with a default value of `1`.
  #### --ntask
  We execute the number of multi-node level tasks, where the number of global-level tasks is equal to ```ntask``` * (```nnodes``` / ```nodes per task```).
  #### --tensorNetSize
  The size of tensor networks.
  #### --typeCom
  Data type for communication.  If provided ```int4kernel```, ```int8kernel```, ```HalfKernel```, uses user-defined int4, int8 and half for communication, respectively.

  #### --groupsize
  Group size when typeCom equals int4kernel. ```128``` by default.
  
</details>
<br>

### 4. 4T TensorNetwork (Optional)
#### 4.1 Uncompress tensor network configuration
```
unxz TensorNetwork/4T/sc38_reproduce_scheme_n53_m20_ABCDCDAB_3000000_einsum_10_open.pt.xz
```
#### 4.2 Generate tensor network
To grant execution permissions to the script `open_pre4T_xxx.sh` within the `TensorNetwork` directory, use the following command:
```
# without recomputation
chmod +x TensorNetwork/4T/open_pre4T.sh

# with recomputation (only support when GPU memory > 80 GB, tested on A100)
chmod +x TensorNetwork/4T/open_pre4T_recal.sh
```

<details>
<summary><span style="font-weight: bold;"> Explanation of open_pre4T_xxx.sh <span></summary>

  The bash script looks like:
  ```
  export nodes_per_task=4
  export ntasks_per_node=8
  python TensorNetwork/open_pre4T.py
  ```
  Here, `nodes_per_task` represents the number of nodes required for a multi-node level task, while `ntasks_per_node` denotes the number of GPUs per node. Please remember to adjust the values of `nodes_per_task` and `ntasks_per_node` according to the specific node configuration you intend to utilize.
</details>
<br>

To initiate the tensor network generation process, execute the script with the command:
```
# without recomputation
./TensorNetwork/4T/open_pre4T.sh

# with recomputation (only support when GPU memory > 80 GB, tested on A100)
./TensorNetwork/4T/open_pre4T_recal.sh
```

#### 4.3 Excecution
```
# without recomputation
chmod +x run_open_4T.sh
./run_open_4T.sh

# with recomputation (only support when GPU memory > 80 GB, tested on A100)
chmod +x run_open_4T_recal.sh
./run_open_4T_recal.sh
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for run_open_4T_xx.sh </span></summary>
  
  #### nnodes
  The total number of nodes used globally. This corresponds to the global level of our three-level scheme, and this value must be an integer multiple of ```nodes_per_task```.
  
  #### nodes_per_task
  The number of nodes required for a multi-node level task.
  
  #### WORLD_SIZE
  The total number of GPUs required globally, corresponds to the device level of the three-level scheme.
  
  #### --warmup
  Whether the GPU warms up before the official operation. ```0``` represents no warm-up, and ```1``` represents warm-up.
  #### --data_type
  The calculation type. ```0``` represents complexHalf, ```1``` represent complexFloat.
  
  #### --is_scale
  Calculate the scaling factor for the `complex32` calculation mode, with a default value of `1`.
  
  #### --autotune
  Tuning for the best algorithms for einsum calculation, with a default value of `1`.
  #### --ntask
  We execute the number of multi-node level tasks, where the number of global-level tasks is equal to ```ntask``` * (```nnodes``` / ```nodes per task```).
  #### --tensorNetSize
  The size of tensor networks.
  #### --typeCom
  Data type for communication.  If provided ```int4kernel```, ```int8kernel```, ```HalfKernel```, uses user-defined int4, int8 and half for communication, respectively.

  #### --groupsize
  Group size when typeCom equals int4kernel. ```128``` by default.
  
</details>
<br>

### Evaluation
The output looks like:
```
***************** Results *******************

Profile saved to XXX
energy information saved to XXX
total consumption XX kwh
Truetask used time XX s
save result in XX
Calculating fidelity ...
fidelity of 4T             : XX
expected fidelity(0.002)   : XX
fidelity / expected        : XX
```
After the execution, you will find the time-to-solution, a profile path, energy consumption, an energy log, and the fidelity in the output. To assess the time taken by computation and communication, you can visit https://pytorch.org/docs/stable/profiler.html and load the profile json file. If you want to analyze detailed energy information instead of just the total consumption, you can delve into the energy log.



  



