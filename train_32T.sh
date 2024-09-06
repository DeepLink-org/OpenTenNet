#### Default configuration ####
export ntasks_per_node=8
export time=$(date +%m-%d-%H_%M_%S)
tensorNetSize=16T

#### configuration needed to change ####
export nnodes=32 # 全局需要多少个node
export WORLD_SIZE=$((${nnodes}*${ntasks_per_node}))
export nodes_per_task=32 # 做一个子任务需要多少个node
srun -p llm_t --quotatype=reserved --cpus-per-task=8 \
--nodes=${nnodes} --ntasks=$((${nnodes}*${ntasks_per_node})) \
--ntasks-per-node=${ntasks_per_node} \
--gres=gpu:${ntasks_per_node} \
python train/${tensorNetSize}/truetask.py \
--job_select 0 --warmup 1 --data_type 0 --is_scale 1 --autotune 1 \
--ntask 1 --tensorNetSize ${tensorNetSize} --typeCom int4kernel \
# > log/${tensorNetSize}_${nnodes}nodes_${time}.log 2>&1 
