#### Default configuration ####
export ntasks_per_node=8
export time=$(date +%m-%d-%H_%M_%S)


export nnodes=1 # 全局需要多少个node
export WORLD_SIZE=$((${nnodes}*${ntasks_per_node}))
export nodes_per_task=1 # 做一个子任务需要多少个node

srun -p llm_e --quotatype=spot --cpus-per-task=8 \
--nodes=${nnodes} --ntasks=$((${nnodes}*${ntasks_per_node})) \
--ntasks-per-node=${ntasks_per_node} \
--gres=gpu:${ntasks_per_node} \
python scripts/640G/truetask.py \
--job_select 0 --warmup 1 --data_type 0 --is_scale 0 --autotune 1 \
--ntask 1 --tensorNetSize 640G --typeCom complex32