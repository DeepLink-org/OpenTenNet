#### Default configuration ####
export ntasks_per_node=8
export time=$(date +%m-%d-%H_%M_%S)


# #### configuration needed to change ####
# export nnodes=4 # 全局需要多少个node
# export WORLD_SIZE=$((${nnodes}*${ntasks_per_node}))
# export nodes_per_task=4 # 做一个子任务需要多少个node
# srun -p llm0_t --quotatype=spot --cpus-per-task=8 \
# --nodes=${nnodes} --ntasks=$((${nnodes}*${ntasks_per_node})) \
# --ntasks-per-node=${ntasks_per_node} \
# --gres=gpu:${ntasks_per_node} \
# python train/2T/truetask.py \
# --job_select 0 --warmup 1 --data_type 0 --is_scale 1 --autotune 1 \
# --ntask 1 --tensorNetSize 2T --typeCom int4kernel


export nnodes=2 # 全局需要多少个node
export WORLD_SIZE=$((${nnodes}*${ntasks_per_node}))
export nodes_per_task=2 # 做一个子任务需要多少个node

srun -p llm0_t --quotatype=spot --cpus-per-task=8 \
--nodes=${nnodes} --ntasks=$((${nnodes}*${ntasks_per_node})) \
--ntasks-per-node=${ntasks_per_node} \
--gres=gpu:${ntasks_per_node} \
python train/2T/truetask_recal.py \
--job_select 0 --warmup 1 --data_type 0 --is_scale 1 --autotune 1 \
--ntask 2 --tensorNetSize 2T --typeCom int4kernel

