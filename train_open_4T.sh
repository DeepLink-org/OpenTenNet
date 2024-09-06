#### Default configuration ####
export ntasks_per_node=8
export time=$(date +%m-%d-%H_%M_%S)

###############################################################
#######################   加关联子空间   #######################
###############################################################


export nnodes=4 # 全局需要多少个node
export WORLD_SIZE=$((${nnodes}*${ntasks_per_node}))
export nodes_per_task=4 # 做一个子任务需要多少个node
srun -p llm_t --quotatype=reserved --cpus-per-task=8 \
--nodes=${nnodes} --ntasks=$((${nnodes}*${ntasks_per_node})) \
--ntasks-per-node=${ntasks_per_node} \
--gres=gpu:${ntasks_per_node} \
python train/2T/open_truetask.py \
--job_select 0 --warmup 1 --data_type 0 --is_scale 1 --autotune 1 \
--ntask 1  --tensorNetSize 2T --typeCom int4kernel



# # #### configuration needed to change ####
# export nnodes=176 # 全局需要多少个node
# export WORLD_SIZE=$((${nnodes}*${ntasks_per_node}))
# export nodes_per_task=8 # 做一个子任务需要多少个node

# srun -p llm_t --quotatype=reserved --cpus-per-task=16 \
# --nodes=${nnodes} --ntasks=$((${nnodes}*${ntasks_per_node})) \
# --ntasks-per-node=${ntasks_per_node} \
# --gres=gpu:${ntasks_per_node} \
# python train/2T/open_truetask.py \
# --job_select 0 --warmup 1 --data_type 0 --is_scale 1 --autotune 1 \
# --ntask 2 --tensorNetSize 2T --typeCom int4kernel
