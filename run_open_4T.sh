#### Default configuration ####
export ntasks_per_node=8
export time=$(date +%m-%d-%H_%M_%S)

###############################################################
#######################   加关联子空间   #######################
###############################################################
export nnodes=4 # 全局需要多少个node
export WORLD_SIZE=$((${nnodes}*${ntasks_per_node}))
export nodes_per_task=4 # 做一个子任务需要多少个node
srun -p llm_e --quotatype=spot --cpus-per-task=8 \
--nodes=${nnodes} --ntasks=$((${nnodes}*${ntasks_per_node})) \
--ntasks-per-node=${ntasks_per_node} \
--gres=gpu:${ntasks_per_node} \
python scripts/2T/open_truetask.py \
--job_select 0 --warmup 0 --data_type 0 --is_scale 1 --autotune 0 \
--ntask 21  --tensorNetSize 2T --typeCom HalfKernel

