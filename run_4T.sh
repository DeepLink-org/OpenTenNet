#### Default configuration ####
export ntasks_per_node=8
export time=$(date +%m-%d-%H_%M_%S)

###############################################################
#######################   无关联子空间   #######################
###############################################################

#### configuration needed to change ####
export nnodes=4 # 全局需要多少个node
export WORLD_SIZE=$((${nnodes}*${ntasks_per_node}))
export nodes_per_task=4 # 做一个子任务需要多少个node
srun -p llm_e --quotatype=spot --cpus-per-task=8 \
--nodes=${nnodes} --ntasks=$((${nnodes}*${ntasks_per_node})) \
--ntasks-per-node=${ntasks_per_node} \
--gres=gpu:${ntasks_per_node} \
python scripts/2T/truetask.py \
--job_select 0 --warmup 1 --data_type 1 --is_scale 1 --autotune 1 \
--ntask 1  --tensorNetSize 2T --typeCom int4kernel --int4group 128


