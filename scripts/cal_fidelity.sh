export nnodes=1
export ntasks_per_node=1
export nodes_per_task=1
# export time=$(date +%m-%d-%H:%M:%S)

srun -p llm0_t --quotatype=spot --cpus-per-task=1 \
--nodes=${nnodes} --ntasks=$((${nnodes}*${ntasks_per_node})) \
--ntasks-per-node=${ntasks_per_node} \
--gres=gpu:${ntasks_per_node} \
python scripts/cal_fidelity.py \
--data_type 0 --warmup 1 --data_type 0 --is_scale 1 --use_int8 1 --autotune 1 \
--ntask 5 --use_int8kernel 0 --tensorNetSize 2T