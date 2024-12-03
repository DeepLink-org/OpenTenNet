#! /bin/bash
source /opt/conda/etc/profile.d/conda.sh && conda activate base

NNODES=$1
GPUS_PER_NODE=8
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
# MASTER_PORT=0
MASTER_PORT=29512

HOST_NAME=`hostname`
HOST_FILE=$2
MASTER_ADDR=`python /share/dyu/helper/return_master_addr_from_hostfile.py ${HOST_FILE} `
NODE_RANK=`python /share/dyu/helper/return_myrank.py ${HOST_FILE} ${HOST_NAME} `
echo NODE_RANK=$NODE_RANK HOST_FILE=${HOST_FILE} HOST_NAME=${HOST_NAME} MASTER_ADDR=${MASTER_ADDR} 


export ntasks_per_node=8
export time=$(date +%m-%d-%H_%M_%S)


###############################################################
#######################   加关联子空间   #######################
###############################################################
export nnodes=$NNODES # 全局需要多少个node
export WORLD_SIZE=$((${nnodes}*${ntasks_per_node}))
export nodes_per_task=4 # 做一个子任务需要多少个node
# srun -p llm_e --quotatype=spot --cpus-per-task=8 \
# --nodes=${nnodes} --ntasks=$((${nnodes}*${ntasks_per_node})) \
# --ntasks-per-node=${ntasks_per_node} \
# --gres=gpu:${ntasks_per_node} \
# python scripts/2T/open_truetask_recal_matmul.py \
# --warmup 1 --data_type 0 --is_scale 1 --autotune 1 \
# --ntask 10  --tensorNetSize 2T --typeCom int4kernel

torchrun --nnodes=$NNODES --nproc-per-node=${ntasks_per_node} \
--node_rank ${NODE_RANK} \
--master_addr ${MASTER_ADDR} \
--master_port ${MASTER_PORT} \
scripts/2T/open_truetask_recal_matmul.py \
--job_select 0 --warmup 1 --data_type 0 --is_scale 1 --autotune 0 \
--ntask 1 --tensorNetSize 4T --typeCom int4kernel
