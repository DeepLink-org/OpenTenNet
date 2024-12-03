#! /bin/bash
export NNODES=16
export HOST_FILE="/share/dyu/helper/hostfile.txt"

# export NNODES=12
# export HOST_FILE="/share/dyu/helper/hostfile_12nodes.txt"

# export NNODES=8
# export HOST_FILE="/share/dyu/helper/hostfile_8nodes.txt"

# export NNODES=1
# export HOST_FILE="/share/dyu/helper/hostfile_1node.txt"

# export NNODES=2
# export HOST_FILE="/share/dyu/helper/hostfile_2nodes.txt"

# export NNODES=4
# export HOST_FILE="/share/dyu/helper/hostfile_4nodes.txt"

# /opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} /bin/bash /share/work/sh/work/soft/test/sw.sh
cd /share/dyu
/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} ps -ef |grep torch |grep root
/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} ps -ef |grep python |grep root

# step1: 先给各个计算节点拉取镜像
/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker load -i /share/dyu/helper/pytorch2.1.tar.gz

# step2: 先停掉历史的残留容器
/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker stop  mx_Ai4s_test

/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker rm  mx_Ai4s_test

/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker stop lumina_train_share
# # step3: 在各个计算节点上启动容器
# /opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
#    docker run -it --device=/dev/mxcd --device=/dev/dri --group-add video \
#     -v /share/dyu/OpenTenNet-truetask:/root/opentennet \--device=/dev/infiniband \
#     --network=host --security-opt seccomp=unconfined --security-opt apparmor=unconfined\
#      --shm-size '100gb' --ulimit memlock=-1 --rm \
#     metax-tech.com/library/opentennet:2.27.0.1-ubuntu20.04-amd64 /bin/bash
/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker run -itd \
                -u root  \
                --device=/dev/dri/card6 \
                --device=/dev/dri/renderD134 \
                --device=/dev/dri/card7 \
                --device=/dev/dri/renderD135 \
                --device=/dev/mxcd  \
                --device=/dev/infiniband \
                --privileged=true \
                --group-add=video \
                --name mx_Ai4s_test \
                --security-opt seccomp=unconfined \
                --security-opt apparmor=unconfined \
                --shm-size 160gb \
                --ulimit memlock=-1 \
                -v /share:/share  \
                -v /data:/data \
                --network host \
                mxcr.io/cimaster/maca-c500-pytorch2.1-opentennet:20241120-136-ubuntu20.04-amd64 /bin/bash


# # step4: 杀掉计算节点上残留的进程，再启动任务
/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES} \
   docker exec  mx_Ai4s_test /bin/bash -c "/share/dyu/helper/kill.sh"

# /opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES}  \
#    docker exec  mx_Ai4s_test /bin/bash -c "cd /share/dyu/OpenTenNet-truetask && export PYTHONPATH=/share/dyu/OpenTenNet-truetask/submodule/python:$PYTHONPATH && ./run_open_4T_recal.sh ${NNODES} ${HOST_FILE}"

/opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES}  \
   docker exec  mx_Ai4s_test /bin/bash -c "cd /share/dyu/OpenTenNet-truetask && ./run_open_4T_recal.sh ${NNODES} ${HOST_FILE}"
# # /opt/maca/ompi/bin/mpirun -hostfile ${HOST_FILE} -np ${NNODES}  \
# #    docker exec  mx_Ai4s_test /bin/bash -c "cd /share/dyu/OpenTenNet-truetask && ./torchrun.sh ${NNODES} ${HOST_FILE}" 
