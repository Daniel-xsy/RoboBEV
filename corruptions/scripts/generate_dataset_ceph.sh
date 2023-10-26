#!/usr/bin/env bash
set -x

PARTITION=it_p2
JOB_NAME=gen_snow
CONFIG=/mnt/petrelfs/xieshaoyuan/models/RoboBEV/corruptions/project/config/syn_snow_ceph.py
GPUS=1
GPUS_PER_NODE=1
CPUS_PER_TASK=16
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python tools/generate_dataset_ceph.py $CONFIG
