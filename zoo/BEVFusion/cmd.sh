#!/usr/bin/env bash

set -x

PARTITION=it_p2
JOB_NAME=debug
CONFIG=/mnt/petrelfs/xieshaoyuan/models/RoboBEV/zoo/FCOS3D/projects/config/uda/day2night/fcos3d_r101_day2night.py
WORK_DIR=/mnt/petrelfs/xieshaoyuan/models/RoboBEV/zoo/FCOS3D/work_dir/fcos3d_r101_day2night
GPUS=1
GPUS_PER_NODE=1
CPUS_PER_TASK=4
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
    --debug \
    python setup.py develop