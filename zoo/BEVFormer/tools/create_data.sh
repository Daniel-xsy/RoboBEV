#!/usr/bin/env bash

set -x

PARTITION=digitalcontent
JOB_NAME=RoboDet
GPUS=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
srun --partition=${PARTITION} \
     --mpi=pmi2 \
     --gres=gpu:${GPUS} \
     -n${GPUS} \
     --ntasks-per-node=${GPUS} \
     --job-name=${JOB_NAME} \
     --cpus-per-task=18 \
     -x SH-IDC1-10-140-1-162,SH-IDC1-10-140-0-136,SH-IDC1-10-140-1-90,SH-IDC1-10-140-1-29,SH-IDC1-10-140-1-98,SH-IDC1-10-140-1-111,SH-IDC1-10-140-1-20,SH-IDC1-10-140-1-96,SH-IDC1-10-140-0-171,SH-IDC1-10-140-0-168,SH-IDC1-10-140-1-9 \
     --kill-on-bad-exit=1 \
     --quotatype=spot \
    python tools/create_data.py nuscenes \
    --root-path /mnt/petrelfs/share_data/liuyouquan/nuScenes/ \
    --out-dir ../../data \
    --extra-tag nuscenes \
    --version v1.0 \
    --canbus ../../data