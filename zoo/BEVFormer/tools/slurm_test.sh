#!/usr/bin/env bash

set -x

PARTITION=digitalcontent2
JOB_NAME=RoboDet
CONFIG=projects/configs/robust_test/bevformer_base.py
CHECKPOINT=../../models/BEVFormer/bevformer_r101_dcn_24ep.pth
CUDA_HOME=/mnt/cache/share/cuda-11.1/
GPUS=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
srun --partition=${PARTITION} \
     --mpi=pmi2 \
     --gres=gpu:${GPUS} \
     -n${GPUS} \
     --ntasks-per-node=${GPUS} \
     --job-name=${JOB_NAME} \
     --cpus-per-task=16 \
     -x SH-IDC1-10-140-1-162,SH-IDC1-10-140-0-136,SH-IDC1-10-140-1-90,SH-IDC1-10-140-1-29,SH-IDC1-10-140-1-98,SH-IDC1-10-140-1-111,SH-IDC1-10-140-1-20,SH-IDC1-10-140-1-96,SH-IDC1-10-140-0-171,SH-IDC1-10-140-0-168,SH-IDC1-10-140-1-9,SH-IDC1-10-140-1-51,SH-IDC1-10-140-1-120 \
     --kill-on-bad-exit=1 \
     --quotatype=spot \
     python $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher="slurm" --eval bbox --corruption_test