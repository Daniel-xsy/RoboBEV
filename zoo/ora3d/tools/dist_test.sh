#!/usr/bin/env bash

CONFIG=/nvme/konglingdong/models/RoboDet/zoo/ora3d/projects/configs/robust_test/ora3d_res101.py
CHECKPOINT=/nvme/konglingdong/models/RoboDet/models/ORA3D/ora3d-r101.pth
GPUS=8
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
