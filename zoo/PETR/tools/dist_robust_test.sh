#!/usr/bin/env bash

CONFIG=/nvme/konglingdong/models/RoboDet/zoo/PETR/projects/configs/robust_test/petr_vovnet_gridmask_p4_1600x640.py
CHECKPOINT=/nvme/konglingdong/models/RoboDet/models/DETR3D/petr_vov_p4_1600x640.pth
GPUS=8
PORT=${PORT:-29502}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/robust_test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
