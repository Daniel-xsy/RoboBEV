#!/usr/bin/env bash

CONFIG=/nvme/konglingdong/models/RoboDet/zoo/DETR3D/projects/configs/detr3d/detr3d_res101_gridmask_cbgs.py
CHECKPOINT=/nvme/konglingdong/models/RoboDet/models/DETR3D/detr3d_resnet101_cbgs.pth
GPUS=8
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
