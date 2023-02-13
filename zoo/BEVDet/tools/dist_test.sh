#!/usr/bin/env bash

CONFIG=/nvme/konglingdong/models/RoboDet/zoo/BEVDet/configs/bevdepth/bevdepth-r50.py
CHECKPOINT=/nvme/konglingdong/models/RoboDet/models/BEVDepth/bevdepth-r50.pth
GPUS=8
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch --eval bbox
