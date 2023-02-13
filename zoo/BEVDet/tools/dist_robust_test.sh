#!/usr/bin/env bash

CONFIG=/nvme/konglingdong/models/RoboDet/zoo/BEVDet/configs/robust_test/bevdet-r50.py
CHECKPOINT=/nvme/konglingdong/models/RoboDet/models/BEVDet/bevdet-r50.pth
GPUS=8
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/robust_test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox \
