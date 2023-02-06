#!/usr/bin/env bash

CONFIG=/nvme/konglingdong/models/RoboDet/zoo/BEVDet/configs/bevdet/bevdet-r101.py
GPUS=8
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
