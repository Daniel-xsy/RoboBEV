#!/usr/bin/env bash

CONFIG=/nvme/konglingdong/models/RoboDet/zoo/BEVFormer/projects/configs/robust_test/bevformer_small.py
CHECKPOINT=/nvme/konglingdong/models/RoboDet/models/BEVFormer/bevformer_small_epoch_24.pth
GPUS=8
PORT=${PORT:-29501}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/robust_test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
