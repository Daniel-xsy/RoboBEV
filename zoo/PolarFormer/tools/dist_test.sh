#!/usr/bin/env bash

CONFIG=/nvme/konglingdong/models/RoboDet/zoo/PolarFormer/projects/configs/polarformer/polarformer_vovnet.py
CHECKPOINT=/nvme/konglingdong/models/RoboDet/models/PolarFormer/polarformer_v2_99.pth
GPUS=8
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
