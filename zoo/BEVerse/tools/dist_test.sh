#!/usr/bin/env bash

CONFIG=/nvme/konglingdong/models/RoboDet/zoo/BEVerse/projects/configs/robust_test/beverse_tiny.py
CHECKPOINT=/nvme/konglingdong/models/RoboDet/models/BEVerse/beverse_tiny.pth
GPUS=4
PORT=${PORT:-29501}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
# -m debugpy --listen 5678 --wait-for-client 