#!/usr/bin/env bash

CONFIG=/nvme/konglingdong/models/RoboDet/zoo/BEVerse/projects/configs/robust_test/beverse_singleframe_small.py
CHECKPOINT=/nvme/konglingdong/models/RoboDet/models/BEVerse/beverse_small.pth
GPUS=4
PORT=${PORT:-29502}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1,3,6,7 \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/robust_test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox

# -m debugpy --listen 5678 --wait-for-client 
