#!/bin/bash

set -e

# python train_net.py --config_file configs/RGBNT201/TOP-ReID.yml \
#     SOLVER.IMS_PER_BATCH 64

CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --config_file configs/RGBNT201/TOP-ReID.yml \
    SOLVER.IMS_PER_BATCH 80 \
    MODEL.SHARE_BACKBONE True \
    OUTPUT_DIR /mnt/disk/wpy_data/experiments/TOP-ReID/RGBNT100/share_backbone