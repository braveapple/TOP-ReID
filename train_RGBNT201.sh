#!/bin/bash

set -e

# CUDA_VISIBLE_DEVICES='2' 
#     python train_net.py \
#     --config_file configs/RGBNT201/TOP-ReID.yml \
#     SOLVER.IMS_PER_BATCH 32 \
#     OUTPUT_DIR /mnt/disk/wpy_data/experiments/TOP-ReID/RGBNT201/one_gpu

CUDA_VISIBLE_DEVICES='2,3' \
    python -m torch.distributed.launch --nproc_per_node=2 --master_port 6666 train_net.py \
    --config_file configs/RGBNT201/TOP-ReID.yml \
    MODEL.DIST_TRAIN True \
    SOLVER.IMS_PER_BATCH 64

# CUDA_VISIBLE_DEVICES='0,1,2,3' \
#     python -m torch.distributed.launch --nproc_per_node=4 --master_port 6666 train_net.py \
#     --config_file configs/RGBNT201/TOP-ReID.yml \
#     MODEL.DIST_TRAIN True \
#     SOLVER.IMS_PER_BATCH 128