#!/bin/bash

# CUDA_VISIBLE_DEVICES=2 python train_net.py \
#     --config_file configs/MSVR310/TOP-ReID.yml \
#     SOLVER.IMS_PER_BATCH 64

CUDA_VISIBLE_DEVICES='2,3' \
    python -m torch.distributed.launch --nproc_per_node=2 --master_port 6666 train_net.py \
    --config_file configs/MSVR310/TOP-ReID.yml \
    MODEL.DIST_TRAIN True \
    SOLVER.IMS_PER_BATCH 100