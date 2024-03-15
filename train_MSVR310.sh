#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python train_net.py \
    --config_file configs/MSVR310/TOP-ReID.yml \
    SOLVER.IMS_PER_BATCH 64