#!/bin/bash

set -e

CUDA_VISIBLE_DEVICES=1 python train_net.py \
    --config_file configs/RGBNT100/TOP-ReID.yml \
    SOLVER.IMS_PER_BATCH 64