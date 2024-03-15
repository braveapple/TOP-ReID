set -e

python train_net.py --config_file configs/RGBNT201/TOP-ReID.yml \
    SOLVER.IMS_PER_BATCH 64