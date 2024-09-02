set -e

source activate MMReID
export PYTHONPATH=$PYTHONPATH:/mnt/disk/wpy_data/code/shell
export CUDA_VISIBLE_DEVICES=$(python -c 'from nv import get_one_gpu_loop; print(get_one_gpu_loop(min_mem=20, max_day_time=2))')

exp_01() {
    dataset=${1}
    output_dir=/mnt/disk/wpy_data/experiments/multi_modality_object_reidentification/TOP-ReID/${dataset}
    python train.py --config_file configs/${dataset}/TOP-ReID.yml \
        SOLVER.IMS_PER_BATCH 128 \
        OUTPUT_DIR ${output_dir}
}

# CUDA_VISIBLE_DEVICES='2,3' \
#     python -m torch.distributed.launch --nproc_per_node=2 --master_port 6666 train.py \
#     --config_file configs/MSVR310/TOP-ReID.yml \
#     MODEL.DIST_TRAIN True \
#     SOLVER.IMS_PER_BATCH 100

# exp_01 RGBNT100
exp_01 RGBNT201
# exp_01 MSVR310