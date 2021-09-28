python train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'street2shop' \
DATASETS.JSON_TRAIN_PATH '/data/street2shop/train_128_256.json' \
DATASETS.ROOT_DIR '/data/128_256_images' \
SOLVER.IMS_PER_BATCH 48 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 1e-4 \
OUTPUT_DIR './logs/street2shop/256_resnet50' \
DATALOADER.USE_RESAMPLING False \
MODEL.KEEP_CAMID_CENTROIDS False