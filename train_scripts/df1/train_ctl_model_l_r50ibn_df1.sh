python train_ctl_model.py \
--config_file="configs/320_resnet50_ibn_a.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'df1' \
DATASETS.JSON_TRAIN_PATH '/data/deep_fashion/consumer_to_shop/train_320_320.json' \
DATASETS.ROOT_DIR '/data/deep_fashion/consumer_to_shop/320_320_images/' \
SOLVER.IMS_PER_BATCH 12 \
TEST.IMS_PER_BATCH 256 \
SOLVER.BASE_LR 1e-4 \
OUTPUT_DIR './logs/df1/320_resnet50_ibn_a' \
DATALOADER.USE_RESAMPLING False \
MODEL.KEEP_CAMID_CENTROIDS False