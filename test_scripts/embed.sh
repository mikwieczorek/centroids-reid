python inference/create_embeddings.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.ROOT_DIR 'data/ugv/' \
TEST.IMS_PER_BATCH 8 \
OUTPUT_DIR 'outputs/' \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "models/market1501_resnet50_256_128_epoch_120.ckpt"