python inference/get_similar.py \
--config_file="configs/256_resnet50.yml" \
--gallery_data='outputs' \
--normalize_features \
--topk=100 \
GPU_IDS [0] \
DATASETS.ROOT_DIR 'data/ugv/'  \
TEST.IMS_PER_BATCH 128 \
OUTPUT_DIR 'outputs' \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "models/market1501_resnet50_256_128_epoch_120.ckpt" \
SOLVER.DISTANCE_FUNC 'cosine'