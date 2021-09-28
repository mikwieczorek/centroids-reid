python train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'market1501' \
DATASETS.ROOT_DIR '/data/mwieczorek/' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './logs/market1501/256_resnet50' \
DATALOADER.USE_RESAMPLING False \
USE_MIXED_PRECISION False

### 
# For visualization we want this
# /home/mwieczorek/icpp_anon/sigir2021-anon-1547/logs/market1501/256_resnet50/train_ctl_model/version_3

# python train_ctl_model.py \
# --config_file="configs/256_resnet50.yml" \
# GPU_IDS [0] \
# DATASETS.NAMES 'market1501' \
# DATASETS.ROOT_DIR '/data/mwieczorek/' \
# SOLVER.IMS_PER_BATCH 16 \
# TEST.IMS_PER_BATCH 128 \
# SOLVER.BASE_LR 0.00035 \
# OUTPUT_DIR './logs/visualize/market1501' \
# DATALOADER.USE_RESAMPLING False \
# USE_MIXED_PRECISION False \
# TEST.WEIGHT "/home/mwieczorek/icpp_anon/sigir2021-anon-1547/logs/market1501/256_resnet50/train_ctl_model/version_3/checkpoints/epoch=79.ckpt" \
# TEST.ONLY_TEST True \
# TEST.VISUALIZE "yes" \
# TEST.VISUALIZE_TOPK 10 \
# TEST.VISUALIZE_MAX_NUMBER 1000 \
# MODEL.PRETRAIN_PATH "/home/mwieczorek/icpp_anon/sigir2021-anon-1547/logs/market1501/256_resnet50/train_ctl_model/version_3/checkpoints/epoch=79.ckpt" \
# MODEL.RESUME_TRAINING False



# python train_ctl_model.py \
# --config_file="configs/256_resnet50.yml" \
# GPU_IDS [0] \
# DATASETS.NAMES 'market1501' \
# DATASETS.ROOT_DIR '/data/mwieczorek/' \
# SOLVER.IMS_PER_BATCH 16 \
# TEST.IMS_PER_BATCH 128 \
# SOLVER.BASE_LR 0.00035 \
# OUTPUT_DIR './logs/test/market1501' \
# DATALOADER.USE_RESAMPLING False \
# USE_MIXED_PRECISION False \
# TEST.WEIGHT "/home/mwieczorek/icpp_anon/sigir2021-anon-1547/logs/market1501/256_resnet50/train_ctl_model/version_3/checkpoints/epoch=79.ckpt" \
# TEST.ONLY_TEST True \
# TEST.VISUALIZE "no" \
# TEST.VISUALIZE_TOPK 10 \
# TEST.VISUALIZE_MAX_NUMBER 1000 \
# MODEL.PRETRAIN_PATH "/home/mwieczorek/icpp_anon/sigir2021-anon-1547/logs/market1501/256_resnet50/train_ctl_model/version_3/checkpoints/epoch=79.ckpt" \
# MODEL.RESUME_TRAINING False