# To convert images to jpgs
python scripts/reid_centroid_fahion2coco/street2shop_nbs/original_format_to_coco/convert_to_jpg.py \
--source-dir-path "/data/mwieczorek/home/data/street2shop/images" \
--target-dir-path "/data/mwieczorek/home/data/street2shop/images_jpg" \
--num-threads 8

# To convert coco to reid
python scripts/reid_centroid_fahion2coco/street2shop_nbs/coco_to_reid/script.py \
--train-json-path "/data/mwieczorek/home/data/street2shop/meta/all_street_train.json" \
--root-dir-path "/data/mwieczorek/home/data/street2shop" \
--metadata-dir "meta" \
--images-dir "images_new" \
--save-dir "new-meta-dir" \
--target-image-size 320 320 \
--minimum-bbox-area 1