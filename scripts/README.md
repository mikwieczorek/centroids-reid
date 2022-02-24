# Fashion data to ReID format
## Street2Shop

In order to transform raw Street2Shop dataset to COCO-ReID format follow the scripts in the directory as:
1. `convert_to_jpg.py`
2. `street2shop2coco.py`
3. `street2shop_coco2reid.py`


### Convert to JPG

The script `convert_to_jpg.py` allows to easily convert all images in folder to JPG format. 

**NOTE** 
If you want to transform raw Street2Shop dataset to ReID format, the conversion of images to JPG is necessary in order to use the subsequent scripts easily.

Example:
```bash
python convert_to_jpg.py \
--source-dir-path /path/to/s2s/images \
--target-dir-path /path/to/s2s/images_jpg \
--num-threads 8
```

### Street2Shop to COCO

The result of the script is a reformatted train Street2Shop dataset in COCO format, but without ReID annotations.

Example:
```bash
python street2shop2coco.py \
--root-dir-path /path/to/s2s \
--metadata-dir meta \
--images-dir images_jpg \
--save-dir meta
```

### Street2Shos to ReID

The result of the script is a reformatted full Street2Shop dataset in COCO format with ReID annotations. Images with bbox annotations will be cropped, resized to `--target-image-size` and saved to seperate folder. Images without bboxes will be only resized.
Full split to val/test and query/gallery subsets will be produced allowing to train the CTLModel on it.

Example:
```bash
python street2shop_coco2reid.py \
--train-json-path /path/to/s2s/meta/all_street_train.json \
--root-dir-path /path/to/s2s \
--metadata-dir meta \
--images-dir images_jpg \
--save-dir meta_reid \
--target-image-size [320, 320] \
--minimum-bbox-area 1
```


## Deep Fashion

Run `deep_fashion2reid.py` script. The dataset will be transformed to COCO-ReID format.
Images with bbox annotations will be cropped, resized to `--target-image-size` and saved to seperate folder. Images without bboxes will be only resized.

**NOTE** To run the script without errors both low and high resolution images from the original dataset needs to be present in the directory.

Example:
```bash
python deep_fashion2reid.py \
--root-dir-path /path/to/s2s \
--target-image-size [320, 320]
```