import copy
import glob
import json
import logging
import os
import os.path as osp
import shutil
from collections import defaultdict

import numpy as npa
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


def load_json(json_abs_path):
    return json.load(open(json_abs_path))


def create_image_info(
    image_id,
    width,
    height,
    file_name,
    license=0,
    flickr_url="",
    coco_url="",
    data_captured="",
):
    image = {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "license": license,
        "flickr_url": flickr_url,
        "coco_url": coco_url,
        "date_captured": data_captured,
    }

    return image


def create_annotations(
    anno_id,
    image_id,
    category_id,
    source,
    bbox="",
    pair_id="",
    style="",
    segmentation="",
    area=0,
    iscrowd=0,
):
    annotation = {
        "id": anno_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "area": area,
        "bbox": bbox,
        "iscrowd": iscrowd,
        "pair_id": pair_id,
        "style": style,
        "source": source,
    }

    return annotation


def _resize_thumbnail(im: Image, target_image_size: tuple):
    im.thumbnail(target_image_size, Image.ANTIALIAS)
    new_image = Image.new("RGB", target_image_size, (255, 255, 255))
    new_image.paste(
        im,
        (
            int((target_image_size[0] - im.size[0]) / 2),
            int((target_image_size[1] - im.size[1]) / 2),
        ),
    )

    return new_image


def crop_single_bbox(image, bbox, target_image_size: tuple):
    img_arr = np.array(image)
    bbox = np.asarray(bbox)
    bbox_int = bbox.astype(np.int32)
    x1, y1, x2, y2 = bbox_int[:4]
    #     x2 = x1 + bbox_w
    #     y2 = y1 + bbox_h
    cut_arr = img_arr[y1:y2, x1:x2]  # Cut-out the instance detected
    im = Image.fromarray(cut_arr)
    cut_arr = _resize_thumbnail(im, target_image_size)

    return cut_arr


def resize_low_res_bbox_to_high_res(low_res_bbox, low_res_w, low_res_h, w, h):
    # We can do such a transformation as the aspect ratio of low and high
    # resolution images are the same
    x1, y1, x2, y2 = low_res_bbox
    x1_ratio = x1 / low_res_w
    x2_ratio = x2 / low_res_w
    y1_ratio = y1 / low_res_h
    y2_ratio = y2 / low_res_h

    #     global_w_ratio = w

    high_res_bbox = [x1_ratio * w, y1_ratio * h, x2_ratio * w, y2_ratio * h]
    high_res_bbox = list(map(int, high_res_bbox))

    return high_res_bbox


if __name__ == "__main__":
    ROOT_DIR = Path("/data/mwieczorek/data/deep_fashion/consumer_to_shop/")
    IMAGES_ORG_PATH = ROOT_DIR / "img_highres"
    IMAGES_ROOT_DIR_SPLIT = ROOT_DIR / "images_high_res_tmp"
    IMAGES_ROOT_DIR_SPLIT.mkdir(exist_ok=True)
    LOW_RES_IMAGES_ROOT = ROOT_DIR / "img_low_res"

    ## Get Train/val/test split
    split_load_path = ROOT_DIR / "Eval/list_eval_partition.txt"
    with open(split_load_path, "r") as f:
        split_file = f.readlines()

    split_dict = defaultdict(list)
    for line in split_file[2:]:
        # Split long string
        splitted = line.split()
        # Extract source dir
        subset_type = splitted[-1]
        pair_id = splitted[-2]
        source_dir = os.path.split(splitted[0])[0]
        tmp_dict = {"pair_id": pair_id, "source_dir": source_dir}
        split_dict[subset_type].append(tmp_dict)

    # Rename erronous directory
    log.warn(
        f"{ROOT_DIR / 'img_highres/CLOTHING/Summer_Suit/'} needs to be renamed {ROOT_DIR / 'img_highres/CLOTHING/Summer_Wear/'}. Renaming automatically."
    )
    if (ROOT_DIR / "img_highres/CLOTHING/Summer_Suit").is_dir():
        shutil.move(
            str(ROOT_DIR / "img_highres/CLOTHING/Summer_Suit"),
            str(ROOT_DIR / "img_highres/CLOTHING/Summer_Wear"),
        )

    # Split images to subset/id folder
    for subset_name, subset in split_dict.items():
        all_sources = np.unique([item["source_dir"] for item in subset])
        for source in all_sources:
            source_path = ROOT_DIR / source
            files_in_dir = os.listdir(source_path)
            dir_name = source_path.stem
            dir_path = IMAGES_ROOT_DIR_SPLIT / subset_name / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            # to remove "img/" from example like 'img/CLOTHING/Blouse/id_00005025' -> '/CLOTHING/Blouse/id_00005025' as we want high res images
            source_root = source.lstrip("img/")
            for file in files_in_dir:
                shutil.copy(IMAGES_ORG_PATH / source_root / file, dir_path / file)

    # Create global mapping to pair-ids
    global_product_pair_id_map = {}
    unique_pair_id = 0

    for subset_name, subset in split_dict.items():
        all_sources = np.unique([item["source_dir"] for item in subset])
        for source in all_sources:
            dir_name = source.split("/")[-1]
            if dir_name not in global_product_pair_id_map:
                global_product_pair_id_map[dir_name] = unique_pair_id
                unique_pair_id += 1

    # Prepare bboxes
    bbox_load_path = ROOT_DIR / "Anno/list_bbox_consumer2shop.txt"
    with open(bbox_load_path, "r") as f:
        bbox_file = f.readlines()

    source_dict = {"1": "shop", "2": "user"}

    bbox_dict = defaultdict(dict)
    # bbox_dict = {}
    for line in bbox_file[2:]:
        # Split long string
        bbox_txt = line.split()
        # Extract source dir
        path_split = bbox_txt[0].split("/")
        id_name = path_split[-2]
        photo_name = path_split[-1]
        bbox_tmp = bbox_txt[-4:]
        bbox_tmp = [int(item) for item in bbox_tmp]
        style = bbox_txt[1]
        source = source_dict[bbox_txt[2]]
        tmp_dict = {photo_name: {"bbox": bbox_tmp, "style": style, "source": source}}
        #     id_dic = bbox_dict.get(id_name, {})

        bbox_dict[id_name].update(tmp_dict)

    # Copy / rename / save
    TARGET_IMAGE_SIZE = (320, 320)
    CROP_IMAGES_SAVE_ROOT = (
        ROOT_DIR / f"./{TARGET_IMAGE_SIZE[0]}_{TARGET_IMAGE_SIZE[1]}_cropped_images_new"
    )
    # CROP_IMAGES_SAVE_ROOT = f'./{TARGET_IMAGE_SIZE[0]}_{TARGET_IMAGE_SIZE[1]}_full_images'
    CROP_IMAGES_SAVE_ROOT.mkdir(exist_ok=True)

    next_anno_id = 0
    next_img_id = 0
    all_annotations = {}
    all_image_infos = {}  # To easily retrieve images neede for the set
    all_products = []  # To easily retrieve products/pair_ids neede for the set
    pair_id_temp_dict = {}  # To store (old_pairid, style) to map to new pair_ids

    # Crop from high-res images
    for subset_name in list(split_dict.keys()):
        #     if subset_name != 'test':
        #         continue
        CROP_IMAGES_SAVE_PATH = os.path.join(CROP_IMAGES_SAVE_ROOT, subset_name)
        os.makedirs(CROP_IMAGES_SAVE_PATH, exist_ok=True)

        json_obj = {}

        images_info = []
        annos = []
        root_dir = os.path.join(IMAGES_ROOT_DIR_SPLIT, subset_name)
        for dir_name in os.listdir(root_dir):
            pair_id = global_product_pair_id_map[
                dir_name
            ]  # Pair id is assumed to be the same for all items in a directory of template "id_xxxxxx"
            dir_path = os.path.join(root_dir, dir_name)
            for file in os.listdir(dir_path):
                new_filename = dir_name + "_" + file
                source_filepath = os.path.join(dir_path, file)
                # Start preparing image info and annotations
                im_id = next_img_id
                im_filename = new_filename
                image_open = Image.open(source_filepath)
                w, h = image_open.size

                low_res_source_filepath = os.path.join(
                    str(dir_path).replace(
                        str(IMAGES_ROOT_DIR_SPLIT), str(LOW_RES_IMAGES_ROOT)
                    ),
                    file,
                )
                low_res_image_open = Image.open(low_res_source_filepath)
                low_res_w, low_res_h = low_res_image_open.size

                single_image_info = create_image_info(
                    image_id=im_id,
                    width=w,
                    height=h,
                    file_name=im_filename,
                    license=0,
                    flickr_url="",
                    coco_url="",
                    data_captured="",
                )
                images_info.append(single_image_info)
                next_img_id += 1

                ### ANNOTATIONS
                img_anno_dict = bbox_dict[dir_name][file]
                anno_style = img_anno_dict[
                    "style"
                ]  # 1,2 or 3. Upper, lower of full-body clothes
                anno_source = img_anno_dict["source"]
                anno_bbox = img_anno_dict["bbox"]
                anno_pair_id = global_product_pair_id_map[dir_name]
                anno_id = next_anno_id

                if not os.path.isfile(os.path.join(CROP_IMAGES_SAVE_PATH, im_filename)):
                    if anno_bbox != "":
                        anno_bbox = np.asarray(anno_bbox).astype(np.int32)
                        high_res_bbox = resize_low_res_bbox_to_high_res(
                            anno_bbox[:4], low_res_w, low_res_h, w, h
                        )
                        if (
                            high_res_bbox[3] != 0 and high_res_bbox[2] != 0
                        ):  # Remove all annotations that have width/height==0
                            cropped = crop_single_bbox(
                                image_open, high_res_bbox, TARGET_IMAGE_SIZE
                            )
                        else:
                            continue
                    else:
                        cropped = _resize_thumbnail(image_open, TARGET_IMAGE_SIZE)

                single_crop_anno = create_annotations(
                    anno_id=anno_id,
                    image_id=im_id,
                    category_id=anno_style,
                    bbox="",
                    pair_id=pair_id,
                    style=anno_style,
                    source=anno_source,
                    segmentation="",
                    area=0,
                    iscrowd=0,
                )
                annos.append(single_crop_anno)
                next_anno_id += 1

                if os.path.isfile(os.path.join(CROP_IMAGES_SAVE_PATH, im_filename)):
                    continue
                cropped.save(os.path.join(CROP_IMAGES_SAVE_PATH, im_filename))

        all_image_infos[subset_name] = images_info
        all_annotations[subset_name] = annos

        json_obj["images"] = images_info
        json_obj["annotations"] = annos

        with open(
            ROOT_DIR
            / f"{subset_name}_reid_cropped_{TARGET_IMAGE_SIZE[0]}_{TARGET_IMAGE_SIZE[1]}_new_github.json",
            "w",
        ) as f:
            json.dump(json_obj, f)

    #     break

    ### Create splits
    # Prepare test
    test_img_info = np.array(all_image_infos["test"])
    test_img_info_ids = np.array([item["id"] for item in test_img_info])
    test_annotations = np.array(all_annotations["test"])
    test_annotations_ids = np.array([item["id"] for item in test_annotations])

    query_annotations_ids = [
        item["id"] for item in test_annotations if item["source"] == "user"
    ]
    query_annotations = test_annotations[
        np.isin(test_annotations_ids, query_annotations_ids)
    ]
    query_annotations_img_ids = [item["image_id"] for item in query_annotations]
    query_images = test_img_info[np.isin(test_img_info_ids, query_annotations_img_ids)]

    gallery_annotations = test_annotations[
        np.isin(test_annotations_ids, query_annotations_ids, invert=True)
    ]
    gallery_images = test_img_info[
        np.isin(test_img_info_ids, query_annotations_img_ids, invert=True)
    ]

    # Prepare validation
    val_img_info = np.array(all_image_infos["val"])
    val_img_info_ids = np.array([item["id"] for item in val_img_info])
    val_annotations = np.array(all_annotations["val"])
    val_annotations_ids = np.array([item["id"] for item in val_annotations])

    gallery_val_annotations_ids = [
        item["id"] for item in val_annotations if item["source"] == "shop"
    ]
    gallery_val_annotations = val_annotations[
        np.isin(val_annotations_ids, gallery_val_annotations_ids)
    ]
    gallery_val_annotations_img_ids = [
        item["image_id"] for item in gallery_val_annotations
    ]
    gallery_val_images = val_img_info[
        np.isin(val_img_info_ids, gallery_val_annotations_img_ids)
    ]

    gallery_images = list(gallery_images)
    gallery_annotations = list(gallery_annotations)
    gallery_val_images = list(gallery_val_images)
    gallery_val_annotations = list(gallery_val_annotations)

    gallery_images.extend(gallery_val_images)
    gallery_annotations.extend(gallery_val_annotations)

    gallery_pair_ids = [item["pair_id"] for item in gallery_annotations]
    query_pair_ids = [item["pair_id"] for item in query_annotations]
    query_pair_ids.extend(gallery_pair_ids)

    unique_pair_ids_set = np.unique(query_pair_ids)

    pid2label = {pid: label for label, pid in enumerate(unique_pair_ids_set)}

    for idx, anno in enumerate(query_annotations):
        anno["pair_id"] = pid2label[anno["pair_id"]]
        query_annotations[idx] = anno

    for idx, anno in enumerate(gallery_annotations):
        anno["pair_id"] = pid2label[anno["pair_id"]]
        gallery_annotations[idx] = anno

    # Query
    json_obj = {}
    json_obj["images"] = list(query_images)
    json_obj["annotations"] = list(query_annotations)
    with open(
        ROOT_DIR
        / f"query_reid_cropped_{TARGET_IMAGE_SIZE[0]}_{TARGET_IMAGE_SIZE[1]}_new_github.json",
        "w",
    ) as f:
        json.dump(json_obj, f)

    # Gallery
    json_obj = {}

    json_obj["images"] = gallery_images
    json_obj["annotations"] = gallery_annotations
    with open(
        ROOT_DIR
        / f"gallery_reid_cropped_{TARGET_IMAGE_SIZE[0]}_{TARGET_IMAGE_SIZE[1]}_new_github.json",
        "w",
    ) as f:
        json.dump(json_obj, f)

    for mode, img_info_set in zip(("query", "gallery"), (query_images, gallery_images)):
        IMG_SOURCE = CROP_IMAGES_SAVE_ROOT
        os.makedirs(os.path.join(IMG_SOURCE, mode), exist_ok=True)
        for img_info in img_info_set:
            img_filename = img_info["file_name"]
            for name in ["test", "val"]:
                source = os.path.join(IMG_SOURCE, name, img_filename)
                if os.path.isfile(source):
                    target_path = os.path.join(IMG_SOURCE, mode, img_filename)
                    shutil.copy(source, target_path)
        #             continue