import argparse
import json
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from data_utils import (
    _resize_thumbnail,
    create_annotations,
    create_image_info,
    crop_single_bbox,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

SOURCES_DICT = {"1": "shop", "2": "user"}


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


def get_data_splits(split_file):
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
    return split_dict


def scatter_images_to_folders(
    root_dir, images_org_path, images_root_dir_split, subset_name, subset
):
    all_sources = np.unique([item["source_dir"] for item in subset])
    for source in all_sources:
        source_path = root_dir / source
        files_in_dir = os.listdir(source_path)
        dir_name = source_path.stem
        dir_path = images_root_dir_split / subset_name / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        source_root = source.lstrip("img/")
        for file in files_in_dir:
            if (dir_path / file).is_file():
                continue
            shutil.copy(images_org_path / source_root / file, dir_path / file)


def create_gloabl_to_pair_id_mapping(split_dict):
    global_product_pair_id_map = {}
    unique_pair_id = 0
    for _, subset in split_dict.items():
        all_sources = np.unique([item["source_dir"] for item in subset])
        for source in all_sources:
            dir_name = source.split("/")[-1]
            if dir_name not in global_product_pair_id_map:
                global_product_pair_id_map[dir_name] = unique_pair_id
                unique_pair_id += 1
    return global_product_pair_id_map


def prepare_bboxes(SOURCES_DICT, bbox_file):
    bbox_dict = defaultdict(dict)
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
        source = SOURCES_DICT[bbox_txt[2]]
        tmp_dict = {photo_name: {"bbox": bbox_tmp, "style": style, "source": source}}
        bbox_dict[id_name].update(tmp_dict)
    return bbox_dict


def crop_all_images(
    split_dict,
    global_product_pair_id_map,
    root_dir,
    image_root_dir_split,
    low_res_image_root,
    crop_images_save_root,
    target_image_size,
):
    next_anno_id = 0
    next_img_id = 0
    all_annotations = {}
    all_image_infos = {}  # To easily retrieve images neede for the set
    # Crop from high-res images
    for subset_name in list(split_dict.keys()):
        CROP_IMAGES_SAVE_PATH = crop_images_save_root / subset_name
        CROP_IMAGES_SAVE_PATH.mkdir(exist_ok=True, parents=True)

        json_obj = {}
        images_info = []
        annos = []
        images_subset_root = image_root_dir_split / subset_name

        for dir_name in os.listdir(images_subset_root):
            if not (images_subset_root / dir_name).is_dir():
                continue
            pair_id = global_product_pair_id_map[
                dir_name
            ]  # Pair id is assumed to be the same for all items in a directory of template "id_xxxxxx"
            dir_path = os.path.join(images_subset_root, dir_name)
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
                        str(image_root_dir_split), str(low_res_image_root)
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
                anno_id = next_anno_id

                if not (CROP_IMAGES_SAVE_PATH / im_filename).is_file():
                    if anno_bbox != "":
                        anno_bbox = np.asarray(anno_bbox).astype(np.int32)
                        high_res_bbox = resize_low_res_bbox_to_high_res(
                            anno_bbox[:4], low_res_w, low_res_h, w, h
                        )
                        if (
                            high_res_bbox[3] != 0 and high_res_bbox[2] != 0
                        ):  # Remove all annotations that have width/height==0
                            cropped = crop_single_bbox(
                                image_open, high_res_bbox, target_image_size
                            )
                        else:
                            continue
                    else:
                        cropped = _resize_thumbnail(image_open, target_image_size)

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

                if (CROP_IMAGES_SAVE_PATH / im_filename).is_file():
                    continue
                cropped.save(CROP_IMAGES_SAVE_PATH / im_filename)

        all_image_infos[subset_name] = images_info
        all_annotations[subset_name] = annos

        json_obj["images"] = images_info
        json_obj["annotations"] = annos

        with open(
            root_dir
            / f"{subset_name}_reid_cropped_{target_image_size[0]}_{target_image_size[1]}.json",
            "w",
        ) as f:
            json.dump(json_obj, f)

    return all_image_infos, all_annotations


def create_query_gallery_split(root_dir, all_image_infos, all_annotations):

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
        root_dir
        / f"query_reid_cropped_{TARGET_IMAGE_SIZE[0]}_{TARGET_IMAGE_SIZE[1]}.json",
        "w",
    ) as f:
        json.dump(json_obj, f)

    # Gallery
    json_obj = {}

    json_obj["images"] = gallery_images
    json_obj["annotations"] = gallery_annotations
    with open(
        root_dir
        / f"gallery_reid_cropped_{TARGET_IMAGE_SIZE[0]}_{TARGET_IMAGE_SIZE[1]}.json",
        "w",
    ) as f:
        json.dump(json_obj, f)

    return query_images, gallery_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to transform DeepFashion Consumer-to-Shop annotations to ReID-ready COCO format."
    )
    parser.add_argument(
        "--root-dir-path",
        help="path to root directory of Deep Fashion Consumer-to-Shop data",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--target-image-size",
        help=(
            "size of images that will be fed into the network.\
Script crops the bounding boxes and resizes them to the target size. Width x Height"
        ),
        required=False,
        type=int,
        nargs="+",
        default=[320, 320],
    )

    args = parser.parse_args()

    ### PARAMS
    ROOT_DIR = Path(args.root_dir_path)
    TARGET_IMAGE_SIZE = tuple([int(item) for item in args.target_image_size])
    IMAGES_ORG_PATH = ROOT_DIR / "img_highres"
    LOW_RES_IMAGES_ROOT = ROOT_DIR / "img_low_res"
    IMAGES_ROOT_DIR_SPLIT = ROOT_DIR / "images_high_res_tmp"
    IMAGES_ROOT_DIR_SPLIT.mkdir(exist_ok=True)
    CROP_IMAGES_SAVE_ROOT = (
        ROOT_DIR / f"./{TARGET_IMAGE_SIZE[0]}_{TARGET_IMAGE_SIZE[1]}_cropped_images"
    )
    CROP_IMAGES_SAVE_ROOT.mkdir(exist_ok=True)
    assert IMAGES_ORG_PATH.is_dir()

    # Rename erronous directory
    log.warn(
        f"{ROOT_DIR / 'img_highres/CLOTHING/Summer_Suit/'} needs to be renamed {ROOT_DIR / 'img_highres/CLOTHING/Summer_Wear/'}. Renaming automatically."
    )
    if (ROOT_DIR / "img_highres/CLOTHING/Summer_Suit").is_dir():
        shutil.move(
            str(ROOT_DIR / "img_highres/CLOTHING/Summer_Suit"),
            str(ROOT_DIR / "img_highres/CLOTHING/Summer_Wear"),
        )

    ## Get Train/val/test split
    split_load_path = ROOT_DIR / "Eval/list_eval_partition.txt"
    with open(split_load_path, "r") as f:
        split_file = f.readlines()
    split_dict = get_data_splits(split_file)

    # Split images to subset/id folder
    log.info(f"Scattering images temporarily to seperate folders...")
    for subset_name, subset in split_dict.items():
        scatter_images_to_folders(
            ROOT_DIR, IMAGES_ORG_PATH, IMAGES_ROOT_DIR_SPLIT, subset_name, subset
        )

    # Create global mapping to pair-ids
    global_product_pair_id_map = create_gloabl_to_pair_id_mapping(split_dict)

    # Prepare bboxes
    bbox_load_path = ROOT_DIR / "Anno/list_bbox_consumer2shop.txt"
    with open(bbox_load_path, "r") as f:
        bbox_file = f.readlines()
    bbox_dict = prepare_bboxes(SOURCES_DICT, bbox_file)

    # Copy / rename / save
    log.info(
        f"Cropping and resizing images to {TARGET_IMAGE_SIZE}. This may take some time..."
    )
    all_image_infos, all_annotations = crop_all_images(
        split_dict=split_dict,
        global_product_pair_id_map=global_product_pair_id_map,
        root_dir=ROOT_DIR,
        image_root_dir_split=IMAGES_ROOT_DIR_SPLIT,
        low_res_image_root=LOW_RES_IMAGES_ROOT,
        crop_images_save_root=CROP_IMAGES_SAVE_ROOT,
        target_image_size=TARGET_IMAGE_SIZE,
    )

    ### Create splits
    log.info("Creating query and gallery splits...")
    query_images, gallery_images = create_query_gallery_split(
        ROOT_DIR, all_image_infos, all_annotations
    )

    log.info(
        f"Final processing of images. Scattering them to correctly arranged folders. May take some time..."
    )
    for mode, img_info_set in zip(("query", "gallery"), (query_images, gallery_images)):
        (CROP_IMAGES_SAVE_ROOT / mode).mkdir(exist_ok=True, parents=True)
        for img_info in img_info_set:
            img_filename = img_info["file_name"]
            for name in ["test", "val"]:
                source = os.path.join(CROP_IMAGES_SAVE_ROOT, name, img_filename)
                if os.path.isfile(source):
                    target_path = os.path.join(
                        CROP_IMAGES_SAVE_ROOT, mode, img_filename
                    )
                    shutil.copy(source, target_path)

    log.info(f"Removing temporary folder with images: {IMAGES_ROOT_DIR_SPLIT}")
    shutil.rmtree(path=IMAGES_ROOT_DIR_SPLIT)
