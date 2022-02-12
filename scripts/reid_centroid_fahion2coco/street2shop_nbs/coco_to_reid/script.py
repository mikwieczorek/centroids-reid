import argparse
import copy
import glob
import json
import logging
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

ORIGINAL_CATEGORIES = [
    "bags",
    "belts",
    "dresses",
    "eyewear",
    "footwear",
    "hats",
    "leggings",
    "outerwear",
    "pants",
    "skirts",
    "tops",
]

SET_NAMES = ["train", "query", "gallery"]


def load_json(json_abs_path: str):
    return json.load(open(json_abs_path))


### global to unique pair id mapping
def create_global_to_pair_id_mapping(
    load_json, ORIGINAL_CATEGORIES, meta_dir, category_name2category_id
):
    global_product_pair_id_map = {}
    unique_pair_id = 1
    for category_name in ORIGINAL_CATEGORIES:
        category_id = category_name2category_id[category_name]
        # style_idx = category_id
        # print(category_name, category_id, style_idx)

        s2s_products2pair_ids = {}

        retrival_json = load_json(
            meta_dir / f"retrieval_{category_name}.json"
        )  # Contains only shop photos
        train_json = load_json(
            meta_dir / f"train_pairs_{category_name}.json"
        )  # Contains only train street photos
        test_json = load_json(
            meta_dir / f"test_pairs_{category_name}.json"
        )  # Contains only test street photos
        single_category_all_jsons = train_json + test_json + retrival_json

        single_category_all_products = sorted(
            np.unique([item["product"] for item in single_category_all_jsons])
        )

        for item in single_category_all_products:
            s2s_products2pair_ids[item] = unique_pair_id
            unique_pair_id += 1

        global_product_pair_id_map[category_name] = s2s_products2pair_ids
    return global_product_pair_id_map


def remap_raw_coco_to_pair_ids(
    load_json, ORIGINAL_CATEGORIES, meta_dir, global_product_pair_id_map
):
    ### remap dataset
    remapped_datasets = {}
    for idx, category_name in enumerate(ORIGINAL_CATEGORIES):
        single_category_retrieva_map = global_product_pair_id_map[category_name]

        retrival_json = load_json(
            meta_dir / f"retrieval_{category_name}.json"
        )  # Contains only shop photos
        train_json = load_json(
            meta_dir / f"train_pairs_{category_name}.json"
        )  # Contains only train street photos
        test_json = load_json(
            meta_dir / f"test_pairs_{category_name}.json"
        )  # Contains only test street photos

        for name, dataset in zip(
            ["train", "test", "retrieval"], [train_json, test_json, retrival_json]
        ):
            new_dataset = []
            for jdx, dic in enumerate(dataset):
                product_id = dic["product"]
                dic["product"] = single_category_retrieva_map[product_id]
                new_dataset.append(dic)
            remapped_datasets[f"{name}_pairs_{category_name}.json"] = new_dataset
    return remapped_datasets


# Create annos
def create_annotations(
    anno_id,
    image_id,
    category_id,
    bbox="",
    pair_id="",
    style="",
    segmentation="",
    source="",
    area=0,
    iscrowd=0,
):
    annotation = {
        "id": int(anno_id),
        "image_id": int(image_id),
        "category_id": int(category_id),
        "segmentation": segmentation,
        "area": area,
        "bbox": bbox,
        "iscrowd": int(iscrowd),
        "pair_id": int(pair_id),
        "style": style,
        "source": source,
    }

    return annotation


### Create all images info
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
        "id": int(image_id),
        "file_name": file_name,
        "width": width,
        "height": height,
        "license": license,
        "flickr_url": flickr_url,
        "coco_url": coco_url,
        "date_captured": data_captured,
    }

    return image


def get_pair_ids_less_than_m_examples(annotations, m=2):
    annotations_pair_ids = [item["pair_id"] for item in annotations]
    counter = Counter(annotations_pair_ids)
    pair_ids_less_than_m = np.array(
        ([k for k, v in sorted(counter.items(), key=lambda item: item[1]) if v < m])
    )

    if len(pair_ids_less_than_m) != 0:
        return pair_ids_less_than_m
    else:
        False


def get_bbox_area(item):
    bbox_temp = item.get("bbox", None)
    if bbox_temp is not None:
        bbox = [
            bbox_temp["left"],
            bbox_temp["top"],
            bbox_temp["width"],
            bbox_temp["height"],
        ]
        bbox = [int(item) for item in bbox]
        area = int(np.ceil(bbox_temp["width"] * bbox_temp["height"]))
    else:
        bbox = ""
        area = 0
    return bbox, area


def get_train_test_file_stats(json_obj):
    try:
        photos = [item["photo"] for item in json_obj]
        products = [item["product"] for item in json_obj]
    except KeyError:
        photos = [item["image_id"] for item in json_obj]
        products = [item["pair_id"] for item in json_obj]

    print(
        f"Number of photos: {len(photos)} | Number of unique photos: {len(np.unique(photos))}"
    )
    print(
        f"Number of products: {len(products)} | Number of unique products: {len(np.unique(products))}"
    )


def create_coco_json_with_unique_pair_id_per_category(
    category_name2category_id,
    remapped_datasets,
    category_name,
    info,
    categories,
    licenses,
    all_images_infos,
    all_json_image_ids,
    mode,
    save_dir,
):
    assert mode in ["train", "test"]
    category_id = category_name2category_id[category_name]
    style_id = category_id

    output_json = {}
    output_json["info"] = info
    output_json["categories"] = categories
    output_json["licenses"] = licenses

    anno_id = 0
    all_annos = []
    all_image_ids = []  # To easily retrieve images neede for the set
    all_products = []  # To easily retrieve products/pair_ids neede for the set

    json_data = remapped_datasets[f"{mode}_pairs_{category_name}.json"]
    print("=" * 50)
    print(f"Category: {category_name}")
    print("Stats for original file:")
    get_train_test_file_stats(json_data)

    for item in json_data:
        bbox_temp = item.get("bbox", None)
        if bbox_temp is not None:
            bbox = [
                bbox_temp["left"],
                bbox_temp["top"],
                bbox_temp["width"],
                bbox_temp["height"],
            ]
            bbox = [int(item) for item in bbox]
            # It return False if width or height of bbox is < 1.0 and ommits this item
            area = int(np.ceil(bbox_temp["width"] * bbox_temp["height"]))
        else:
            bbox = ""
            area = 0
        source = "user"

        image_id = item.get("photo")  # It may not be existing?
        all_image_ids.append(image_id)

        pair_id = item["product"]
        all_products.append(pair_id)
        all_annos.append(
            create_annotations(
                anno_id,
                image_id,
                category_id,
                bbox=bbox,
                pair_id=pair_id,
                style=style_id,
                segmentation="",
                source=source,
                area=area,
                iscrowd=0,
            )
        )
        anno_id += 1

    all_image_ids = np.unique(all_image_ids)

    if mode == "train":
        all_products = np.unique(all_products)
    elif mode == "test":
        train_json = load_json(save_dir / f"train_{category_name}.json")
        all_products = np.array([item["pair_id"] for item in train_json["annotations"]])

    print("Stats after creting annotations")
    get_train_test_file_stats(all_annos)

    retrival_json = np.array(
        remapped_datasets[f"retrieval_pairs_{category_name}.json"]
    )  # Contains only shop photos
    retrieval_products = np.array([item["product"] for item in retrival_json])
    retrieval_products_to_take_ind = np.isin(
        retrieval_products, all_products, invert=True if mode == "test" else False
    )
    retrieval_products_to_take = retrival_json[retrieval_products_to_take_ind]
    retrieval_images_to_take = np.array(
        [item["photo"] for item in retrieval_products_to_take]
    )

    # Take images from all_json data
    all_images_ids_to_take_per_category = np.unique(
        np.concatenate((all_image_ids, retrieval_images_to_take), axis=0)
    )
    all_json_image_info_to_take_inds = np.isin(
        all_json_image_ids, all_images_ids_to_take_per_category
    )
    all_json_image_info_to_take = all_images_infos[all_json_image_info_to_take_inds]
    output_json["images"] = list(all_json_image_info_to_take)

    all_annos = list(all_annos)
    # Create annotations for retrieval and merge with train
    for item in retrieval_products_to_take:
        source = "shop"
        image_id = item.get("photo")
        bbox, area = get_bbox_area(item)
        pair_id = item["product"]
        all_annos.append(
            create_annotations(
                anno_id,
                image_id,
                category_id,
                bbox=bbox,
                pair_id=pair_id,
                style=style_id,
                segmentation="",
                source=source,
                area=area,
                iscrowd=0,
            )
        )
        anno_id += 1

    output_json["annotations"] = all_annos

    print("Stats after creting annotations")
    get_train_test_file_stats(all_annos)

    with open(save_dir / f"{mode}_{category_name}.json", "w") as f:
        json.dump(output_json, f)


def create_info_for_all_images(images_dir):
    all_images_paths = glob.glob(str(images_dir / "*.jpg"))
    all_image_infos = []
    for image_path in tqdm(all_images_paths):
        file_name = os.path.basename(image_path)
        img_id = int(file_name.strip(".jpg"))

        img = Image.open(image_path)
        w, h = img.size

        all_image_infos.append(
            create_image_info(
                image_id=img_id,
                width=w,
                height=h,
                file_name=file_name,
                license=0,
                flickr_url="",
                coco_url="",
                data_captured="",
            )
        )
    all_images_infos = np.array(all_image_infos)
    all_json_image_ids = np.array([int(item["id"]) for item in all_images_infos])
    return all_images_infos, all_json_image_ids


def split_test_to_query_gallery(category_name: str, load_dir: str):
    test = load_json(load_dir / f"test_{category_name}.json")

    all_annotations = np.array(test["annotations"])
    all_annotations_ids = np.array([item["id"] for item in test["annotations"]])
    users_annotations_ids = np.array(
        [
            item["id"]
            for item in test["annotations"]
            if item["source"] == "user"
            and item["style"]
            >= 0  ### TODO Style is deep fasion 2 attribute may be deleted
        ]
    )
    users_annotations = list(
        all_annotations[np.isin(all_annotations_ids, users_annotations_ids)]
    )
    users_annos_image_ids = np.unique([item["image_id"] for item in users_annotations])

    gallery_annotations = list(
        all_annotations[
            np.isin(all_annotations_ids, users_annotations_ids, invert=True)
        ]
    )
    gallery_annos_image_ids = np.unique(
        [item["image_id"] for item in gallery_annotations]
    )

    all_images = np.array(test["images"])
    print(f"Number of all images in the set: {len(all_images)}")
    all_images_ids = np.array([item["id"] for item in all_images])
    user_images = all_images[np.isin(all_images_ids, users_annos_image_ids)]
    gallery_images = all_images[np.isin(all_images_ids, gallery_annos_image_ids)]
    print(f"Number of users image ids after filtering images: {len(user_images)}")
    print(f"Number of gallery image ids after filtering images: {len(gallery_images)}")

    query = test.copy()
    query["images"] = list(user_images)
    query["annotations"] = list(users_annotations)
    gallery = test.copy()
    gallery["images"] = list(gallery_images)
    gallery["annotations"] = list(gallery_annotations)
    query_annotations_pair_ids = np.array(
        [item["pair_id"] for item in users_annotations]
    )
    gallery_annotations_pair_ids = np.array(
        [item["pair_id"] for item in gallery_annotations]
    )

    print(f"Len query['images']: {len(query['images'])}")
    print(f"Len query['annotations']: {len(query['annotations'])}")
    print(f"Number unique pair_ids: {len(np.unique(query_annotations_pair_ids))}")
    print(f"Len gallery['images']: {len(gallery['images'])}")
    print(f"Len gallery['annotations']: {len(gallery['annotations'])}")
    print(f"Number unique pair_ids: {len(np.unique(gallery_annotations_pair_ids))}")

    with open(load_dir / f"./query_{category_name}.json", "w") as f:
        json.dump(query, f)
    with open(load_dir / f"./gallery_{category_name}.json", "w") as f:
        json.dump(gallery, f)


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
    x1, y1, bbox_w, bbox_h = bbox_int[:4]
    x2 = x1 + bbox_w
    y2 = y1 + bbox_h
    cut_arr = img_arr[y1:y2, x1:x2]  # Cut-out the instance detected
    im = Image.fromarray(cut_arr)
    cut_arr = _resize_thumbnail(im, target_image_size)

    return cut_arr


def crop_train_images(
    root_dir: str,
    images_dir: str,
    save_dir,
    TARGET_IMAGE_SIZE,
    MINIMUM_BBOX_AREA,
):
    """
    When we crop the train images we need to assign them new pair-ids as there may be more
    than one produt in the image.
    """
    pair_id_temp_dict = {}  # To store (old_pairid, style) to map to new pair_ids
    next_image_id = 1
    next_anno_id = 1
    next_pair_id = 0

    for category_name in ORIGINAL_CATEGORIES:
        TARGET_IMAGES_DIR = (
            root_dir
            / f"images_cropped_{TARGET_IMAGE_SIZE[0]}_{TARGET_IMAGE_SIZE[1]}/{category_name}/"
        )
        TARGET_TRAIN_IMAGES_DIR = TARGET_IMAGES_DIR / "train"
        GALLERY_IMAGES_DIR = TARGET_IMAGES_DIR / "gallery"
        QUERY_IMAGES_DIR = TARGET_IMAGES_DIR / "query"
        GALLERY_IMAGES_DIR.mkdir(exist_ok=True, parents=True)
        QUERY_IMAGES_DIR.mkdir(exist_ok=True, parents=True)
        TARGET_TRAIN_IMAGES_DIR.mkdir(exist_ok=True, parents=True)

        train = load_json(save_dir / f"train_{category_name}.json")
        query = load_json(save_dir / f"query_{category_name}.json")
        gallery = load_json(save_dir / f"gallery_{category_name}.json")

        datasets_for_recalculating = [train, query, gallery]
        save_paths_names = [
            TARGET_TRAIN_IMAGES_DIR,
            QUERY_IMAGES_DIR,
            GALLERY_IMAGES_DIR,
        ]

        for name, IMAGE_SAVE_DIR, dataset in zip(
            SET_NAMES, save_paths_names, datasets_for_recalculating
        ):
            images_info = []
            annotations_list = []

            for _, image_info in enumerate(dataset["images"]):
                im_id = image_info["id"]

                im_filename = image_info["file_name"]
                annos_per_img = [
                    anno for anno in dataset["annotations"] if anno["image_id"] == im_id
                ]

                for img_idx, anno in enumerate(annos_per_img):
                    anno_style = anno["style"]
                    anno_pair_id = anno["pair_id"]
                    anno_cat_id = anno["category_id"]
                    anno_bbox = anno["bbox"]
                    anno_area = anno["area"]
                    anno_source = anno["source"]

                    old_filename, ext = os.path.splitext(im_filename)
                    new_filename = old_filename + f"_{anno_style}_{img_idx}" + ext

                    if os.path.isfile(
                        images_dir / im_filename
                    ):  #### TODO REMOVE THIS LINE
                        if not os.path.isfile(
                            IMAGE_SAVE_DIR / new_filename
                        ):  ## TODO This line remove
                            image_open = Image.open(images_dir / im_filename)
                            if anno_bbox != "":
                                anno_bbox = np.asarray(anno["bbox"]).astype(np.int32)
                                if (
                                    anno_bbox[3] != 0
                                    and anno_bbox[2] != 0
                                    and anno_area >= MINIMUM_BBOX_AREA
                                ):  # Remove all annotations that have width/height==0, or small area
                                    cropped = crop_single_bbox(
                                        image_open, anno_bbox, TARGET_IMAGE_SIZE
                                    )
                                else:
                                    continue
                            else:
                                cropped = _resize_thumbnail(
                                    image_open, TARGET_IMAGE_SIZE
                                )
                            cropped.save(IMAGE_SAVE_DIR / new_filename)

                    if (anno_pair_id, anno_style) not in pair_id_temp_dict:
                        pair_id_temp_dict[(anno_pair_id, anno_style)] = next_pair_id
                        new_pair_id = next_pair_id
                        next_pair_id += 1
                    else:
                        new_pair_id = pair_id_temp_dict[(anno_pair_id, anno_style)]

                    single_crop_info = create_image_info(
                        image_id=next_image_id,
                        width=TARGET_IMAGE_SIZE[0],
                        height=TARGET_IMAGE_SIZE[1],
                        file_name=new_filename,
                    )
                    single_crop_anno = create_annotations(
                        anno_id=next_anno_id,
                        image_id=next_image_id,
                        category_id=anno_cat_id,
                        bbox="",
                        pair_id=new_pair_id,
                        style=anno_style,
                        segmentation="",
                        area=anno_area,
                        iscrowd=0,
                        source=anno_source,
                    )
                    next_image_id += 1
                    next_anno_id += 1

                    images_info.append(single_crop_info)
                    annotations_list.append(single_crop_anno)

            dataset["images"] = images_info
            dataset["annotations"] = annotations_list

            with open(save_dir / f"{name}_{category_name}_cropped.json", "w") as f:
                json.dump(dataset, f)


def merge_single_set_jsons(
    set_name: str, ORIGINAL_CATEGORIES: List[str], json_load_dir: str
):
    global_json = {}
    anno_id = 0
    all_annos = []
    all_image_ids = []
    all_images_info = []

    for category_name in ORIGINAL_CATEGORIES:
        set_single_category_json = load_json(
            json_load_dir / f"{set_name}_{category_name}_cropped.json"
        )
        set_single_category_json_images_ids = np.array(
            [item["id"] for item in set_single_category_json["images"]]
        )
        set_single_category_all_images_infos = np.array(
            copy.deepcopy(set_single_category_json["images"])
        )

        for item in set_single_category_json["annotations"]:
            image_id = item.get("image_id")
            all_image_ids.append(image_id)
            anno_id += 1
            item["id"] = anno_id
            all_annos.append(item)

        all_json_image_info_to_take_inds = np.isin(
            set_single_category_json_images_ids, all_image_ids
        )
        all_json_image_info_to_take = list(
            set_single_category_all_images_infos[all_json_image_info_to_take_inds]
        )
        all_images_info.extend(all_json_image_info_to_take)

    print(f"Number of all annotations in the set {set_name}: {len(all_annos)}")
    print(f"Number of all images in the set {set_name}: {len(all_image_ids)}")
    all_image_ids = list(np.unique(all_image_ids))
    all_image_ids = [int(item) for item in all_image_ids]
    print(f"Number of UNIQUE images in the set {set_name}: {len(all_image_ids)}")

    all_json_image_info_image_filenames = [
        item["file_name"] for item in all_images_info
    ]

    print(
        f"Number of all images file_names in images: {len(all_json_image_info_image_filenames)}"
    )
    print(
        f"Number of unique images file_names in images: {len(np.unique(all_json_image_info_image_filenames))}"
    )

    global_json = copy.deepcopy(set_single_category_json)
    global_json["images"] = list(all_images_info)
    global_json["annotations"] = list(all_annos)

    with open(json_load_dir / f"{set_name}_coco_reid.json", "w") as f:
        json.dump(global_json, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to transform COCO-format Street2Shop annotations to ReID-ready COCO format."
    )
    parser.add_argument(
        "--train-json-path",
        help="path to json file produed by 'street2shop_to_coco.py' script",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--root-dir-path",
        help="path to root directory of Steet2Shop data",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--metadata-dir",
        help="directory name with Steet2Shop metadata",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--images-dir",
        help="directory name with Steet2Shop images",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save-dir",
        help="directory name where all new data will be saved",
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
    parser.add_argument(
        "--minimum-bbox-area",
        help=(
            "minimum area in pixels covered by bounding boxes. If the area is smaller than the threshold the annotation is discarded."
        ),
        type=int,
        required=False,
        default=1,
    )

    args = parser.parse_args()

    # all_train_json_load_path = (
    #     "/data/mwieczorek/home/data/street2shop/new_meta_test/all_street_train.json"
    # )
    # root_dir = Path("/data/mwieczorek/home/data/street2shop/")
    # meta_dir = root_dir / "meta"
    # images_dir = root_dir / "images"
    # save_dir = root_dir / "new_meta_dir"
    # TARGET_IMAGE_SIZE = (224, 244)
    # MINIMUM_BBOX_AREA = 1

    all_train_json_load_path = args.train_json_path
    root_dir = Path(args.root_dir_path)
    meta_dir = root_dir / args.metadata_dir
    images_dir = root_dir / args.images_dir
    save_dir = root_dir / args.save_dir
    TARGET_IMAGE_SIZE = tuple([int(item) for item in args.target_image_size])
    MINIMUM_BBOX_AREA = args.minimum_bbox_area
    os.makedirs(save_dir, exist_ok=True)

    # This file only contains images and annotations of street photos.
    train_json_coco = json.load(open(all_train_json_load_path, "r"))

    category_name2category_id = {k: idx for idx, k in enumerate(ORIGINAL_CATEGORIES)}

    log.info("Creating global product to pair-id mapping...")
    global_product_pair_id_map = create_global_to_pair_id_mapping(
        load_json, ORIGINAL_CATEGORIES, meta_dir, category_name2category_id
    )
    ### TODO Susbsitute for a proper function
    log.info(f"Creating coco-format images info of jpgs stored at {images_dir}...")
    all_images_infos, all_json_image_ids = create_info_for_all_images(images_dir)
    # all_images_infos = np.array(
    #     load_json("/data/mwieczorek/home/data/street2shop/all_images_info.json")
    # )
    all_json_image_ids = np.array([item["id"] for item in all_images_infos])
    remapped_datasets = remap_raw_coco_to_pair_ids(
        load_json, ORIGINAL_CATEGORIES, meta_dir, global_product_pair_id_map
    )

    for mode in ["train", "test"]:
        log.info(f"Running script for {mode} data subset...")
        if mode == "test":
            print(mode)
        for category_name in ORIGINAL_CATEGORIES:
            log.info(f"Currently processed category {category_name}.")
            create_coco_json_with_unique_pair_id_per_category(
                category_name2category_id=category_name2category_id,
                remapped_datasets=remapped_datasets,
                category_name=category_name,
                info=train_json_coco["info"],
                categories=train_json_coco["categories"],
                licenses=train_json_coco["licenses"],
                all_images_infos=all_images_infos,
                all_json_image_ids=all_json_image_ids,
                mode=mode,
                save_dir=save_dir,
            )

            if mode == "test":
                split_test_to_query_gallery(category_name, save_dir)

    log.info(
        f"Cropping and resizing images to {TARGET_IMAGE_SIZE}. This may take some time..."
    )

    crop_train_images(
        root_dir,
        images_dir,
        save_dir,
        TARGET_IMAGE_SIZE,
        MINIMUM_BBOX_AREA,
    )

    ### Scatter images to folder for reid training
    for set_name in SET_NAMES:
        merge_single_set_jsons(set_name, ORIGINAL_CATEGORIES, save_dir)

    CROPPED_IMAGES_SOURCE_ROOT_DIR = (
        root_dir / f"images_cropped_{TARGET_IMAGE_SIZE[0]}_{TARGET_IMAGE_SIZE[1]}"
    )
    CROPPED_IMAGES_TARGET_ROOT_DIR = (
        root_dir / f"images_reid_cropped_{TARGET_IMAGE_SIZE[0]}_{TARGET_IMAGE_SIZE[1]}"
    )
    log.info(
        f"Final processing of images. Scattering them to correctly arranged folders. May take some time..."
    )
    for name in SET_NAMES:
        log.info(f"Processing set: {name}")
        data_set = load_json(save_dir / f"{name}_coco_reid.json")
        IMAGES_TARGET_DIR = CROPPED_IMAGES_TARGET_ROOT_DIR / f"{name}"
        IMAGES_TARGET_DIR.mkdir(exist_ok=True, parents=True)
        log.info(f"Saving images from {name} subset to {IMAGES_TARGET_DIR}")

        for image_info in data_set["images"]:
            file_name = image_info["file_name"]
            for category_name in ORIGINAL_CATEGORIES:
                IMAGES_SOURCE_DIR = (
                    CROPPED_IMAGES_SOURCE_ROOT_DIR / f"{category_name}/{name}"
                )
                source_path = IMAGES_SOURCE_DIR / file_name
                if os.path.isfile(source_path):
                    target_path = IMAGES_TARGET_DIR / file_name
                    if not os.path.isfile(target_path):
                        shutil.copy(source_path, target_path)
