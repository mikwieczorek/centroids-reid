import glob
import json
import os

import numpy as np
from PIL import Image

S2S_ORIGINAL_CATEGORIES = [
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


def load_json(json_abs_path):
    return json.load(open(json_abs_path))


def save_json(obj_to_save, save_path, mode="w"):
    file_handler = open(save_path, mode)
    json.dump(obj_to_save, file_handler)


def search_in_dir_for_files(dir_path, search_phrase):
    return sorted(glob.glob(os.path.join(dir_path, f"{search_phrase}")))


def extract_json_data(
    jsons_path,
    save_dir,
    save_filename,
    key_name="photo",
    ext="jpg",
    mode="train",
    zfill_num=9,
):
    """Create a txt file with all images names that are present in the given set of jsons and train/test sets"""
    jsons_list = sorted(
        search_in_dir_for_files(dir_path=jsons_path, search_phrase=f"{mode}_*.json")
    )

    with open(os.path.join(save_dir, save_filename), "w") as f:
        for jsonn_path in jsons_list:
            jsonn = load_json(jsonn_path)
            for item in jsonn:
                image_name = str(item.get(f"{key_name}")).zfill(zfill_num) + f".{ext}"
                f.write(image_name + "\n")


def merge_train_test_subsets(filenames, save_dir, save_filename):
    with open(os.path.join(save_dir, save_filename), "w") as f:
        for item in filenames:
            with open(os.path.join(save_dir, item), "r") as ff:
                temp = ff.readlines()
                f.writelines(temp)


def load_all_images_paths_from_txt(path, mode="strip"):
    with open(path, "r") as f:
        lines = f.read().splitlines()

        if mode == "split":
            all_images = [item.split(",")[0] for item in lines]
            return all_images
        elif mode == "strip":
            return lines


def get_images_size(path):
    try:
        image = Image.open(path)
    except:
        return (1, 1)
    return image.size


def get_image_id(image_name):
    return image_name.split(".")[0].lstrip("0")


def select_products_ids(categories, meta_dir):
    product_photos = set()  # Set to disallow duplicates
    for category in categories:
        filepath = os.path.join(meta_dir, "json", f"retrieval_{category}.json")
        json_file = json.load(open(filepath))
        for item in json_file:
            product_photos.add(item["photo"])

    return list(product_photos)


def create_category_txt_filepaths(categories_dict, meta_dir, save_dir, mode="single"):
    categories_jsons = list(categories_dict.keys())

    if mode == "all":
        temp_list = []
        temp_list.append(categories_jsons)

    for category in categories_jsons:
        if not type(category) == list:
            category = [category]

        produt_ids_list = select_products_ids(categories=category, meta_dir=meta_dir)
        product_filenames = [str(item).zfill(9) + ".jpg" for item in produt_ids_list]
        # product_filenames = [item for item in product_filenames]

        if mode == "all":
            category = "all"
        else:
            # category = categories_dict.get(category[0])[0]
            category = category[0]

        with open(os.path.join(save_dir, f"{category}_products.txt"), "w") as f:
            for product in product_filenames:
                f.write(product + "\n")


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
