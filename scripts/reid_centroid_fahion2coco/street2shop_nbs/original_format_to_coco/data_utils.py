import os
import json
from PIL import Image
import glob
import itertools
import numpy as np


def check_if_file_exist(path):
    return os.path.isfile(path)


def check_dir_existance_create(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_json(json_abs_path):
    return json.load(open(json_abs_path))


def save_json(obj_to_save, save_path, mode="w"):
    file_handler = open(save_path, mode)
    json.dump(obj_to_save, file_handler)


def search_in_dir_for_files(dir_path, search_phrase):
    return sorted(glob.glob(os.path.join(dir_path, f"{search_phrase}")))


def save_images_paths_to_txt(save_dir_path, save_filename, txt_file_list):
    with open(os.path.join(save_dir_path, f"{save_filename}"), "w") as f:
        for item in txt_file_list:
            f.write(item + "\n")


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


def get_used_images_ids(identifier, data):
    # identifier  :  json identifier used to harvest the images
    # data  :  just data in json.dump format
    return [item[identifier] for item in data]


def remove_all_annotation_files(path, ext):
    os.system(f'find {path} -maxdepth 1 -name "*.{ext}" -delete ')


def create_class_index_dict(list_of_labels):
    class_label_dict = {}
    for idx, class_name in enumerate(list_of_labels):
        class_label_dict[class_name] = idx
    return class_label_dict


def search_for_image_dimension(images_data, identifier, img_searched):
    # Function to find an image corressponding to an annotation
    # Both are supposed to be in json dumps formats
    # Where image and annotation can be matched by common indentifier
    for idx, img_data in enumerate(images_data):
        if img_data[identifier] == img_searched:
            width = img_data["width"]
            height = img_data["height"]
            break
    return width, height, idx


def load_image(img_path):
    return Image.open(img_path)


def load_all_images_paths_from_txt(path, mode="strip"):
    with open(path, "r") as f:
        lines = f.read().splitlines()

        if mode == "split":
            all_images = [item.split(",")[0] for item in lines]
            return all_images
        elif mode == "strip":
            return lines


def get_all_images_in_dir(dir_path, ext):
    return sorted(list(glob.glob(os.path.join(dir_path, f"*.{ext}"))))


def get_images_size(path):
    image = Image.open(path)
    return image.size


def get_image_id(image_name):
    return image_name.split(".")[0].lstrip("0")


def remove_duplicated_path(train_file_path):
    with open(train_file_path, "r") as f:
        lines = f.readlines()
    lines = [item.strip() for item in lines]

    lines = list(set(lines))

    with open(train_file_path, "w") as f:
        for line in lines:
            f.write(line + "\n")


def check_if_images_exists_save(images_paths_list, train_file_path=None, save=False):
    existng_images_list = []
    for img in images_paths_list:
        img = img.replace("txt", "jpg")
        if os.path.exists(img):
            existng_images_list.append(img)
    if save:
        with open(train_file_path, "w") as f:
            for item in existng_images_list:
                f.write(item.replace("jpg", "txt") + "\n")

    return existng_images_list


def detect_corrupted_images(path):

    ## To verify images
    images_to_remove = []
    for idx, image_name in enumerate(list(glob.glob(os.path.join(path, "*.jpg")))):
        image_path = os.path.join(path, image_name)
        try:
            width, height = load_image(image_path).size
        except (IOError, SyntaxError):
            images_to_remove.append(image_path)

    print(images_to_remove)

    return images_to_remove


def remove_greyscale_images(images_paths_list):
    grey_scale_images_to_remove = []
    for path in images_paths_list:
        path = os.path.splitext(path)[0] + ".jpg"
        try:
            img = Image.open(path)
            if img.mode == "L":
                print(path)
                grey_scale_images_to_remove.append(path)
        except FileNotFoundError:
            print(f"File not found: {path}")

    for filename in grey_scale_images_to_remove:
        filename = os.path.splitext(filename)[0]
        filename_txt = filename + ".txt"
        filename_img = filename + ".jpg"
        os.system(f"rm {filename_txt}")
        os.system(f"rm {filename_img}")


def select_products_ids(categories, meta_dir):
    product_photos = set()  # Set to disallow duplicates
    for category in categories:
        filepath = os.path.join(meta_dir, f"retrieval_{category}.json")
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


def create_index_of_images(save_dir, product_filenames):
    temp = []
    for item in zip(itertools.count(), product_filenames):
        temp.append(item)
    np.save(os.path.join(save_dir, f"index.npy"), np.array(temp))
