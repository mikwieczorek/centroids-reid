import os
from data_utils import *
from data_format_transforms import *
from coco_format_utils import COCO_json


def main():

    ### PARAMETERS
    working_dir = "/data/mwieczorek/home/data/street2shop/"
    save_dir = working_dir
    data_dir = os.path.join(working_dir, "images")
    images_dir = data_dir
    save_dir = os.path.join(working_dir, "new_meta_test")
    os.makedirs(save_dir, exist_ok=True)
    train_filename = "train_data.txt"
    test_filename = "test_data.txt"
    train_all_filename = "train_all.txt"
    train_file_path = os.path.join(save_dir, train_all_filename)
    coco_json_save_name = "all_street_train.json"
    coco_json_save_path = os.path.join(save_dir, coco_json_save_name)
    meta_dir = os.path.join(working_dir, "meta")
    categories_dict = load_json(
        "scripts/reid_centroid_fahion2coco/street2shop_nbs/original_format_to_coco/street2shop_categories.json"
    )

    # # Create single txt files with names of all images in train / test set
    extract_json_data(
        jsons_path=meta_dir,
        save_dir=save_dir,
        save_filename=train_filename,
        key_name="photo",
        ext="jpg",
        mode="train",
    )
    extract_json_data(
        jsons_path=meta_dir,
        save_dir=save_dir,
        save_filename=test_filename,
        key_name="photo",
        ext="jpg",
        mode="test",
    )

    # Merge train and test into one txt file
    sub_files = [train_filename, test_filename]
    merge_train_test_subsets(
        filenames=sub_files, save_dir=save_dir, save_filename=train_all_filename
    )

    # Removing corrupted/missing etc files
    # remove_duplicated_path(train_file_path=train_file_path)
    images_names = load_all_images_paths_from_txt(path=train_file_path)
    # images_paths_list = check_if_images_exists_save(
    #     images_paths_list=images_paths_list, train_file_path=train_file_path
    # )
    # remove_greyscale_images(images_paths_list)
    create_category_txt_filepaths(
        categories_dict=categories_dict,
        meta_dir=meta_dir,
        save_dir=save_dir,
        mode="single",
    )

    # Create COCO format annotations
    coco_json = COCO_json(
        images_dir=images_dir,
        save_dir=save_dir,
        categories_dict=categories_dict,
        sets=["train", "test"],
        images_names=images_names,
        meta_dir=meta_dir,
    )
    coco_json.create_full_coco_json(bbox_transform_func=transform_bbox_s2s_to_coco)
    save_json(coco_json.json, coco_json_save_path, mode="w")
    print("Street2Shop_to_coco processing finished")


if __name__ == "__main__":
    main()
