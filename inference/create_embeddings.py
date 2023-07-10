import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(".")

from config import cfg
from train_ctl_model import CTLModel

from inference_utils import (
    ImageDataset,
    ImageFolderWithPaths,
    calculate_centroids,
    create_pid_path_index,
    make_inference_data_loader,
    run_inference,
)

### Prepare logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

### Functions used to extract pair_id
exctract_func = (
    lambda x: (x).rsplit(".", 1)[0].rsplit("_", 1)[0]
)  ## To extract pid from filename. Example: /path/to/dir/product001_04.jpg -> pid = product001
exctract_func = lambda x: Path(
    x
).parent.name  ## To extract pid from parent directory of an iamge. Example: /path/to/root/001/image_04.jpg -> pid = 001

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create embeddings for images that will serve as the database (gallery)"
    )
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "--images-in-subfolders",
        help="if images are stored in the subfloders use this flag. If images are directly under DATASETS.ROOT_DIR path do not use it.",
        action="store_true",
    )
    parser.add_argument(
        "--print_freq",
        help="number of batches the logging message is printed",
        type=int,
        default=10,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)


    ### Data preparation

    if args.images_in_subfolders:
        dataset_type = ImageFolderWithPaths
    else:
        dataset_type = ImageDataset

    log.info(f"Preparing data using {dataset_type} dataset class")

    val_loader = make_inference_data_loader(cfg, cfg.DATASETS.ROOT_DIR, dataset_type)

    if len(val_loader) == 0:
        raise RuntimeError("Lenght of dataloader = 0")


    ### Build model

    model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH)
    use_cuda = True if torch.cuda.is_available() and cfg.GPU_IDS else False

    ### Inference

    log.info("Running inference")
    embeddings, paths = run_inference(
        model, val_loader, cfg, print_freq=args.print_freq, use_cuda=use_cuda
    )


    ### Create centroids

    log.info("Creating centroids")
    if cfg.MODEL.USE_CENTROIDS:
        pid_path_index = create_pid_path_index(paths=paths, func=exctract_func)
        embeddings, paths = calculate_centroids(embeddings, pid_path_index)


    ### Save

    SAVE_DIR = Path(cfg.OUTPUT_DIR)
    SAVE_DIR.mkdir(exist_ok=True, parents=True)

    log.info(f"Saving embeddings and index to {str(SAVE_DIR)}")
    np.save(SAVE_DIR / "embeddings.npy", embeddings)
    np.save(SAVE_DIR / "paths.npy", paths)
