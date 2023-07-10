import logging
import os
from typing import Callable, Dict, List, Union

import numpy as np
import torch
from datasets.transforms import ReidTransforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import is_image_file

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def get_all_images(path: Union[str, List[str]]) -> List[str]:
    print(path, len(os.listdir(path)))
    if os.path.isdir(path):
        images = os.listdir(path)
        images = [os.path.join(path, item) for item in images if is_image_file(item)]
        return images
    elif is_image_file(path):
        return [path]
    else:
        raise Exception(
            f"{path} is neither a path to a valid image file nor a path to folder containing images"
        )


class ImageFolderWithPaths(ImageFolder):
    """
    Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


class ImageDataset(Dataset):
    def __init__(self, dataset: str, transform=None, loader=pil_loader):
        self.dataset = get_all_images(dataset)
        print(self.dataset)
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return (
            img,
            "",
            img_path,
        )  ## Hack to be consistent with ImageFolderWithPaths dataset


def make_inference_data_loader(cfg, path, dataset_class):
    transforms_base = ReidTransforms(cfg)

    val_transforms = transforms_base.build_transforms(is_train=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS

    val_set = dataset_class(path, val_transforms)

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
    )

    return val_loader


def _inference(model, batch, use_cuda, normalize_with_bn=True):
    model.eval()
    with torch.no_grad():
        data, _, filename = batch
        _, global_feat = model.backbone(
            data.cuda() if use_cuda else data
        )
        if normalize_with_bn:
            global_feat = model.bn(global_feat)
        return global_feat, filename


def run_inference(model, val_loader, cfg, print_freq, use_cuda):
    embeddings = []
    paths = []
    model = model.cuda() if use_cuda else model

    for pos, x in enumerate(val_loader):
        if pos % print_freq == 0:
            log.info(f"Number of processed images: {pos*cfg.TEST.IMS_PER_BATCH}")

        embedding, path = _inference(model, x, use_cuda)

        for vv, pp in zip(embedding, path):
            paths.append(pp)
            embeddings.append(vv.detach().cpu().numpy())

    embeddings = np.array(np.vstack(embeddings))
    paths = np.array(paths)

    return embeddings, paths


def create_pid_path_index(
    paths: List[str], func: Callable[[str], str]
) -> Dict[str, list]:
    paths_pids = [func(item) for item in paths]
    pid2paths_index = {}
    for idx, item in enumerate(paths_pids):
        if item not in pid2paths_index:
            pid2paths_index[item] = [idx]
        else:
            pid2paths_index[item].append(idx)
    return pid2paths_index


def calculate_centroids(embeddings, pid_path_index):
    pids_centroids_inds = []
    centroids = []
    for pid, indices in pid_path_index.items():
        inds = np.array(indices)
        pids_vecs = embeddings[inds]
        length = pids_vecs.shape[0]
        centroid = np.sum(pids_vecs, 0) / length
        pids_centroids_inds.append(pid)
        centroids.append(centroid)
    centroids_arr = np.vstack(np.array(centroids))
    pids_centroids_inds = np.array(pids_centroids_inds, dtype=np.str_)
    return centroids_arr, pids_centroids_inds
