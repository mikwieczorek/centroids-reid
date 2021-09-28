# encoding: utf-8
"""
Partially based on work by:
@author:  sherlock
@contact: sherlockliao01@gmail.com

Adapted and extended by:
@author: mikwieczorek
"""

import glob
import json
import os.path as osp
import random
import re
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SequentialSampler
from tqdm import tqdm

from .samplers import get_sampler
from .transforms import ReidTransforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ReidBaseDataModule(pl.LightningDataModule):
    """
    Base class for reid datasets
    """

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.num_workers = kwargs.get("num_workers") if "num_workers" in kwargs else 6
        self.num_instances = (
            kwargs.get("num_instances") if "num_instances" in kwargs else 4
        )

    def _get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid, *_ in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def _print_dataset_statistics(self, train, query=None, gallery=None):
        num_train_pids, num_train_imgs, num_train_cams = self._get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self._get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self._get_imagedata_info(
            gallery
        )

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print(
            "  train    | {:5d} | {:8d} | {:9d}".format(
                num_train_pids, num_train_imgs, num_train_cams
            )
        )
        print(
            "  query    | {:5d} | {:8d} | {:9d}".format(
                num_query_pids, num_query_imgs, num_query_cams
            )
        )
        print(
            "  gallery  | {:5d} | {:8d} | {:9d}".format(
                num_gallery_pids, num_gallery_imgs, num_gallery_cams
            )
        )
        print("  ----------------------------------------")

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def train_dataloader(
        self, cfg, trainer, sampler_name: str = "random_identity", **kwargs
    ):
        if trainer.distributed_backend == "ddp_spawn":
            rank = trainer.root_gpu
        else:
            rank = trainer.local_rank
        world_size = trainer.num_nodes * trainer.num_processes
        sampler = get_sampler(
            sampler_name,
            data_source=self.train_dict,
            batch_size=self.cfg.SOLVER.IMS_PER_BATCH,
            num_instances=self.num_instances,
            world_size=world_size,
            rank=rank,
        )
        return DataLoader(
            self.train,
            self.cfg.SOLVER.IMS_PER_BATCH,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=sampler,
            collate_fn=collate_fn_alternative,
            **kwargs,
        )

    def val_dataloader(self):
        sampler = SequentialSampler(
            self.val
        )  ## This get replaced with ddp mode by lightning
        return DataLoader(
            self.val,
            self.cfg.TEST.IMS_PER_BATCH,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=False,
        )

    def test_dataloader(self):
        sampler = SequentialSampler(
            self.train
        )  ## This get replaced with ddp mode by lightning
        return DataLoader(
            self.train,
            self.cfg.TEST.IMS_PER_BATCH,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=False,
        )

    @staticmethod
    def _load_json(path):
        with open(path, "r") as f:
            js = json.load(f)

        return js


class COCODatasetBase(ReidBaseDataModule):
    """
    Dataset class to load COCO-format data into re-id model
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        assert (
            cfg.DATASETS.JSON_TRAIN_PATH != ""
        ), "DATASETS.JSON_TRAIN_PATH is not specified in the config"

        self.dataset_dir = cfg.DATASETS.ROOT_DIR
        self.json_train_path = cfg.DATASETS.JSON_TRAIN_PATH
        self.json_query_path = self.json_train_path.replace("train", "query")
        self.json_gallery_path = self.json_train_path.replace("train", "gallery")
        self.train_dir = osp.join(self.dataset_dir, "train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "gallery")
        self.use_resampling = cfg.DATALOADER.USE_RESAMPLING

    def setup(self):
        self._check_before_run()
        transforms_base = ReidTransforms(self.cfg)

        self.train_json = self._load_json(self.json_train_path)
        self.query_json = self._load_json(
            self.json_query_path
        )  ## Just replace train with proper set name. JSONs filenames should  be consistent
        self.gallery_json = self._load_json(
            self.json_gallery_path
        )  ## Just replace train with proper set name. JSONs filenames should  be consistent

        train, train_dict = self._process_dir(self.train_dir, self.train_json)
        self.train_dict = train_dict
        self.train_list = train
        gallery, gallery_dict = self._process_dir(self.gallery_dir, self.gallery_json)
        query, query_dict = self._process_dir(self.query_dir, self.query_json)
        self.query_list = query
        self.gallery_list = gallery

        self.train = BaseDatasetLabelledPerPid(
            train_dict,
            transforms_base.build_transforms(is_train=True),
            self.num_instances,
            self.use_resampling,
        )  ## Only train needs to be alternative dataset and sampler
        self.val = BaseDatasetLabelled(
            query + gallery, transforms_base.build_transforms(is_train=False)
        )

        self._print_dataset_statistics(train, query, gallery)
        # For reid_metic to evaluate properly
        num_train_pids, num_train_imgs, num_train_cams = self._get_imagedata_info(train)

        self.num_query = len(query)
        self.num_classes = num_train_pids

    def _check_before_run(self):
        super()._check_before_run()
        if not osp.exists(self.json_train_path):
            raise RuntimeError("'{}' is not available".format(self.json_train_path))
        if not osp.exists(self.json_query_path):
            raise RuntimeError("'{}' is not available".format(self.json_query_path))
        if not osp.exists(self.json_gallery_path):
            raise RuntimeError("'{}' is not available".format(self.json_gallery_path))

    def _process_dir(self, images_path, json_file, relabel=False):
        if "gallery" in images_path.lower():
            camid = 1  ## For all gallery images
        else:
            camid = 0  ## For train and query images

        annotations_pair_ids = np.array(
            [item["pair_id"] for item in json_file["annotations"]]
        )
        _unique_pair_ids_set = list(set(annotations_pair_ids))
        image_ids = np.array([item["image_id"] for item in json_file["annotations"]])
        image_info_ids = np.array([item["id"] for item in json_file["images"]])
        image_info = np.array(json_file["images"])
        image_filenames = np.array([item["file_name"] for item in json_file["images"]])

        image_ids_dict = {k: v for v, k in enumerate(image_info_ids)}

        len_data = len(image_info_ids)

        if "train" in images_path.lower():
            relabel = True
            mode = "train"
        elif "query" in images_path.lower():
            mode = "query"
        else:
            mode = "gallery"

        if mode == "train":
            unique_pair_ids_set = set()
            print("Filtering train dataset to remove pair_ids with only 1 image...")
            n = 0
            for idx, pair_id in enumerate(tqdm(_unique_pair_ids_set)):
                assert pair_id >= 0

                inds = np.where(annotations_pair_ids == pair_id)[0]
                image_ids_selected = image_ids[inds]
                image_info_inds = [
                    image_ids_dict[id_]
                    for id_ in image_ids_selected
                    if id_ in image_ids_dict
                ]
                image_filenames_selected = image_filenames[image_info_inds]
                num_files = len(image_filenames_selected)

                if num_files <= 1:
                    n += 1
                    continue

                unique_pair_ids_set.add(pair_id)
            print(f"Filtered out {n} pair ids with single image")
        else:
            unique_pair_ids_set = _unique_pair_ids_set

        unique_pair_ids_set = sorted(list(unique_pair_ids_set))

        if relabel:
            pid2label = {pid: label for label, pid in enumerate(unique_pair_ids_set)}

        dataset_dict = defaultdict(list)
        dataset = []
        len_ids = []
        for idx, pair_id in enumerate(tqdm(unique_pair_ids_set)):
            assert pair_id >= 0

            inds = np.where(annotations_pair_ids == pair_id)[0]
            image_ids_selected = image_ids[inds]

            image_info_inds = [
                image_ids_dict[id_]
                for id_ in image_ids_selected
                if id_ in image_ids_dict
            ]
            image_filenames_selected = image_filenames[image_info_inds]

            num_files = len(image_filenames_selected)
            len_ids.append(num_files)

            if relabel:
                pair_id = pid2label[pair_id]

            for image_entry in image_filenames_selected:
                image_path = osp.join(images_path, image_entry)
                dataset.append((image_path, pair_id, camid, mode))
                dataset_dict[pair_id].append((image_path, pair_id, camid, mode))

        return dataset, dataset_dict

    def train_dataloader(
        self, cfg, trainer, sampler_name: str = "random_identity", **kwargs
    ):
        if trainer.distributed_backend == "ddp_spawn":
            rank = trainer.root_gpu
        else:
            rank = trainer.local_rank
        world_size = trainer.num_nodes * trainer.num_processes
        sampler = get_sampler(
            sampler_name,
            data_source=self.train_dict,
            batch_size=self.cfg.SOLVER.IMS_PER_BATCH,
            num_instances=self.num_instances,
            world_size=world_size,
            rank=rank,
        )
        return DataLoader(
            self.train,
            self.cfg.SOLVER.IMS_PER_BATCH,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=sampler,
            collate_fn=collate_fn_alternative,
            **kwargs,
        )


class BaseDatasetLabelledPerPid(Dataset):
    def __init__(self, data, transform=None, num_instances=4, resample=False):
        self.samples = data
        self.transform = transform
        self.num_instances = num_instances
        self.resample = resample

    def __getitem__(self, pid):
        """
        Retrives self.num_instances per given pair_id
        Args:
            pid (int): Pair_id number actually

        Returns:
            num_instace of given pid
        """
        pid = int(pid)
        list_of_samples = self.samples[pid][
            :
        ]  # path, target, camid, idx <- in each inner tuple
        _len = len(list_of_samples)
        assert (
            _len > 1
        ), f"len of samples for pid: {pid} is <=1. len: {len_}, samples: {list_of_samples}"

        if _len < self.num_instances:
            choice_size = _len
            needPad = True
        else:
            choice_size = self.num_instances
            needPad = False

        # We shuffle self.samples[pid] as we extract instances from this dict directly
        random.shuffle(self.samples[pid])

        out = []
        for _ in range(choice_size):
            tup = self.samples[pid].pop(0)
            path, target, camid, idx = tup
            img = self.prepare_img(path)
            out.append(
                (img, target, camid, idx, True)
            )  ## True stand if the sample is real or mock

        if needPad:
            num_missing = self.num_instances - _len
            assert (
                num_missing != self.num_instances
            ), f"Number of missings sample in the batch is equal to num_instances. PID: {pid}"
            if self.resample:
                assert len(list_of_samples) > 0
                resampled = np.random.choice(
                    range(len(list_of_samples)), size=num_missing, replace=True
                )
                for idx in resampled:
                    path, target, camid, idx = list_of_samples[idx]
                    img = self.prepare_img(path)
                    out.append((img, target, camid, idx, True))
            else:
                img_mock = torch.zeros_like(img)
                for _ in range(num_missing):
                    out.append((img_mock, target, camid, idx, False))

        assert (
            len(out) == self.num_instances
        ), f"Number of returned tuples per id needs to be equal self.num_instance. It is: {len(out)}"

        return out

    def __len__(self):
        return len(self.samples) * self.num_instances

    def prepare_img(self, path):
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class BaseDatasetLabelled(Dataset):
    def __init__(self, data, transform=None, return_paths=False):
        self.samples = data
        self.transform = transform
        self.return_paths = return_paths

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, camid, idx = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_paths:
            return sample, target, camid, path
        else:
            return sample, target, camid, idx

    def __len__(self):
        return len(self.samples)


def collate_fn_alternative(batch):
    # imgs, pids, _, _, isReal = zip(*batch)
    imgs = [item[0] for sample in batch for item in sample]
    pids = [item[1] for sample in batch for item in sample]
    camids = [item[2] for sample in batch for item in sample]
    isReal = [item[4] for sample in batch for item in sample]

    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, torch.tensor(camids), torch.tensor(isReal)
