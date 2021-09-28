# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T
from .random_erasing import RandomErasing

class ReidTransforms():

    def __init__(self, cfg):
        self.cfg = cfg

    def build_transforms(self, is_train=True):
        normalize_transform = T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)
        if is_train:
            transform = T.Compose([
                T.Resize(self.cfg.INPUT.SIZE_TRAIN),
                T.RandomHorizontalFlip(p=self.cfg.INPUT.PROB),
                T.Pad(self.cfg.INPUT.PADDING),
                T.RandomCrop(self.cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                normalize_transform,
                RandomErasing(probability=self.cfg.INPUT.RE_PROB, mean=self.cfg.INPUT.PIXEL_MEAN)
            ])
        else:
            transform = T.Compose([
                T.Resize(self.cfg.INPUT.SIZE_TEST),
                T.ToTensor(),
                normalize_transform
            ])

        return transform