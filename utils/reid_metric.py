# encoding: utf-8
"""
Based on code from:
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com

Adapted and extended by:
@author: mikwieczorek
"""

import os
import warnings
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .eval_reid import eval_func

from .visrank import visualize_ranked_results


def get_euclidean(x, y, **kwargs):
    m = x.shape[0]
    n = y.shape[0]
    distmat = (
        torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat.addmm_(1, -2, x, y.t())
    return distmat


def cosine_similarity(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    Computes cosine similarity between two tensors.
    Value == 1 means the same vector
    Value == 0 means perpendicular vectors
    """
    x_n, y_n = x.norm(dim=1)[:, None], y.norm(dim=1)[:, None]
    x_norm = x / torch.max(x_n, eps * torch.ones_like(x_n))
    y_norm = y / torch.max(y_n, eps * torch.ones_like(y_n))
    sim_mt = torch.mm(x_norm, y_norm.transpose(0, 1))
    return sim_mt


def get_cosine(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Computes cosine distance between two tensors.
    The cosine distance is the inverse cosine similarity
    -> cosine_distance = abs(-cosine_distance) to make it
    similar in behaviour to euclidean distance
    """
    sim_mt = cosine_similarity(x, y, eps)
    return torch.abs(1 - sim_mt).clamp(min=eps)


def get_dist_func(func_name="euclidean"):
    if func_name == "cosine":
        dist_func = get_cosine
    elif func_name == "euclidean":
        dist_func = get_euclidean
    print(f"Using {func_name} as distance function during evaluation")
    return dist_func


class R1_mAP:
    def __init__(self, pl_module, num_query, max_rank=50, feat_norm=True):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.current_epoch = pl_module.trainer.current_epoch
        self.hparms = pl_module.hparams
        self.dist_func = get_dist_func(self.hparms.SOLVER.DISTANCE_FUNC)
        self.pl_module = pl_module

        try:
            self.save_root_dir = pl_module.trainer.logger.log_dir
        except:
            self.save_root_dir = pl_module.trainer.logger[0].log_dir

        try:
            self.dataset = pl_module.trainer.val_dataloaders[0].dataset.samples
        except:
            self.dataset = pl_module.trainer.test_dataloaders[0].dataset.samples

    # @staticmethod
    def _commpute_batches_double(self, qf, gf):
        gf_num = gf.shape[0]
        num_batches = (gf_num // 200000) + 35
        gf_batchsize = int((gf_num // num_batches))
        results = []

        if isinstance(qf, np.ndarray):
            qf = torch.from_numpy(qf).float().cuda()

        for i in tqdm(range(num_batches + 1)):
            gf_temp = gf[i * gf_batchsize : (i + 1) * gf_batchsize, :]

            if isinstance(gf_temp, np.ndarray):
                gf_temp = torch.from_numpy(gf_temp).float().cuda()

            distmat_temp = self.dist_func(x=qf, y=gf_temp)
            results.append(distmat_temp)
        return np.hstack(results)

    def compute(self, feats, pids, camids, respect_camids=False):
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[: self.num_query]
        q_pids = np.asarray(pids[: self.num_query])
        q_camids = np.asarray(camids[: self.num_query])
        # gallery
        gf = feats[self.num_query :]
        g_pids = np.asarray(pids[self.num_query :])
        g_camids = np.asarray(camids[self.num_query :])
        m, n = qf.shape[0], gf.shape[0]

        if n > 30000 and self.pl_module.hparams.MODEL.USE_CENTROIDS:
            print(f"Reid metric no-ranking. Computing batches as n > 30000")
            distmat = self._commpute_batches_double(qf, gf)
            indices = np.argsort(distmat, axis=1)
        else:
            distmat = self.dist_func(x=qf, y=gf)
            indices = np.argsort(distmat, axis=1)

        cmc, mAP, all_topk, single_performance = eval_func(
            indices, q_pids, g_pids, q_camids, g_camids, 50, respect_camids
        )

        if self.hparms.TEST.VISUALIZE == "yes":
            print("Start visualization...")
            visualize_ranked_results(
                distmat,
                self.dataset,
                "image",
                self.hparms,
                width=self.hparms.INPUT.SIZE_TEST[1],
                height=self.hparms.INPUT.SIZE_TEST[0],
                save_dir=os.path.join(self.hparms.OUTPUT_DIR, "visrank"),
                topk=self.hparms.TEST.VISUALIZE_TOPK,
            )

        return cmc, mAP, all_topk