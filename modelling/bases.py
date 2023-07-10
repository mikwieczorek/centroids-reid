# encoding: utf-8
"""
@author: mikwieczorek
"""

import copy
import os
import random
from collections import defaultdict
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import AttributeDict, rank_zero_only
from torch import tensor
from tqdm import tqdm

from config import cfg
from losses.center_loss import CenterLoss
from losses.triplet_loss import CrossEntropyLabelSmooth, TripletLoss
from modelling.baseline import Baseline
from solver import build_optimizer, build_scheduler
from utils.reid_metric import R1_mAP


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class ModelBase(pl.LightningModule):
    def __init__(self, cfg=None, test_dataloader=None, **kwargs):
        super().__init__()

        if cfg is None:
            hparams = {**kwargs}
        elif isinstance(cfg, dict):
            hparams = {**cfg, **kwargs}
            if cfg.TEST.ONLY_TEST:
                # To make sure that loaded hparams are overwritten by cfg we may have chnaged
                hparams = {**kwargs, **cfg}

        self.hparams.update(hparams)
        self.save_hyperparameters(self.hparams)

        if test_dataloader is not None:
            self.test_dataloader = test_dataloader

        # Create backbone model
        self.backbone = Baseline(self.hparams)

        self.contrastive_loss = TripletLoss(
            self.hparams.SOLVER.MARGIN, self.hparams.SOLVER.DISTANCE_FUNC
        )

        d_model = self.hparams.MODEL.BACKBONE_EMB_SIZE
        self.xent = CrossEntropyLabelSmooth(num_classes=self.hparams.num_classes)
        self.center_loss = CenterLoss(
            num_classes=self.hparams.num_classes, feat_dim=d_model
        )
        self.center_loss_weight = self.hparams.SOLVER.CENTER_LOSS_WEIGHT

        self.bn = torch.nn.BatchNorm1d(d_model)
        self.bn.bias.requires_grad_(False)

        self.fc_query = torch.nn.Linear(d_model, self.hparams.num_classes, bias=False)
        self.fc_query.apply(weights_init_classifier)

        self.losses_names = ["query_xent", "query_triplet", "query_center"]
        self.losses_dict = {n: [] for n in self.losses_names}

    @staticmethod
    def _calculate_centroids(vecs, dim=1):
        length = vecs.shape[dim]
        return torch.sum(vecs, dim) / length

    def configure_optimizers(self):
        optimizers_list = build_optimizer(self.named_parameters(), self.hparams)
        self.lr_scheduler = build_scheduler(optimizers_list[0], self.hparams)
        return optimizers_list, self.lr_scheduler

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
        **kwargs,
    ):

        if self.hparams.SOLVER.USE_WARMUP_LR:
            if epoch < self.hparams.SOLVER.WARMUP_EPOCHS:
                lr_scale = min(
                    1.0, float(epoch + 1) / float(self.hparams.SOLVER.WARMUP_EPOCHS)
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.hparams.SOLVER.BASE_LR

        super().optimizer_step(
            epoch=epoch,
            optimizer=optimizer,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
            optimizer_closure=optimizer_closure,
            on_tpu=on_tpu,
            using_native_amp=using_native_amp,
            using_lbfgs=using_lbfgs,
            **kwargs,
        )

    def training_step(self, batch, batch_idx, opt_idx=None):
        raise NotImplementedError(
            "A used model should have its own training_step method implemented"
        )

    def training_epoch_end(self, outputs):
        if hasattr(self.trainer.train_dataloader.sampler, "set_epoch"):
            self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch + 1)

        lr = self.lr_scheduler.get_last_lr()[0]
        loss = torch.stack([x.pop("loss") for x in outputs]).mean().cpu().detach()
        epoch_dist_ap = np.mean([x["other"].pop("step_dist_ap") for x in outputs])
        epoch_dist_an = np.mean([x["other"].pop("step_dist_an") for x in outputs])
        l2_mean_norm = np.mean([x["other"].pop("l2_mean_centroid") for x in outputs])

        del outputs

        log_data = {
            "epoch_train_loss": float(loss),
            "epoch_dist_ap": epoch_dist_ap,
            "epoch_dist_an": epoch_dist_an,
            "lr": lr,
            "l2_mean_centroid": l2_mean_norm,
        }

        if hasattr(self, "losses_dict"):
            for name, loss_val in self.losses_dict.items():
                val_tmp = np.mean(loss_val)
                log_data.update({name: val_tmp})
                self.losses_dict[name] = []  ## Zeroing values after a completed epoch

        self.trainer.logger.log_metrics(log_data, step=self.trainer.current_epoch)
        self.trainer.accelerator_backend.barrier()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        self.backbone.eval()
        self.bn.eval()
        x, class_labels, camid, idx = batch
        with torch.no_grad():
            _, emb = self.backbone(x)
            emb = self.bn(emb)
        return {"emb": emb, "labels": class_labels, "camid": camid, "idx": idx}

    @rank_zero_only
    def validation_create_centroids(
        self, embeddings, labels, camids, respect_camids=False
    ):
        num_query = self.hparams.num_query
        # Keep query data samples seperated
        embeddings_query = embeddings[:num_query].cpu()
        labels_query = labels[:num_query]

        # Process gallery samples further
        embeddings_gallery = embeddings[num_query:]
        labels_gallery = labels[num_query:]

        labels2idx = defaultdict(list)
        for idx, label in enumerate(labels_gallery):
            labels2idx[label].append(idx)

        labels2idx_q = defaultdict(list)
        for idx, label in enumerate(labels_query):
            labels2idx_q[label].append(idx)

        unique_labels = sorted(np.unique(list(labels2idx.keys())))

        centroids_embeddings = []
        centroids_labels = []

        if respect_camids:
            centroids_camids = []
            query_camid = camids[:num_query]

        # Create centroids for each pid seperately
        for label in unique_labels:
            cmaids_combinations = set()
            inds = labels2idx[label]
            inds_q = labels2idx_q[label]
            if respect_camids:
                selected_camids_g = camids[inds]

                selected_camids_q = camids[inds_q]
                unique_camids = sorted(np.unique(selected_camids_q))

                for current_camid in unique_camids:
                    # We want to select all gallery images that comes from DIFFERENT cameraId
                    camid_inds = np.where(selected_camids_g != current_camid)[0]
                    if camid_inds.shape[0] == 0:
                        continue
                    used_camids = sorted(
                        np.unique(
                            [cid for cid in selected_camids_g if cid != current_camid]
                        )
                    )
                    if tuple(used_camids) not in cmaids_combinations:
                        cmaids_combinations.add(tuple(used_camids))
                        centroids_emb = embeddings_gallery[inds][camid_inds]
                        centroids_emb = self._calculate_centroids(centroids_emb, dim=0)
                        centroids_embeddings.append(centroids_emb.detach().cpu())
                        centroids_camids.append(used_camids)
                        centroids_labels.append(label)

            else:
                centroids_labels.append(label)
                centroids_emb = embeddings_gallery[inds]
                centroids_emb = self._calculate_centroids(centroids_emb, dim=0)
                centroids_embeddings.append(centroids_emb.detach().cpu())

        # Make a single tensor from query and gallery data
        centroids_embeddings = torch.stack(centroids_embeddings).squeeze()
        centroids_embeddings = torch.cat(
            (embeddings_query, centroids_embeddings), dim=0
        )
        centroids_labels = np.hstack((labels_query, np.array(centroids_labels)))

        if respect_camids:
            query_camid = [[item] for item in query_camid]
            centroids_camids = query_camid + centroids_camids

        if not respect_camids:
            # Create dummy camids for query na gallery features
            # it is used in eval_reid script
            camids_query = np.zeros_like(labels_query)
            camids_gallery = np.ones_like(np.array(centroids_labels))
            centroids_camids = np.hstack((camids_query, np.array(camids_gallery)))

        return centroids_embeddings.cpu(), centroids_labels, centroids_camids

    @rank_zero_only
    def get_val_metrics(self, embeddings, labels, camids):
        self.r1_map_func = R1_mAP(
            pl_module=self,
            num_query=self.hparams.num_query,
            feat_norm=self.hparams.TEST.FEAT_NORM,
        )
        respect_camids = (
            True
            if (
                self.hparams.MODEL.KEEP_CAMID_CENTROIDS
                and self.hparams.MODEL.USE_CENTROIDS
            )
            else False
        )
        cmc, mAP, all_topk = self.r1_map_func.compute(
            feats=embeddings.float(),
            pids=labels,
            camids=camids,
            respect_camids=respect_camids,
        )

        topks = {}
        for top_k, kk in zip(all_topk, [1, 5, 10, 20, 50]):
            print("top-k, Rank-{:<3}:{:.1%}".format(kk, top_k))
            topks[f"Top-{kk}"] = top_k
        print(f"mAP: {mAP}")

        log_data = {"mAP": mAP}

        # TODO This line below is hacky, but it works when grad_monitoring is active
        self.trainer.logger_connector.callback_metrics.update(log_data)
        log_data = {**log_data, **topks}
        self.trainer.logger.log_metrics(log_data, step=self.trainer.current_epoch)

    def validation_epoch_end(self, outputs):
        if self.trainer.global_rank == 0 and self.trainer.local_rank == 0:
            embeddings = torch.cat([x.pop("emb") for x in outputs]).detach().cpu()
            labels = (
                torch.cat([x.pop("labels") for x in outputs]).detach().cpu().numpy()
            )
            camids = torch.cat([x.pop("camid") for x in outputs]).cpu().detach().numpy()
            del outputs
            if self.hparams.MODEL.USE_CENTROIDS:
                print("Evaluation is done using centroids")
                embeddings, labels, camids = self.validation_create_centroids(
                    embeddings,
                    labels,
                    camids,
                    respect_camids=self.hparams.MODEL.KEEP_CAMID_CENTROIDS,
                )
            if self.trainer.global_rank == 0 and self.trainer.local_rank == 0:
                self.get_val_metrics(embeddings, labels, camids)
            del embeddings, labels, camids
        self.trainer.accelerator_backend.barrier()

    @rank_zero_only
    def eval_on_train(self):
        if self.trainer.global_rank == 0 and self.trainer.local_rank == 0:
            outputs = []
            device = list(self.backbone.parameters())[0].device
            for batch_idx, batch in enumerate(self.test_dataloader):
                x, class_labels, camid, idx = batch
                with torch.no_grad():
                    emb = self.backbone(x.to(device))
                outputs.append(
                    {"emb": emb, "labels": class_labels, "camid": camid, "idx": idx}
                )

            embeddings = torch.cat([x["emb"] for x in outputs]).detach()
            labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
            camids = torch.cat([x["camid"] for x in outputs]).cpu().detach().numpy()
            inds = torch.cat([x["idx"] for x in outputs]).cpu().detach().numpy()

            embeddings, labels, camids = self.validation_create_centroids(
                embeddings, labels, camids
            )

            self.r1_map_func = R1_mAP(self.hparams.num_query)
            cmc, mAP, all_topk = self.r1_map_func.compute(
                feats=embeddings, pids=labels, camids=camids
            )

            topks = {}
            for top_k, kk in zip(all_topk, [1, 5, 10, 20, 50]):
                print("Train top-k, Rank-{:<3}:{:.1%}".format(kk, top_k))
                topks[f"Train Top-{kk}"] = top_k
            print(f"Train mAP: {mAP}")

            log_data = {"Train mAP": mAP}
            log_data = {**log_data, **topks}
            for key, val in log_data.items():
                tensorboard = self.logger.experiment
                tensorboard.add_scalar(key, val, self.current_epoch)

    @staticmethod
    def create_masks_train(class_labels):
        labels_dict = defaultdict(list)
        class_labels = class_labels.detach().cpu().numpy()
        for idx, pid in enumerate(class_labels):
            labels_dict[pid].append(idx)
        labels_list = [v for k, v in labels_dict.items()]
        labels_list_copy = copy.deepcopy(labels_list)
        lens_list = [len(item) for item in labels_list]
        lens_list_cs = np.cumsum(lens_list)

        max_gal_num = max(
            [len(item) for item in labels_dict.values()]
        )  ## TODO Should allow usage of all permuations

        masks = torch.ones((max_gal_num, len(class_labels)), dtype=bool)
        for _ in range(max_gal_num):
            for i, inner_list in enumerate(labels_list):
                if len(inner_list) > 0:
                    masks[_, inner_list.pop(0)] = 0
                else:
                    start_ind = lens_list_cs[i - 1]
                    end_ind = start_ind + lens_list[i]
                    masks[_, start_ind:end_ind] = 0

        return masks, labels_list_copy

    @rank_zero_only
    def test_step(self, batch, batch_idx):
        ret = self.validation_step(batch, batch_idx)
        return ret

    @rank_zero_only
    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)
