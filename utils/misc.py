# encoding: utf-8
"""
Adapted and extended by:
@author: mikwieczorek
"""

import multiprocessing
import os

import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from lightning_fabric.utilities.seed import seed_everything
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

from callbacks import ModelCheckpointPeriodic
from datasets import init_dataset
from modelling.backbones.resnet_ibn_a import resnet50_ibn_a


def get_distributed_sampler(
    trainer, dataset, train, **kwargs
) -> torch.utils.data.sampler.Sampler:
    world_size = {
        "ddp": trainer.num_nodes * trainer.num_processes,
        "ddp_spawn": trainer.num_nodes * trainer.num_processes,
        "ddp2": trainer.num_nodes,
        "ddp_cpu": trainer.num_processes * trainer.num_nodes,
    }
    assert trainer.distributed_backend is not None
    kwargs = dict(
        num_replicas=world_size[trainer.distributed_backend], rank=trainer.global_rank
    )

    kwargs["shuffle"] = train and not trainer.overfit_batches
    sampler = DistributedSampler(dataset, **kwargs)
    return sampler


def get_backbone(name: str, **kwargs) -> torch.nn.Module:
    """
    Gets just the encoder portion of a torchvision model (replaces final layer with identity)
    :param name: (str) name of the model
    :param kwargs: kwargs to send to the model
    :return:
    """

    if name in torchvision.models.__dict__:
        model_creator = torchvision.models.__dict__.get(name)
    elif name == "resnet50_ibn_a":
        model = resnet50_ibn_a(last_stride=1, **kwargs)
        model_creator = True
    else:
        raise AttributeError(f"Unknown architecture {name}")

    assert model_creator is not None, f"no torchvision model named {name}"
    if name != "resnet50_ibn_a":
        model = model_creator(**kwargs)
    if hasattr(model, "fc"):
        model.fc = torch.nn.Identity()
    elif hasattr(model, "classifier"):
        model.classifier = torch.nn.Identity()
    else:
        raise NotImplementedError(f"Unknown class {model.__class__}")

    return model


def run_single(cfg, method, logger_save_dir):

    logger = TensorBoardLogger(cfg.LOG_DIR, name=logger_save_dir)
    mlflow_logger = MLFlowLogger(experiment_name="default")

    loggers = [logger, mlflow_logger]

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="{epoch}",
        monitor=cfg.SOLVER.MONITOR_METRIC_NAME,
        mode=cfg.SOLVER.MONITOR_METRIC_MODE,
        verbose=True,
    )

    periodic_checkpointer = ModelCheckpointPeriodic(
        dirname=os.path.join(logger.log_dir, "auto_checkpoints"),
        filename_prefix="checkpoint",
        n_saved=1,
        save_interval=1,
    )

    dm = init_dataset(
        cfg.DATASETS.NAMES, cfg=cfg, num_workers=cfg.DATALOADER.NUM_WORKERS
    )
    dm.setup()
    test_dataloader = None

    trainer = pl.Trainer(
        gpus=cfg.GPU_IDS,
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        logger=loggers,
        fast_dev_run=False,
        check_val_every_n_epoch=cfg.SOLVER.EVAL_PERIOD,
        accelerator=cfg.SOLVER.DIST_BACKEND,
        num_sanity_val_steps=0,
        replace_sampler_ddp=False,
        checkpoint_callback=checkpoint_callback,
        precision=16 if cfg.USE_MIXED_PRECISION else 32,
        resume_from_checkpoint=cfg.MODEL.PRETRAIN_PATH
        if cfg.MODEL.RESUME_TRAINING
        else None,
        callbacks=[periodic_checkpointer],
        enable_pl_optimizer=True,
        reload_dataloaders_every_epoch=True,
        automatic_optimization=cfg.SOLVER.USE_AUTOMATIC_OPTIM,
    )

    train_loader = dm.train_dataloader(
        cfg,
        trainer,
        sampler_name=cfg.DATALOADER.SAMPLER,
        drop_last=cfg.DATALOADER.DROP_LAST,
    )
    val_dataloader = dm.val_dataloader()
    if cfg.TEST.ONLY_TEST:
        method = method.load_from_checkpoint(
            cfg.MODEL.PRETRAIN_PATH,
            cfg=cfg,
            num_query=dm.num_query,
            num_classes=dm.num_classes,
            use_multiple_loggers=True if len(loggers) > 1 else False,
        )
        trainer.test(model=method, test_dataloaders=val_dataloader)
        method.hparams.MODEL.USE_CENTROIDS = not method.hparams.MODEL.USE_CENTROIDS
        trainer.test(model=method, test_dataloaders=val_dataloader)
        method.hparams.MODEL.USE_CENTROIDS = not method.hparams.MODEL.USE_CENTROIDS
    else:
        if cfg.MODEL.RESUME_TRAINING:
            method = method.load_from_checkpoint(
                cfg.MODEL.PRETRAIN_PATH,
                num_query=dm.num_query,
                num_classes=dm.num_classes,
                use_multiple_loggers=True if len(loggers) > 1 else False,
            )
        else:
            method = method(
                cfg,
                test_dataloader=test_dataloader,
                num_query=dm.num_query,
                num_classes=dm.num_classes,
                use_multiple_loggers=True if len(loggers) > 1 else False,
            )
        trainer.fit(
            method, train_dataloader=train_loader, val_dataloaders=[val_dataloader]
        )
        method.hparams.MODEL.USE_CENTROIDS = not method.hparams.MODEL.USE_CENTROIDS
        trainer.test(model=method, test_dataloaders=val_dataloader)
        method.hparams.MODEL.USE_CENTROIDS = not method.hparams.MODEL.USE_CENTROIDS


def run_main(cfg, method, logger_save_dir):
    cfg.DATALOADER.NUM_WORKERS = int(multiprocessing.cpu_count() // len(cfg.GPU_IDS))
    cfg.LOG_DIR = (
        f"logs/{cfg.DATASETS.NAMES}" if cfg.OUTPUT_DIR == "" else cfg.OUTPUT_DIR
    )

    if cfg.REPRODUCIBLE:
        for seed in range(
            cfg.REPRODUCIBLE_SEED, cfg.REPRODUCIBLE_SEED + cfg.REPRODUCIBLE_NUM_RUNS
        ):
            cfg.REPRODUCIBLE_SEED = seed
            seed_everything(seed=seed)
            run_single(cfg, method, logger_save_dir)
    else:
        for _ in range(0, cfg.REPRODUCIBLE_NUM_RUNS):
            run_single(cfg, method, logger_save_dir)
