import os
import random
import numpy as np
import itertools
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric import utils
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from proteinshake import datasets

from proteinshake_eval.transforms import get_pretrain_dataset
from proteinshake_eval.models.protein_model import ProteinStructureEncoder
from proteinshake_eval.utils import get_cosine_schedule_with_warmup
from proteinshake_eval.utils import catchtime

from motifiesta.utils import assign_index 

from pretrain_motif import MotifTrainer
from pretrain_mask_residues import MaskTrainer

import pytorch_lightning as pl
import logging


log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../config", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    dset = datasets.AlphaFoldDataset(root=cfg.task.path, organism=cfg.task.organism)
    print("Loading data")
    # dset = datasets.RCSBDataset(root=cfg.task.path)
    print("Converting..")
    dset = get_pretrain_dataset(cfg, dset)

    save_dir = Path(cfg.paths.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_loader = DataLoader(
        dset, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    print("model init")
    net = ProteinStructureEncoder(cfg.model)

    if cfg.training.strategy == 'motif':
        model = MotifTrainer(net, cfg)
    elif cfg.training.strategy == 'mask':
        model = MaskTrainer(net, cfg)


    logger = pl.loggers.CSVLogger(cfg.paths.output_dir, name='csv_logs')
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.TQDMProgressBar(refresh_rate=1),
        pl.callbacks.ModelCheckpoint(dirpath=cfg.paths.output_dir,
                           every_n_train_steps=100)
    ]

    limit_train_batches = 5 if cfg.training.debug else None

    trainer = pl.Trainer(
        limit_train_batches=limit_train_batches,
        max_epochs=cfg.training.epochs,
        devices='auto',
        accelerator='auto',
        enable_checkpointing=True,
        logger=[logger],
        callbacks=callbacks
    )

    with catchtime(log, "trainer.fit()"):
        trainer.fit(model=model, train_dataloaders=data_loader)

    OmegaConf.save(cfg, Path(save_dir, "config.yaml"))
    print(f"Saving model to {save_dir}")
    net.save(save_dir / "model.pt")

if __name__ == "__main__":
    main()
