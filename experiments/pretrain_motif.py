import sys
import os
import random
import numpy as np
import time
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

from motifiesta.transforms import RewireTransform
from motifiesta.losses import freq_loss
from motifiesta.losses import rec_loss
from motifiesta.losses import margin_loss
from motifiesta.losses import theta_loss
from motifiesta.utils import assign_index 
from motifiesta.models import MotiFiestaModel
from motifiesta.datasets import MotiFiestaData 


import pytorch_lightning as pl
import logging

log = logging.getLogger(__name__)


class MotifTrainer(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model 
        self.cfg = cfg

    def training_step(self, batch, batch_idx):
        # print("Getting loss")
        loss = self._shared_eval_step(batch, batch_idx, self.current_epoch)

        return loss

    def switch_grads(self, mode='rec'):
        for name, param in self.model.named_parameters():
            if mode == 'rec':
                if 'scorer' in name or 'embed' in name:
                    param.requires_grad = False
                if 'encoder' in name:
                    param.requires_grad = True
            if mode == 'freq':
                if 'scorer' in name or 'embed' in name:
                    param.requires_grad = True
                if 'encoder' in name:
                    param.requires_grad = False

    def _shared_eval_step(self, batch, batch_idx, epoch):
        """

        Args:
            batch:
            batch_idx:

        Returns:

        """

        print(batch_idx)
        print("Rewire")
        rewire_transform = RewireTransform(n_iter=self.cfg.training.rewire_iters)

        start = time.time()
        batch_neg = rewire_transform(batch)

        #print("forward")
        start = time.time()

        out_pos = self.model(batch)
            
        out_neg = self.model(batch_neg)

        #print("loss")
        start = time.time()
        l_r = 0
        loss = 0
        if not epoch % self.cfg.training.rec_epochs:
            # self.switch_grads(mode='rec')
            l_r = rec_loss(steps=self.cfg.model.steps,
                           ee=out_pos['e_ind'],
                           spotlights=out_pos['merge_info']['spotlights'],
                           batch=batch.batch,
                           node_feats=batch.x,
                           edge_index_base=batch.edge_index,
                           edge_feats=batch.edge_attr,
                           internals=out_pos['internals'],
                           num_nodes=self.cfg.training.rec_samples,
                           simfunc=self.cfg.training.simfunc,
                           device=self.device,
                           do_cache=True
                           )
            loss += l_r

        if True: 
            # self.switch_grads(mode='freq')
            start = time.time()
            l_m = freq_loss(out_pos['internals'],
                            out_neg['internals'],
                            out_pos['e_prob'],
                            steps=self.cfg.model.steps,
                            beta=self.cfg.training.beta,
                            lam=self.cfg.training.lam_sigma,
                            estimator=self.cfg.training.estimator
                            )
            freq_time = time.time() - start
            loss += l_m
        # print(f"forward time: {forward_time}, rec_time: {rec_time}, freq time: {freq_time}")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.cfg.training.warmup, self.cfg.training.epochs
        )
        return [optimizer], [lr_scheduler]

@hydra.main(version_base="1.3", config_path="../config", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    # dset = datasets.AlphaFoldDataset(root=cfg.task.path, organism=cfg.task.organism)
    dset = datasets.RCSBDataset(root=cfg.task.path)

    dset = get_pretrain_dataset(cfg, dset)

    data_loader = DataLoader(
        dset, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers
    )

    net = ProteinStructureEncoder(cfg.model)

    model = MotifTrainer(net, cfg)

    logger = pl.loggers.CSVLogger(cfg.paths.output_dir, name='csv_logs')
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.TQDMProgressBar(refresh_rate=1000)
    ]

    limit_train_batches = 5 if cfg.training.debug else None

    trainer = pl.Trainer(
        limit_train_batches=limit_train_batches,
        max_epochs=cfg.training.epochs,
        # devices='auto',
        accelerator='mps',
        enable_checkpointing=False,
        logger=[logger],
        callbacks=callbacks
    )

    trainer.fit(model=model, train_dataloaders=data_loader)

    save_dir = Path(cfg.paths.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    net.save(save_dir / "model.pt")


if __name__ == "__main__":
    main()
