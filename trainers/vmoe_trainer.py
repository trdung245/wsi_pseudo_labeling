"""
Phase 1: VMoE Trainer

Trains a single VMoE on one of the five dataset CSVs using
Class-Balanced Focal Loss + fixed-weight Huber regression (CBFocalMultiTaskLoss).

One trainer is instantiated per dataset:
  balanced, neg_heavy, benign_heavy, atypical_heavy, malignant_heavy
"""

from __future__ import annotations

from collections import Counter

import torch

from losses import CBFocalMultiTaskLoss
from models import VMoE
from .base_trainer import BaseTrainer


class VMoETrainer(BaseTrainer):
    """Trainer for Phase 1: single VMoE on a single dataset."""

    def __init__(
        self,
        model: VMoE,
        train_loader,
        val_loader,
        cfg: dict,
        class_weights: torch.Tensor | None = None,
        dataset_name: str = "balanced",
    ) -> None:
        super().__init__(model, train_loader, val_loader, cfg, phase_name="vmoe")
        self.phase_name = f"vmoe_{dataset_name}"   # separate logging per dataset

        num_classes = cfg["labels"]["num_classes"]
        loss_cfg    = cfg.get("loss", {})

        dataset = train_loader.dataset
        counts_map = Counter(r["label"] for r in dataset.records)
        samples_per_class = [counts_map.get(i, 1) for i in range(num_classes)]

        self.criterion = CBFocalMultiTaskLoss(
            samples_per_class=samples_per_class,
            num_classes=num_classes,
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            reg_weight=loss_cfg.get("reg_weight", 0.1),
            load_balance_coeff=cfg["model"]["vmoe"]["load_balance_coeff"],
        ).to(self.device)

    # ------------------------------------------------------------------

    def _train_step(self, batch):
        images     = batch["image"]
        labels     = batch["label"]
        attentions = batch["attention"]

        out = self.model(images)

        loss_dict = self.criterion(
            logits=out["logits"],
            attn_score=out["attn_score"],
            labels=labels,
            attentions=attentions,
            load_bal_loss=out["load_bal_loss"],
        )

        return loss_dict, out["logits"], out["attn_score"], labels, attentions

    def _eval_step(self, batch):
        images     = batch["image"]
        labels     = batch["label"]
        attentions = batch["attention"]

        out = self.model(images)
        return out["logits"], out["attn_score"], labels, attentions
