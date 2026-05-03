"""
Phase 2: Stacking Ensemble Trainer

Loads the three trained VMoE models (frozen), then trains only the MetaLearner
on a combined validation / held-out set.

Strategy
--------
  - All three VMoE models are frozen (no gradient flow into them).
  - The MetaLearner is a small MLP; it trains quickly on top-level predictions.
  - Loss: standard cross-entropy + Huber regression (no uncertainty weighting
    needed here because the MetaLearner has fixed-scale inputs).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import StackingEnsemble
from .base_trainer import BaseTrainer


class _EnsembleValCriterion(nn.Module):
    """Simple CE + Huber criterion for EnsembleTrainer validation."""

    def forward(self, logits, attn_score, labels, attentions, **_):
        cls_loss = F.cross_entropy(logits, labels)
        reg_loss = F.huber_loss(attn_score.squeeze(-1), attentions, delta=0.1)
        total = cls_loss + 0.5 * reg_loss
        return {
            "total":    total,
            "cls_loss": cls_loss.detach(),
            "reg_loss": reg_loss.detach(),
        }

logger = logging.getLogger(__name__)


class EnsembleTrainer(BaseTrainer):
    """Trainer for Phase 2: meta-learner on top of frozen VMoE models."""

    def __init__(
        self,
        model: StackingEnsemble,
        train_loader,
        val_loader,
        cfg: dict,
    ) -> None:
        # Only meta-learner parameters are trainable; base models are frozen
        super().__init__(model, train_loader, val_loader, cfg, phase_name="ensemble")
        self.criterion = _EnsembleValCriterion().to(self.device)

    # ------------------------------------------------------------------

    def _train_step(self, batch):
        images    = batch["image"]
        labels    = batch["label"]
        attentions = batch["attention"]

        out = self.model(images)

        cls_loss  = F.cross_entropy(out["logits"], labels)
        attn_pred = out["attn_score"].squeeze(-1)
        reg_loss  = F.huber_loss(attn_pred, attentions, delta=0.1)

        total = cls_loss + 0.5 * reg_loss

        loss_dict = {
            "total":    total,
            "cls_loss": cls_loss.detach(),
            "reg_loss": reg_loss.detach(),
        }
        return loss_dict, out["logits"], out["attn_score"], labels, attentions

    def _eval_step(self, batch):
        images    = batch["image"]
        labels    = batch["label"]
        attentions = batch["attention"]

        out = self.model(images)
        return out["logits"], out["attn_score"], labels, attentions
