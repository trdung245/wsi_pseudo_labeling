"""
Stacking Ensemble (Level-1 meta-learner).

Architecture
------------
Five VMoE models (trained separately on balanced, neg_heavy, benign_heavy,
atypical_heavy, malignant_heavy CSVs) are frozen.  Their predictions are
concatenated and fed into a small MLP meta-learner:

  VMoE_balanced        ──► (softmax probs B×4 + attn B×1) ─┐
  VMoE_neg_heavy       ──► (softmax probs B×4 + attn B×1) ──┤
  VMoE_benign_heavy    ──► (softmax probs B×4 + attn B×1) ──┼──► MetaLearner ──► (logits, attn_score)
  VMoE_atypical_heavy  ──► (softmax probs B×4 + attn B×1) ──┤
  VMoE_malignant_heavy ──► (softmax probs B×4 + attn B×1) ─┘

Input to MetaLearner: (B, 25)  [5 models × (4 classes + 1 attn) = 25]

The MetaLearner itself has two output heads:
  - Classification head: (B, num_classes)
  - Attention regression head: (B, 1)

Both share the same trunk.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vmoe import VMoE


class MetaLearner(nn.Module):
    """Small MLP that combines predictions from multiple VMoE models.

    Parameters
    ----------
    num_models:     Number of base VMoE models (default 3).
    num_classes:    Number of output classes.
    hidden_dims:    Hidden layer sizes for the trunk MLP.
    dropout:        Dropout rate.
    """

    def __init__(
        self,
        num_models: int = 3,
        num_classes: int = 4,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # input: (num_classes + 1) * num_models
        in_dim = (num_classes + 1) * num_models

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)

        self.cls_head  = nn.Linear(prev, num_classes)
        self.attn_head = nn.Sequential(nn.Linear(prev, 1), nn.Sigmoid())

    def forward(self, base_predictions: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        base_predictions:
            List of dicts, each with keys "logits" (B, C) and "attn_score" (B, 1).

        Returns
        -------
        dict with keys: logits (B, C), attn_score (B, 1)
        """
        parts = []
        for pred in base_predictions:
            probs = F.softmax(pred["logits"].detach(), dim=-1)   # (B, C)
            attn  = pred["attn_score"].detach()                   # (B, 1)
            parts.append(torch.cat([probs, attn], dim=-1))       # (B, C+1)

        x = torch.cat(parts, dim=-1)   # (B, (C+1)*num_models)
        h = self.trunk(x)
        return {
            "logits":     self.cls_head(h),
            "attn_score": self.attn_head(h),
        }


class StackingEnsemble(nn.Module):
    """Full stacking ensemble: frozen VMoE base models + trainable MetaLearner.

    During Phase 2 training, only the MetaLearner parameters are updated.
    The VMoE models are loaded with their Phase 1 checkpoints.

    Parameters
    ----------
    vmoe_configs:
        List of VMoE config dicts (one per dataset), each with the keys
        required by VMoE.__init__.  The order must match the checkpoints.
    checkpoints:
        List of checkpoint paths to load into each VMoE model.
    meta_hidden_dims / meta_dropout:
        MetaLearner architecture parameters.
    num_classes:
        Number of output classes.
    """

    def __init__(
        self,
        vmoe_configs: list[dict],
        checkpoints: Optional[list[str]] = None,
        meta_hidden_dims: Optional[list[int]] = None,
        meta_dropout: float = 0.3,
        num_classes: int = 4,
    ) -> None:
        super().__init__()

        self.base_models = nn.ModuleList([VMoE(**cfg) for cfg in vmoe_configs])

        if checkpoints is not None:
            for model, ckpt_path in zip(self.base_models, checkpoints):
                if ckpt_path is None:
                    continue   # skip slot — base model keeps random init
                state = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(state["model"])

        # Freeze all base model parameters
        for param in self.base_models.parameters():
            param.requires_grad_(False)

        self.meta = MetaLearner(
            num_models=len(self.base_models),
            num_classes=num_classes,
            hidden_dims=meta_hidden_dims or [128, 64],
            dropout=meta_dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run all base models (frozen) then meta-learner."""
        with torch.no_grad():
            base_preds = [model(x) for model in self.base_models]

        return self.meta(base_preds)

    def unfreeze_base_models(self) -> None:
        """Optional: unfreeze base models for end-to-end fine-tuning."""
        for param in self.base_models.parameters():
            param.requires_grad_(True)
