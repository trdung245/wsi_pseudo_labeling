"""
Averaging Ensemble — ablation baseline for the stacking ensemble.

Instead of a learned MetaLearner, predictions from all five frozen VMoE
base models are combined by simple averaging:

  logits_avg     = mean(logits_i,     i=0..N-1)
  attn_score_avg = mean(attn_score_i, i=0..N-1)

The output dict is identical to StackingEnsemble so compute_metrics.py
can consume it without modification.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .vmoe import VMoE


class AveragingEnsemble(nn.Module):
    """Five frozen VMoE models combined by simple logit averaging.

    Parameters
    ----------
    vmoe_configs:
        List of VMoE config dicts (one per dataset).  Order must match
        checkpoints.
    checkpoints:
        List of checkpoint paths to load into each VMoE model.
    """

    def __init__(
        self,
        vmoe_configs: list[dict],
        checkpoints: Optional[list[str]] = None,
    ) -> None:
        super().__init__()

        self.base_models = nn.ModuleList([VMoE(**cfg) for cfg in vmoe_configs])

        if checkpoints is not None:
            for model, ckpt_path in zip(self.base_models, checkpoints):
                if ckpt_path is None:
                    continue
                state = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(state["model"])

        for param in self.base_models.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = [model(x) for model in self.base_models]

        avg_logits = torch.stack([o["logits"] for o in outputs], dim=0).mean(dim=0)
        avg_attn   = torch.stack([o["attn_score"] for o in outputs], dim=0).mean(dim=0)

        return {
            "logits":     avg_logits,
            "attn_score": avg_attn,
        }
