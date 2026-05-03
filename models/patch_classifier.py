"""
PatchClassifier — single-backbone model for WSI patch classification.

Architecture
------------
  Pretrained EfficientNet backbone (timm)
      └─► Global avg pool → dropout → feature vector
              ├─► cls_head  → (B, num_classes)  raw logits
              └─► attn_head → (B, 1)            attention score ∈ [0, 1]

Replaces the VMoE as the per-dataset base model in Phase 1.
The ensemble (Phase 2) already provides the mixture-of-experts effect by
combining five independently-biased classifiers, so there is no need for
intra-model expert routing.

Outputs
-------
Forward returns a dict with keys:
  "logits"      – (B, num_classes)
  "attn_score"  – (B, 1)
  "features"    – (B, feature_dim)  pooled backbone output
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .experts import ExpertBackbone


class PatchClassifier(nn.Module):
    """Single-backbone patch classifier with dual output heads.

    Parameters
    ----------
    backbone:
        Full timm pretrained model ID, e.g. "efficientnet_b2.ra_in1k".
    pretrained:
        Load ImageNet pretrained weights.
    num_classes:
        Number of output classes.
    dropout:
        Dropout applied to backbone features.
    freeze_blocks:
        Leading backbone blocks to freeze (0 = train everything).
    cls_head_hidden:
        Hidden dim of the classification MLP head.
    attn_head_hidden:
        Hidden dim of the attention regression MLP head.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b2.ra_in1k",
        pretrained: bool = True,
        num_classes: int = 4,
        dropout: float = 0.1,
        freeze_blocks: int = 5,
        cls_head_hidden: int = 256,
        attn_head_hidden: int = 256,
    ) -> None:
        super().__init__()

        self.backbone_net = ExpertBackbone(
            model_name=backbone,
            pretrained=pretrained,
            dropout=dropout,
            freeze_blocks=freeze_blocks,
        )
        self.feature_dim: int = self.backbone_net.feature_dim

        self.cls_head = nn.Sequential(
            nn.Linear(self.feature_dim, cls_head_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(cls_head_hidden, num_classes),
        )
        self.attn_head = nn.Sequential(
            nn.Linear(self.feature_dim, attn_head_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(attn_head_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x: (B, 3, H, W)

        Returns
        -------
        dict with keys: logits, attn_score, features
        """
        features   = self.backbone_net(x)          # (B, feature_dim)
        logits     = self.cls_head(features)        # (B, num_classes)
        attn_score = self.attn_head(features)       # (B, 1)
        return {
            "logits":     logits,
            "attn_score": attn_score,
            "features":   features,
        }

    def param_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        """Return optimizer parameter groups for differential learning rate.

        Frozen backbone params are excluded automatically (requires_grad=False).

        Groups:
          1. Unfrozen backbone params → backbone_lr (lower, pretrained)
          2. cls_head + attn_head    → head_lr     (new, train faster)
        """
        head_params = list(self.cls_head.parameters()) + list(self.attn_head.parameters())
        return [
            {"params": self.backbone_net.backbone_params(), "lr": backbone_lr},
            {"params": head_params,                         "lr": head_lr},
        ]

    def unfreeze_all(self) -> None:
        """Unfreeze the full backbone for end-to-end fine-tuning."""
        self.backbone_net.unfreeze_all()
