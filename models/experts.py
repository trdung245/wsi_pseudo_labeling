"""
Individual expert networks used inside Vision Mixture of Experts.

Each expert wraps a timm backbone and exposes:
  - feature_dim: output feature dimensionality
  - backbone_params() / head_params(): parameter groups for differential LR
  - forward(x) -> feature tensor  (no classification head)

Layer freezing strategy
-----------------------
When `freeze_blocks` > 0, the first N EfficientNet blocks are frozen
(weights fixed, no gradients).  Only the last (total - freeze_blocks) blocks
and the final BN/pooling layers are trained.

Typical schedule for fine-tuning on a medical domain:
  freeze_blocks=5  → train only last 2 blocks + head (~30% of backbone params)
  freeze_blocks=0  → train entire backbone (default, slower)

The frozen portion still contributes to the forward pass — pretrained low-level
features (edges, textures) are reused as-is while high-level semantic features
adapt to histopathology.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class ExpertBackbone(nn.Module):
    """Single expert: a pretrained timm backbone without its classification head.

    Parameters
    ----------
    model_name:
        Full timm pretrained model ID, e.g. "efficientnet_b2.ra_in1k".
    pretrained:
        Load ImageNet pretrained weights.
    dropout:
        Dropout applied to the backbone output features.
    freeze_blocks:
        Number of leading backbone blocks to freeze.  0 = train everything.
        For EfficientNet-B2 (7 blocks total), freeze_blocks=5 leaves the last
        2 blocks + normalization trainable.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b2.ra_in1k",
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze_blocks: int = 5,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.feature_dim: int = self.backbone.num_features
        self.dropout = nn.Dropout(p=dropout)

        if pretrained and freeze_blocks > 0:
            self._freeze_early_blocks(freeze_blocks)

    # ------------------------------------------------------------------

    def _freeze_early_blocks(self, n: int) -> None:
        """Freeze the stem + first n blocks of the backbone."""
        # timm EfficientNet exposes blocks as backbone.blocks (a ModuleList).
        # Freeze the stem unconditionally, then blocks[0..n-1].
        frozen_modules: list[nn.Module] = []

        if hasattr(self.backbone, "conv_stem"):
            frozen_modules.append(self.backbone.conv_stem)
        if hasattr(self.backbone, "bn1"):
            frozen_modules.append(self.backbone.bn1)
        if hasattr(self.backbone, "blocks"):
            for i, block in enumerate(self.backbone.blocks):
                if i < n:
                    frozen_modules.append(block)

        for mod in frozen_modules:
            for param in mod.parameters():
                param.requires_grad_(False)

    def unfreeze_all(self) -> None:
        """Unfreeze the entire backbone (call before fine-tuning phase)."""
        for param in self.backbone.parameters():
            param.requires_grad_(True)

    # ------------------------------------------------------------------
    # Parameter groups for differential learning rate
    # ------------------------------------------------------------------

    def backbone_params(self) -> list[nn.Parameter]:
        """Trainable backbone parameters (unfrozen blocks + BN)."""
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def head_params(self) -> list[nn.Parameter]:
        """Dropout has no parameters; exposed for API consistency."""
        return list(self.dropout.parameters())

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled feature vector of shape (B, feature_dim)."""
        return self.dropout(self.backbone(x))
