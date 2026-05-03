"""
Student model for knowledge distillation (Phase 3).

A smaller PatchClassifier (EfficientNet-B0 backbone) trained to mimic the
stacking ensemble teacher while also learning from hard ground-truth labels.
"""

from __future__ import annotations

from .patch_classifier import PatchClassifier


def build_student(cfg: dict, num_classes: int = 4) -> PatchClassifier:
    """Construct the student PatchClassifier from a config dict."""
    return PatchClassifier(
        backbone=cfg.get("backbone", "efficientnet_b0.ra4_e3600_r224_in1k"),
        pretrained=cfg.get("pretrained", True),
        num_classes=num_classes,
        dropout=cfg.get("dropout", 0.1),
        freeze_blocks=cfg.get("freeze_blocks", 3),
        cls_head_hidden=cfg.get("cls_head_hidden", 128),
        attn_head_hidden=cfg.get("attn_head_hidden", 128),
    )
