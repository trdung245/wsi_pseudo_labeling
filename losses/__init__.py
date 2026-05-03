from .multitask_loss import UncertaintyMultiTaskLoss, CBFocalMultiTaskLoss
from .distillation_loss import DistillationLoss

__all__ = ["UncertaintyMultiTaskLoss", "CBFocalMultiTaskLoss", "DistillationLoss"]
