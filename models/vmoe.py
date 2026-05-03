"""
Vision Mixture of Experts (VMoE) model.

Architecture
------------
                       ┌─ Expert 0 ─┐
                       ├─ Expert 1 ─┤
  Input Image ──────►  │  ...       │ ──► Weighted sum ──► cls head  ──► logits
          │             ├─ Expert K ─┘                  └► attn head ──► score
          └──► Gate ──► routing weights (sparse top-k)

Each expert is an independent ExpertBackbone (pretrained CNN).
The gate is a lightweight MLP that receives a low-resolution image embedding
from a shared frozen ViT-tiny (or a simple average pool of the image) and
outputs per-expert routing logits.

Sparse routing
--------------
Only the top-k experts are activated per sample:
  1. Gate computes raw logits g ∈ ℝ^N.
  2. Top-k logits are kept; the rest are set to -∞.
  3. Softmax yields routing weights w ∈ [0,1]^N with at most k non-zeros.
  4. Output = sum_i w_i * expert_i(x).

Load-balancing loss (auxiliary)
--------------------------------
Prevents expert collapse: penalizes uneven expert utilisation.
  L_lb = N * sum_i (f_i * p_i)
  f_i = fraction of tokens routed to expert i
  p_i = mean gate probability for expert i  (differentiable)
Coefficient controlled by `load_balance_coeff` in config.

Outputs
-------
A dict with keys:
  "logits"       – (B, num_classes)  raw classification logits
  "attn_score"   – (B, 1)            predicted attention score
  "load_bal_loss"– scalar            auxiliary load-balancing loss
  "routing_weights" – (B, N)         full (possibly sparse) routing weights
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .experts import ExpertBackbone


class RouterGate(nn.Module):
    """Lightweight gating network.

    Takes the global-average-pooled image as input (no extra forward pass
    needed; the gate shares the same image tensor as the experts).

    Parameters
    ----------
    in_dim:
        Dimension of the gate input feature (e.g. 3 * H * W after avg pool).
    num_experts:
        Total number of experts N.
    hidden:
        Hidden units in the 2-layer MLP gate.
    """

    def __init__(self, in_dim: int, num_experts: int, hidden: int = 128) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),        # (B, C, 4, 4)
            nn.Flatten(),                   # (B, C*16)
            nn.Linear(in_dim * 16, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_experts),
        )
        self._in_channels = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return routing logits of shape (B, num_experts)."""
        return self.gate(x)


class VMoE(nn.Module):
    """Vision Mixture of Experts.

    Parameters
    ----------
    backbone:        Full timm pretrained model ID, e.g. "efficientnet_b2.ra_in1k".
    pretrained:      Whether to load ImageNet pretrained weights.
    num_experts:     N – total number of expert networks.
    top_k:           k – number of experts activated per sample (k ≤ N).
    num_classes:     Number of output classes.
    load_balance_coeff: Weight of the auxiliary load-balancing loss.
    expert_dropout:  Dropout applied inside each ExpertBackbone.
    freeze_blocks:   Leading backbone blocks to freeze (0 = train everything).
    cls_head_hidden: Hidden dim of the classification head.
    attn_head_hidden:Hidden dim of the attention regression head.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b2.ra_in1k",
        pretrained: bool = True,
        num_experts: int = 4,
        top_k: int = 2,
        num_classes: int = 4,
        load_balance_coeff: float = 0.01,
        expert_dropout: float = 0.1,
        freeze_blocks: int = 5,
        cls_head_hidden: int = 256,
        attn_head_hidden: int = 256,
    ) -> None:
        super().__init__()
        assert 1 <= top_k <= num_experts, "top_k must be in [1, num_experts]"

        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_coeff = load_balance_coeff
        self.num_classes = num_classes

        # ── Expert networks ──────────────────────────────────────────────
        self.experts = nn.ModuleList([
            ExpertBackbone(
                backbone, pretrained=pretrained,
                dropout=expert_dropout, freeze_blocks=freeze_blocks,
            )
            for _ in range(num_experts)
        ])
        self.feature_dim: int = self.experts[0].feature_dim

        # ── Router gate ──────────────────────────────────────────────────
        # Gate inputs: raw image pixels passed through AdaptiveAvgPool2d(4)
        # in_channels = 3 (RGB)
        self.router = RouterGate(in_dim=3, num_experts=num_experts)

        # ── Shared task heads ────────────────────────────────────────────
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
            nn.Sigmoid(),  # attention scores ∈ [0, 1]
        )

    # ------------------------------------------------------------------
    # Sparse top-k routing
    # ------------------------------------------------------------------

    def _sparse_route(self, gate_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sparse routing weights and full softmax probabilities.

        Parameters
        ----------
        gate_logits: (B, N)

        Returns
        -------
        routing_weights: (B, N) – sparse; zeros for experts outside top-k
        gate_probs:      (B, N) – full softmax (used for load-balancing loss)
        """
        gate_probs = F.softmax(gate_logits, dim=-1)          # (B, N)

        # Keep only top-k; mask the rest with -inf before softmax
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)  # (B, k)
        sparse_logits = torch.full_like(gate_logits, float("-inf"))
        sparse_logits.scatter_(1, topk_idx, topk_vals)
        routing_weights = F.softmax(sparse_logits, dim=-1)   # (B, N), zeros outside top-k

        return routing_weights, gate_probs

    # ------------------------------------------------------------------
    # Load-balancing auxiliary loss
    # ------------------------------------------------------------------

    @staticmethod
    def _load_balance_loss(
        routing_weights: torch.Tensor,  # (B, N)
        gate_probs: torch.Tensor,        # (B, N)
    ) -> torch.Tensor:
        """Switch Transformer auxiliary load-balancing loss.

        L_lb = N * Σ_i (f_i * p_i)
          f_i = mean fraction of samples assigned to expert i  (non-differentiable)
          p_i = mean gate probability for expert i             (differentiable)
        """
        n = gate_probs.shape[-1]
        # f_i: fraction of samples where expert i is in top-k (mask > 0)
        dispatched = (routing_weights > 0).float()   # (B, N)
        f = dispatched.mean(dim=0)                   # (N,)
        p = gate_probs.mean(dim=0)                   # (N,)
        return n * (f * p).sum()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x: (B, 3, H, W)

        Returns
        -------
        dict with keys: logits, attn_score, load_bal_loss, routing_weights
        """
        B = x.size(0)

        # 1. Route
        gate_logits = self.router(x)                          # (B, N)
        routing_weights, gate_probs = self._sparse_route(gate_logits)

        # 2. Run only the experts that have at least one assigned sample.
        #    In practice with small N (4) we just run all and mask.
        expert_features = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # (B, N, feature_dim)

        # 3. Weighted combination of expert outputs
        w = routing_weights.unsqueeze(-1)                     # (B, N, 1)
        combined = (expert_features * w).sum(dim=1)           # (B, feature_dim)

        # 4. Task heads
        logits     = self.cls_head(combined)                  # (B, num_classes)
        attn_score = self.attn_head(combined)                 # (B, 1)

        # 5. Auxiliary loss
        lb_loss = self._load_balance_loss(routing_weights, gate_probs)

        return {
            "logits":           logits,
            "attn_score":       attn_score,
            "features":         combined,       # (B, feature_dim) — used by SupCon auxiliary loss
            "load_bal_loss":    lb_loss,
            "routing_weights":  routing_weights,
        }

    def param_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        """Return optimizer parameter groups for differential learning rate.

        Frozen backbone params are excluded automatically (requires_grad=False).

        Groups:
          1. Unfrozen backbone params  → backbone_lr  (lower, pretrained)
          2. Router gate params        → head_lr      (new, train faster)
          3. cls_head + attn_head      → head_lr      (new, train faster)
        """
        backbone_params, head_params = [], []
        for expert in self.experts:
            backbone_params.extend(expert.backbone_params())
        head_params.extend(self.router.parameters())
        head_params.extend(self.cls_head.parameters())
        head_params.extend(self.attn_head.parameters())
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params,     "lr": head_lr},
        ]

    def unfreeze_all(self) -> None:
        """Unfreeze every expert backbone for full fine-tuning."""
        for expert in self.experts:
            expert.unfreeze_all()

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the combined expert feature vector (B, feature_dim).

        Used by the meta-learner to extract intermediate representations.
        """
        gate_logits = self.router(x)
        routing_weights, _ = self._sparse_route(gate_logits)
        expert_features = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )
        w = routing_weights.unsqueeze(-1)
        return (expert_features * w).sum(dim=1)
