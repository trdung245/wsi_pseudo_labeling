#!/usr/bin/env python3
"""
Single-image inference script.

Supports three model modes:
  --model vmoe       Run a single VMoE checkpoint
  --model ensemble   Run the full stacking ensemble (5 VMoE + MetaLearner)
  --model student    Run the distilled student PatchClassifier

Usage examples
--------------
  # Single VMoE:
  python predict.py --model vmoe \
      --checkpoint checkpoints/vmoe/epoch_008_f1_0.6789.pth \
      --image organized_patches/atypical/BRACS_1775_99152_20608.png

  # Full stacking ensemble (needs all 5 VMoE checkpoints + ensemble checkpoint):
  python predict.py --model ensemble \
      --vmoe-ckpts \
          checkpoints/vmoe/epoch_008_f1_0.6789.pth \
          checkpoints/vmoe/epoch_008_f1_0.5748.pth \
          checkpoints/vmoe/epoch_003_f1_0.6593.pth \
          checkpoints/vmoe/epoch_004_f1_0.6780.pth \
          checkpoints/vmoe/epoch_008_f1_0.6763.pth \
      --checkpoint checkpoints/ensemble/epoch_007_f1_0.8606.pth \
      --image path/to/patch.png

  # Student model:
  python predict.py --model student \
      --checkpoint checkpoints/distillation/epoch_009_f1_0.7817.pth \
      --image path/to/patch.png

  # Use 3-class config:
  python predict.py --model ensemble ... --config configs/config_3class.yaml

  # Visualise with a bar chart:
  python predict.py --model vmoe --checkpoint ... --image ... --visualize
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

# ── ensure the project root is on the path ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from data.transforms import val_transforms
from data.dataset import LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES
from models import VMoE, StackingEnsemble, build_student


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_label_map(cfg: dict) -> dict[str, int]:
    lbl_cfg = cfg.get("labels", {})
    return lbl_cfg.get("label_to_idx", LABEL_TO_IDX)


def get_num_classes(cfg: dict) -> int:
    return cfg.get("labels", {}).get("num_classes", NUM_CLASSES)


def get_idx_to_label(cfg: dict) -> dict[int, str]:
    label_map = get_label_map(cfg)
    return {v: k for k, v in label_map.items()}


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def load_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """Load and preprocess a single image → (1, 3, H, W) tensor."""
    transform = val_transforms(image_size)
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)   # (1, 3, H, W)


def load_vmoe_checkpoint(model: VMoE, ckpt_path: str, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------

def run_vmoe(args, cfg: dict) -> dict:
    device    = get_device(cfg.get("device", "auto"))
    n_classes = get_num_classes(cfg)
    vmoe_cfg  = cfg["model"]["vmoe"]

    model = VMoE(
        backbone=vmoe_cfg["backbone"],
        pretrained=False,
        num_experts=vmoe_cfg["num_experts"],
        top_k=vmoe_cfg["top_k"],
        num_classes=n_classes,
        load_balance_coeff=vmoe_cfg["load_balance_coeff"],
        expert_dropout=0.0,           # no dropout at inference
        freeze_blocks=0,
        cls_head_hidden=vmoe_cfg["cls_head_hidden"],
        attn_head_hidden=vmoe_cfg["attn_head_hidden"],
    ).to(device)

    load_vmoe_checkpoint(model, args.checkpoint, device)
    model.eval()

    image = load_image(args.image, cfg["data"]["image_size"]).to(device)

    with torch.no_grad():
        out = model(image)

    probs  = F.softmax(out["logits"], dim=-1).squeeze(0).cpu()
    attn   = out["attn_score"].squeeze().cpu().item()
    rw     = out["routing_weights"].squeeze(0).cpu().tolist()

    return {
        "probs":           probs.tolist(),
        "attn_score":      attn,
        "routing_weights": rw,
        "model_type":      "VMoE",
        "checkpoint":      args.checkpoint,
    }


def run_ensemble(args, cfg: dict) -> dict:
    device    = get_device(cfg.get("device", "auto"))
    n_classes = get_num_classes(cfg)
    vmoe_cfg  = cfg["model"]["vmoe"]
    ens_cfg   = cfg["model"]["ensemble"]

    dataset_names = list(cfg["data"]["csv_files"].keys())
    if len(args.vmoe_ckpts) != len(dataset_names):
        print(
            f"[warn] Config has {len(dataset_names)} datasets but "
            f"{len(args.vmoe_ckpts)} --vmoe-ckpts provided. "
            f"Continuing with {len(args.vmoe_ckpts)} base models."
        )

    vmoe_model_cfg = dict(
        backbone=vmoe_cfg["backbone"],
        pretrained=False,
        num_experts=vmoe_cfg["num_experts"],
        top_k=vmoe_cfg["top_k"],
        num_classes=n_classes,
        load_balance_coeff=vmoe_cfg["load_balance_coeff"],
        expert_dropout=0.0,
        freeze_blocks=0,
        cls_head_hidden=vmoe_cfg["cls_head_hidden"],
        attn_head_hidden=vmoe_cfg["attn_head_hidden"],
    )

    n_models = len(args.vmoe_ckpts)
    ensemble = StackingEnsemble(
        vmoe_configs=[vmoe_model_cfg] * n_models,
        checkpoints=args.vmoe_ckpts,
        meta_hidden_dims=ens_cfg["hidden_dims"],
        meta_dropout=0.0,
        num_classes=n_classes,
    ).to(device)

    # Load MetaLearner weights
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ensemble.load_state_dict(state["model"])
    ensemble.eval()

    image = load_image(args.image, cfg["data"]["image_size"]).to(device)

    # Collect per-model outputs alongside the ensemble output
    per_model = []
    with torch.no_grad():
        for base_model in ensemble.base_models:
            bm_out = base_model(image)
            per_model.append({
                "probs":      F.softmax(bm_out["logits"], dim=-1).squeeze(0).cpu().tolist(),
                "attn_score": bm_out["attn_score"].squeeze().cpu().item(),
                "routing_weights": bm_out["routing_weights"].squeeze(0).cpu().tolist(),
            })

        ens_out = ensemble(image)

    probs = F.softmax(ens_out["logits"], dim=-1).squeeze(0).cpu()
    attn  = ens_out["attn_score"].squeeze().cpu().item()

    return {
        "probs":       probs.tolist(),
        "attn_score":  attn,
        "per_model":   per_model,
        "model_type":  "StackingEnsemble",
        "checkpoint":  args.checkpoint,
    }


def run_student(args, cfg: dict) -> dict:
    device      = get_device(cfg.get("device", "auto"))
    n_classes   = get_num_classes(cfg)
    student_cfg = cfg["model"]["student"]

    model = build_student(student_cfg, num_classes=n_classes).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    image = load_image(args.image, cfg["data"]["image_size"]).to(device)

    with torch.no_grad():
        out = model(image)

    probs = F.softmax(out["logits"], dim=-1).squeeze(0).cpu()
    attn  = out["attn_score"].squeeze().cpu().item()

    return {
        "probs":      probs.tolist(),
        "attn_score": attn,
        "model_type": "Student (PatchClassifier)",
        "checkpoint": args.checkpoint,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_result(result: dict, idx_to_label: dict[int, str]) -> None:
    probs     = result["probs"]
    pred_idx  = int(torch.tensor(probs).argmax())
    pred_label = idx_to_label[pred_idx]
    attn       = result["attn_score"]

    bar_width = 30
    print()
    print(f"  Model       : {result['model_type']}")
    print(f"  Checkpoint  : {result['checkpoint']}")
    print(f"  Image       : {result.get('image', '')}")
    print()
    print("  ─── Class Probabilities ──────────────────────────────")
    for i, p in enumerate(probs):
        label  = idx_to_label[i]
        filled = int(p * bar_width)
        bar    = "█" * filled + "░" * (bar_width - filled)
        marker = " ◄ PREDICTED" if i == pred_idx else ""
        print(f"  {label:<14} [{bar}]  {p:.4f}{marker}")
    print()
    print(f"  Prediction  : {pred_label}  (confidence {probs[pred_idx]:.1%})")
    print(f"  Attn score  : {attn:.4f}")

    if "routing_weights" in result:
        rw = result["routing_weights"]
        print(f"  Routing     : " + "  ".join(f"E{i}={w:.3f}" for i, w in enumerate(rw)))

    if "per_model" in result:
        print()
        print("  ─── Per-VMoE Breakdown ───────────────────────────────")
        for i, pm in enumerate(result["per_model"]):
            pm_pred = idx_to_label[int(torch.tensor(pm["probs"]).argmax())]
            pm_conf = max(pm["probs"])
            print(f"  VMoE {i+1:<2}  →  {pm_pred:<14} ({pm_conf:.1%})  attn={pm['attn_score']:.3f}")
        print()
        print(f"  MetaLearner →  {pred_label:<14} ({probs[pred_idx]:.1%})  attn={attn:.3f}")

    print()


def visualize_result(result: dict, idx_to_label: dict[int, str], image_path: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from PIL import Image as PILImage

    probs      = result["probs"]
    n_classes  = len(probs)
    pred_idx   = int(np.argmax(probs))
    labels     = [idx_to_label[i] for i in range(n_classes)]
    colors     = ["#5DADE2", "#58D68D", "#F39C12", "#E74C3C"][:n_classes]
    has_models = "per_model" in result

    ncols = 3 if has_models else 2
    fig   = plt.figure(figsize=(5 * ncols, 5))
    fig.patch.set_facecolor("#F8F9FA")

    # ── Patch image ──────────────────────────────────────────────────────────
    ax_img = fig.add_subplot(1, ncols, 1)
    ax_img.imshow(PILImage.open(image_path).convert("RGB"))
    ax_img.axis("off")
    ax_img.set_title("Input Patch", fontsize=11, fontweight="bold")

    # ── Final prediction bar chart ────────────────────────────────────────────
    ax_bar = fig.add_subplot(1, ncols, 2)
    x = np.arange(n_classes)
    bars = ax_bar.bar(x, probs, color=colors, edgecolor="white", zorder=3)
    bars[pred_idx].set_edgecolor("#1A1A1A")
    bars[pred_idx].set_linewidth(2.2)
    for bar, p in zip(bars, probs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, p + 0.012,
                    f"{p:.3f}", ha="center", va="bottom", fontsize=9)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax_bar.set_ylabel("Probability")
    ax_bar.set_title(
        f"{result['model_type']}\n→ {labels[pred_idx]}  ({probs[pred_idx]:.1%})\nattn={result['attn_score']:.3f}",
        fontsize=10, fontweight="bold"
    )
    ax_bar.set_facecolor("#FFFFFF")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # ── Per-model breakdown (ensemble only) ──────────────────────────────────
    if has_models:
        ax_pm = fig.add_subplot(1, ncols, 3)
        per_model = result["per_model"]
        n_models  = len(per_model)
        model_labels = [f"VMoE {i+1}" for i in range(n_models)] + ["MetaLearner"]
        all_probs    = [pm["probs"] for pm in per_model] + [probs]

        bar_width = 0.6 / n_classes
        for cls_i in range(n_classes):
            offsets = np.arange(len(all_probs)) + cls_i * bar_width - (n_classes - 1) * bar_width / 2
            vals    = [p[cls_i] for p in all_probs]
            ax_pm.bar(offsets, vals, width=bar_width, color=colors[cls_i],
                      alpha=0.85, label=labels[cls_i], zorder=3)

        ax_pm.set_xticks(np.arange(len(all_probs)))
        ax_pm.set_xticklabels(model_labels, fontsize=8, rotation=20, ha="right")
        ax_pm.set_ylim(0, 1.1)
        ax_pm.set_ylabel("Probability")
        ax_pm.set_title("Per-Model Breakdown", fontsize=10, fontweight="bold")
        ax_pm.set_facecolor("#FFFFFF")
        ax_pm.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax_pm.spines["top"].set_visible(False)
        ax_pm.spines["right"].set_visible(False)
        ax_pm.legend(fontsize=8, loc="upper right", framealpha=0.7)

        # Separator line before MetaLearner column
        sep_x = len(per_model) - 0.5
        ax_pm.axvline(sep_x, color="#333333", linewidth=1.2, linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_path = Path(image_path).stem + "_prediction.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  [saved] {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on a single patch image.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=["vmoe", "ensemble", "student"],
        default="ensemble",
        help="Model type to use for inference.",
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the input patch image (.png / .jpg).",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help=(
            "Checkpoint path.\n"
            "  vmoe     → path to a single VMoE .pth\n"
            "  ensemble → path to the MetaLearner (ensemble) .pth\n"
            "  student  → path to a distilled student .pth"
        ),
    )
    parser.add_argument(
        "--vmoe-ckpts", nargs="+", default=None, metavar="PATH",
        help="(ensemble only) 5 VMoE checkpoint paths, one per dataset in config order.",
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to the YAML config file (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device override: cpu | cuda | mps (default: auto-detect).",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show and save a bar-chart visualisation of the prediction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.image).exists():
        sys.exit(f"[error] Image not found: {args.image}")

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device

    idx_to_label = get_idx_to_label(cfg)

    # ── Run inference ─────────────────────────────────────────────────────────
    if args.model == "vmoe":
        if not args.checkpoint:
            sys.exit("[error] --checkpoint is required for --model vmoe")
        result = run_vmoe(args, cfg)

    elif args.model == "ensemble":
        if not args.checkpoint:
            sys.exit("[error] --checkpoint (ensemble .pth) is required for --model ensemble")
        if not args.vmoe_ckpts:
            sys.exit("[error] --vmoe-ckpts (5 paths) is required for --model ensemble")
        result = run_ensemble(args, cfg)

    elif args.model == "student":
        if not args.checkpoint:
            sys.exit("[error] --checkpoint is required for --model student")
        result = run_student(args, cfg)

    result["image"] = args.image

    # ── Print ─────────────────────────────────────────────────────────────────
    print_result(result, idx_to_label)

    # ── Visualise ─────────────────────────────────────────────────────────────
    if args.visualize:
        visualize_result(result, idx_to_label, args.image)


if __name__ == "__main__":
    main()
