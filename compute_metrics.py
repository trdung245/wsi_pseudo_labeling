#!/usr/bin/env python3
"""
Full metrics computation script.

Evaluates the stacking ensemble on the balanced test split and produces:
  - Confusion matrix (absolute + row-normalised)
  - Per-class precision, recall, F1, support
  - Macro / weighted averages
  - Per-class AUC-ROC (OvR)
  - Matthews Correlation Coefficient (MCC)
  - Cohen's Kappa
  - Top-2 accuracy
  - Cosine similarity between per-class feature centroids
  - Euclidean distance between per-class feature centroids
  - Attention regression: MAE, RMSE, Spearman ρ, Pearson r

Feature vectors are the meta-learner input (concatenated VMoE softmax + attn,
25-dim) — these are the representations the ensemble actually classifies from.

Usage
-----
  python compute_metrics.py \
    --checkpoint checkpoints/ensemble/epoch_007_f1_0.8606.pth \
    --split test \
    --dataset balanced
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    top_k_accuracy_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from data import PatchDataset, val_transforms, IDX_TO_LABEL, NUM_CLASSES
from models import VMoE, StackingEnsemble, AveragingEnsemble, build_student
from trainers.base_trainer import get_device

CLASS_NAMES = ["non-cancer", "benign", "atypical", "malignant"]


# ---------------------------------------------------------------------------
# Build models
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_ensemble(cfg: dict, checkpoint: str) -> StackingEnsemble:
    vmoe_cfg = cfg["model"]["vmoe"]
    ens_cfg  = cfg["model"]["ensemble"]
    vmoe_base_cfg = dict(
        backbone=vmoe_cfg["backbone"],
        pretrained=False,
        num_experts=vmoe_cfg["num_experts"],
        top_k=vmoe_cfg["top_k"],
        num_classes=NUM_CLASSES,
        load_balance_coeff=vmoe_cfg["load_balance_coeff"],
        expert_dropout=vmoe_cfg["expert_dropout"],
        freeze_blocks=0,
        cls_head_hidden=vmoe_cfg["cls_head_hidden"],
        attn_head_hidden=vmoe_cfg["attn_head_hidden"],
    )
    n_models = len(["balanced", "neg_heavy", "benign_heavy", "atypical_heavy", "malignant_heavy"])
    model = StackingEnsemble(
        vmoe_configs=[vmoe_base_cfg] * n_models,
        checkpoints=None,
        meta_hidden_dims=ens_cfg["hidden_dims"],
        meta_dropout=ens_cfg["dropout"],
        num_classes=NUM_CLASSES,
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    return model


def build_averaging_ensemble(cfg: dict, checkpoints: list[str]) -> AveragingEnsemble:
    vmoe_cfg = cfg["model"]["vmoe"]
    vmoe_base_cfg = dict(
        backbone=vmoe_cfg["backbone"],
        pretrained=False,
        num_experts=vmoe_cfg["num_experts"],
        top_k=vmoe_cfg["top_k"],
        num_classes=NUM_CLASSES,
        load_balance_coeff=vmoe_cfg["load_balance_coeff"],
        expert_dropout=vmoe_cfg["expert_dropout"],
        freeze_blocks=0,
        cls_head_hidden=vmoe_cfg["cls_head_hidden"],
        attn_head_hidden=vmoe_cfg["attn_head_hidden"],
    )
    n_models = len(checkpoints)
    return AveragingEnsemble(
        vmoe_configs=[vmoe_base_cfg] * n_models,
        checkpoints=checkpoints,
    )


def build_vmoe_model(cfg: dict, checkpoint: str) -> VMoE:
    vmoe_cfg = cfg["model"]["vmoe"]
    model = VMoE(
        backbone=vmoe_cfg["backbone"],
        pretrained=False,
        num_experts=vmoe_cfg["num_experts"],
        top_k=vmoe_cfg["top_k"],
        num_classes=NUM_CLASSES,
        load_balance_coeff=vmoe_cfg["load_balance_coeff"],
        expert_dropout=vmoe_cfg["expert_dropout"],
        freeze_blocks=0,
        cls_head_hidden=vmoe_cfg["cls_head_hidden"],
        attn_head_hidden=vmoe_cfg["attn_head_hidden"],
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    return model


def build_student_model(cfg: dict, checkpoint: str):
    model = build_student(cfg["model"]["student"], num_classes=NUM_CLASSES)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device, collect_features: bool = False):
    """Returns arrays: labels, probs, preds, attn_pred, attn_true, features."""
    model.eval()
    all_labels, all_probs, all_attn_pred, all_attn_true, all_features = [], [], [], [], []

    for batch in loader:
        images     = batch["image"].to(device)
        labels     = batch["label"].numpy()
        attentions = batch["attention"].numpy()

        out = model(images)
        logits     = out["logits"].cpu()
        attn_score = out["attn_score"].cpu().squeeze(-1).numpy()
        probs      = F.softmax(logits, dim=-1).numpy()

        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())
        all_attn_pred.extend(attn_score.tolist())
        all_attn_true.extend(attentions.tolist())

        if collect_features:
            # For ensemble: extract the 25-dim meta-learner input as the feature
            if hasattr(model, "base_models") and hasattr(model, "meta"):
                parts = []
                for bm in model.base_models:
                    bm_out = bm(images)
                    p = F.softmax(bm_out["logits"].detach(), dim=-1)
                    a = bm_out["attn_score"].detach()
                    parts.append(torch.cat([p, a], dim=-1))
                feat = torch.cat(parts, dim=-1).cpu().numpy()
            elif hasattr(out, "get") and out.get("features") is not None:
                feat = out["features"].cpu().numpy()
            else:
                feat = probs  # fallback: use softmax probs as features
            all_features.extend(feat.tolist())

    labels    = np.array(all_labels)
    probs     = np.array(all_probs)
    preds     = probs.argmax(axis=-1)
    attn_pred = np.array(all_attn_pred)
    attn_true = np.array(all_attn_true)
    features  = np.array(all_features) if collect_features else None

    return labels, probs, preds, attn_pred, attn_true, features


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_all_metrics(labels, probs, preds, attn_pred, attn_true, features):
    results = {}

    # --- Confusion matrix ---
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    results["confusion_matrix_abs"]  = cm
    results["confusion_matrix_norm"] = cm_norm

    # --- Classification report ---
    report = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    results["classification_report"] = report

    # --- AUC-ROC per class (OvR) ---
    try:
        auc_macro = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        auc_per_class = {}
        for i, cls in enumerate(CLASS_NAMES):
            binary_labels = (labels == i).astype(int)
            auc_per_class[cls] = roc_auc_score(binary_labels, probs[:, i])
        results["auc_macro"]     = auc_macro
        results["auc_per_class"] = auc_per_class
    except ValueError as e:
        results["auc_macro"]     = float("nan")
        results["auc_per_class"] = {cls: float("nan") for cls in CLASS_NAMES}

    # --- MCC and Kappa ---
    results["mcc"]   = matthews_corrcoef(labels, preds)
    results["kappa"] = cohen_kappa_score(labels, preds)

    # --- Top-k accuracy ---
    results["top1_acc"] = float((preds == labels).mean())
    results["top2_acc"] = float(top_k_accuracy_score(labels, probs, k=2))

    # --- Attention regression ---
    mae  = float(np.abs(attn_true - attn_pred).mean())
    rmse = float(np.sqrt(((attn_true - attn_pred) ** 2).mean()))
    rho, _  = spearmanr(attn_true, attn_pred)
    r, _    = pearsonr(attn_true, attn_pred)
    results["attn_mae"]      = mae
    results["attn_rmse"]     = rmse
    results["attn_spearman"] = float(rho)
    results["attn_pearson"]  = float(r)

    # --- Feature-space distances (class centroids) ---
    if features is not None:
        centroids = np.array([
            features[labels == i].mean(axis=0) for i in range(NUM_CLASSES)
        ])

        # Cosine similarity matrix
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        normed = centroids / (norms + 1e-8)
        cos_sim = normed @ normed.T

        # Euclidean distance matrix
        euc_dist = np.zeros((NUM_CLASSES, NUM_CLASSES))
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                euc_dist[i, j] = np.linalg.norm(centroids[i] - centroids[j])

        results["cosine_similarity"] = cos_sim
        results["euclidean_distance"] = euc_dist

    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_cm(cm, labels, title, pct=False):
    lines = [f"\n{title}"]
    header = "          " + "".join(f"{l:>12}" for l in labels)
    lines.append(header)
    lines.append("          " + "-" * (12 * len(labels)))
    for i, row_label in enumerate(labels):
        short = row_label[:8].ljust(8)
        vals = ""
        for j in range(len(labels)):
            v = cm[i, j]
            if pct:
                vals += f"{v:>11.1%} "
            else:
                vals += f"{v:>11} "
        lines.append(f"  {short}  {vals}  ← predicted")
    return "\n".join(lines)


def fmt_dist_matrix(mat, labels, title, fmt=".4f"):
    lines = [f"\n{title}"]
    header = "          " + "".join(f"{l:>12}" for l in labels)
    lines.append(header)
    lines.append("          " + "-" * (12 * len(labels)))
    for i, row_label in enumerate(labels):
        short = row_label[:8].ljust(8)
        vals = "".join(f"{mat[i,j]:>12{fmt}}" for j in range(len(labels)))
        lines.append(f"  {short}  {vals}")
    return "\n".join(lines)


def generate_report(results, model_label, split, n_samples) -> str:
    r = results
    report_dict = r["classification_report"]
    lines = []

    lines.append(f"# Extended Metrics Report — {model_label}")
    lines.append(f"\n**Split:** {split}  |  **Samples:** {n_samples:,}  |  **Classes:** {', '.join(CLASS_NAMES)}\n")

    # --- Summary ---
    lines.append("---\n")
    lines.append("## Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|:---:|")
    lines.append(f"| Macro F1 | {report_dict['macro avg']['f1-score']:.4f} |")
    lines.append(f"| Weighted F1 | {report_dict['weighted avg']['f1-score']:.4f} |")
    lines.append(f"| Top-1 Accuracy | {r['top1_acc']:.4f} |")
    lines.append(f"| Top-2 Accuracy | {r['top2_acc']:.4f} |")
    lines.append(f"| Macro AUC-ROC | {r['auc_macro']:.4f} |")
    lines.append(f"| Matthews CC | {r['mcc']:.4f} |")
    lines.append(f"| Cohen's Kappa | {r['kappa']:.4f} |")

    # --- Per-class table ---
    lines.append("\n---\n")
    lines.append("## Per-Class Classification Metrics\n")
    lines.append("| Class | Precision | Recall | F1 | AUC-ROC | Support |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|")
    for cls in CLASS_NAMES:
        cr = report_dict[cls]
        auc = r["auc_per_class"].get(cls, float("nan"))
        lines.append(
            f"| {cls} | {cr['precision']:.4f} | {cr['recall']:.4f} | "
            f"{cr['f1-score']:.4f} | {auc:.4f} | {int(cr['support'])} |"
        )
    # averages
    for avg in ["macro avg", "weighted avg"]:
        cr = report_dict[avg]
        label = avg.replace(" avg", " average")
        lines.append(
            f"| **{label}** | {cr['precision']:.4f} | {cr['recall']:.4f} | "
            f"{cr['f1-score']:.4f} | — | {int(cr['support'])} |"
        )

    # --- Confusion matrix ---
    lines.append("\n---\n")
    lines.append("## Confusion Matrix\n")
    lines.append("Rows = true class, Columns = predicted class.\n")
    lines.append("### Absolute counts\n")
    lines.append("```")
    lines.append(fmt_cm(r["confusion_matrix_abs"], CLASS_NAMES, "Absolute", pct=False))
    lines.append("```")
    lines.append("\n### Row-normalised (recall per cell)\n")
    lines.append("```")
    lines.append(fmt_cm(r["confusion_matrix_norm"], CLASS_NAMES, "Normalised", pct=True))
    lines.append("```")

    # --- Attention regression ---
    lines.append("\n---\n")
    lines.append("## Attention Score Regression\n")
    lines.append("| Metric | Value |")
    lines.append("|---|:---:|")
    lines.append(f"| MAE | {r['attn_mae']:.6f} |")
    lines.append(f"| RMSE | {r['attn_rmse']:.6f} |")
    lines.append(f"| Spearman ρ | {r['attn_spearman']:.4f} |")
    lines.append(f"| Pearson r | {r['attn_pearson']:.4f} |")

    # --- Feature-space distances ---
    if "cosine_similarity" in r:
        lines.append("\n---\n")
        lines.append("## Feature-Space Distances (Class Centroids)\n")
        lines.append(
            "Computed on the 25-dim meta-learner input "
            "(concatenated VMoE softmax probabilities + attention scores). "
            "Each class centroid is the mean feature vector of all correctly classified samples in that class.\n"
        )
        lines.append("### Cosine Similarity\n")
        lines.append("1.0 = identical direction, 0.0 = orthogonal, −1.0 = opposite.\n")
        lines.append("```")
        lines.append(fmt_dist_matrix(r["cosine_similarity"], CLASS_NAMES, "Cosine similarity", fmt=".4f"))
        lines.append("```")
        lines.append("\n### Euclidean Distance\n")
        lines.append("Larger = more separated in feature space.\n")
        lines.append("```")
        lines.append(fmt_dist_matrix(r["euclidean_distance"], CLASS_NAMES, "Euclidean distance", fmt=".4f"))
        lines.append("```")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--checkpoints", nargs="+", help="Multiple checkpoints for averaging ensemble")
    p.add_argument("--model",    choices=["ensemble", "vmoe", "student", "averaging"], default="ensemble")
    p.add_argument("--split",    choices=["val", "test"], default="test")
    p.add_argument("--dataset",  default="balanced")
    p.add_argument("--config",   default="configs/config.yaml")
    p.add_argument("--output",   default="METRICS_REPORT.md")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))
    data_cfg   = cfg["data"]
    image_size = data_cfg["image_size"]
    patch_dir  = Path(data_cfg["root"]) / data_cfg["patch_dir"]

    print(f"Loading {args.model} from {args.checkpoint} ...")
    if args.model == "ensemble":
        model = build_ensemble(cfg, args.checkpoint)
    elif args.model == "averaging":
        ckpts = args.checkpoints if args.checkpoints else [args.checkpoint]
        model = build_averaging_ensemble(cfg, ckpts)
    elif args.model == "vmoe":
        model = build_vmoe_model(cfg, args.checkpoint)
    else:
        model = build_student_model(cfg, args.checkpoint)
    model.to(device)
    model.eval()

    csv_path = Path(data_cfg["root"]) / data_cfg["csv_files"][args.dataset]
    ds = PatchDataset(
        csv_path, patch_dir,
        transform=val_transforms(image_size),
        split=args.split,
        val_split=data_cfg["val_split"],
        test_split=data_cfg["test_split"],
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=64, shuffle=False,
        num_workers=data_cfg["num_workers"], pin_memory=True,
    )
    print(f"Running inference on {len(ds):,} {args.split} patches ...")

    collect_features = (args.model == "ensemble")
    labels, probs, preds, attn_pred, attn_true, features = run_inference(
        model, loader, device, collect_features=collect_features
    )

    print("Computing metrics ...")
    results = compute_all_metrics(labels, probs, preds, attn_pred, attn_true, features)

    model_label = f"{args.model.upper()} — {Path(args.checkpoint).name}"
    report_md = generate_report(results, model_label, args.split, len(labels))

    # Print summary to console
    cr = results["classification_report"]
    print(f"\nMacro F1:    {cr['macro avg']['f1-score']:.4f}")
    print(f"Top-1 Acc:   {results['top1_acc']:.4f}")
    print(f"Top-2 Acc:   {results['top2_acc']:.4f}")
    print(f"Macro AUC:   {results['auc_macro']:.4f}")
    print(f"MCC:         {results['mcc']:.4f}")
    print(f"Kappa:       {results['kappa']:.4f}")
    print("\nPer-class F1:")
    for cls in CLASS_NAMES:
        print(f"  {cls:<15} {cr[cls]['f1-score']:.4f}")

    output_path = Path(args.output)
    output_path.write_text(report_md)
    print(f"\nReport written to {output_path}")


if __name__ == "__main__":
    main()
