#!/usr/bin/env python3
"""
Prepare five training dataset CSVs with boundary relabeling.

Steps
-----
1. RELABEL  – Non-cancer patches in benign/atypical slides that have the
   highest attention scores (top --relabel-frac per slide) are reassigned to
   their slide's true class, recovering boundary-region training examples.
   Malignant slides are left unchanged (malignant tissue is more distinctive).

2. FILTER   – Remove rows whose image does not exist in organized_patches/.

3. RESAMPLE – Build five CSV files with different class ratios:
     balanced         1 : 1 : 1 : 1
     neg_heavy        non-cancer ≈ multiplier × mean(cancer classes)
     benign_heavy     benign ≈ multiplier × mean(other classes)
     atypical_heavy   atypical ≈ multiplier × mean(other classes)
     malignant_heavy  malignant ≈ multiplier × mean(other classes)

4. SYNC     – Update organized_patches/ symlinks for relabeled patches so
   dataset.py finds them in the correct class subfolder.

Outputs
-------
  dataset_balanced_clean.csv
  dataset_neg_heavy_clean.csv
  dataset_benign_heavy_clean.csv
  dataset_atypical_heavy_clean.csv
  dataset_malignant_heavy_clean.csv

Usage
-----
  python prepare_datasets.py [--seed 42] [--relabel-frac 0.20] [--multiplier 3.0]
  python prepare_datasets.py --no-relabel   # skip relabeling step
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR  = Path(__file__).parent
ORG_DIR   = BASE_DIR / "organized_patches"
PATCH_DIR = BASE_DIR / "patch_dataset"
CLASSES   = ["non-cancer", "benign", "atypical", "malignant"]

RAW_CSVS = [
    BASE_DIR / "dataset_balanced.csv",
    BASE_DIR / "dataset_neg_heavy.csv",
    BASE_DIR / "dataset_pos_heavy.csv",
]

# Added patch CSVs — used to tag rows for proportional sampling
ADDED_BEN_CSV = BASE_DIR / "attention_scores_benign_kept.csv"
ADDED_ATY_CSV = BASE_DIR / "atypical_attention_scores_top30_40_2000.csv"
ADDED_FRAC    = 0.45   # target fraction of benign/atypical sampled from added patches

# Output (filename, dataset_kind) pairs
DATASETS = [
    ("dataset_balanced_clean.csv",       "balanced"),
    ("dataset_neg_heavy_clean.csv",      "neg_heavy"),
    ("dataset_benign_heavy_clean.csv",   "benign_heavy"),
    ("dataset_atypical_heavy_clean.csv", "atypical_heavy"),
    ("dataset_malignant_heavy_clean.csv","malignant_heavy"),
]


# ---------------------------------------------------------------------------
# Load & deduplicate raw CSVs
# ---------------------------------------------------------------------------

def load_combined_pool() -> pd.DataFrame:
    """Load all raw CSVs and deduplicate by (slide_id, x, y).

    Priority: dataset_balanced > dataset_neg_heavy > dataset_pos_heavy.
    """
    frames = []
    for pri, csv_path in enumerate(RAW_CSVS):
        if not csv_path.exists():
            print(f"  WARNING: {csv_path.name} not found — skipping.")
            continue
        df = pd.read_csv(csv_path)
        df["_pri"] = pri
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.sort_values("_pri")
        .drop_duplicates(subset=["slide_id", "x", "y"], keep="first")
        .drop(columns=["_pri"])
        .reset_index(drop=True)
    )
    print(f"  Combined pool: {len(combined):,} unique patches.")
    return combined


# ---------------------------------------------------------------------------
# Slide-class detection
# ---------------------------------------------------------------------------

def _build_slide_class_index() -> dict[str, str]:
    """Build {filename: slide_class} for benign and atypical source folders."""
    index: dict[str, str] = {}
    for cls in ["benign", "atypical"]:
        folder = PATCH_DIR / cls
        if not folder.exists():
            print(f"  WARNING: {folder} not found — slide-class detection skipped for {cls}.")
            continue
        for f in folder.iterdir():
            if f.suffix == ".png":
                index[f.name] = cls
    print(f"  Slide-class index: {len(index):,} benign/atypical source files indexed.")
    return index


# ---------------------------------------------------------------------------
# Relabeling
# ---------------------------------------------------------------------------

def relabel_boundary_patches(
    df: pd.DataFrame,
    relabel_top_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relabel top-attention non-cancer patches from benign/atypical slides.

    For each benign/atypical slide, the non-cancer patches with the highest
    attention scores (top `relabel_top_frac` per slide) are relabeled to
    their slide's true class.  Malignant slides are left unchanged.

    Returns
    -------
    df_updated : pd.DataFrame
        Full pool with relabeled labels.
    changed_df : pd.DataFrame
        Subset of rows whose label changed (used for symlink update).
    """
    df = df.copy()
    slide_class_index = _build_slide_class_index()

    if not slide_class_index:
        print("  No slide-class index built — relabeling skipped.")
        return df, pd.DataFrame()

    # Select non-cancer patches and map each to its slide class
    nc_mask = df["label"] == "non-cancer"
    nc_df = df[nc_mask].copy()
    nc_df["fname"] = (
        nc_df["slide_id"].astype(str) + "_"
        + nc_df["x"].astype(str) + "_"
        + nc_df["y"].astype(str) + ".png"
    )
    nc_df["_slide_class"] = nc_df["fname"].map(slide_class_index)

    changed_indices: list[int] = []
    counts = {"benign": 0, "atypical": 0}

    print("  Identifying boundary patches per slide …")
    for slide_class in ["benign", "atypical"]:
        subset = nc_df[nc_df["_slide_class"] == slide_class]
        if subset.empty:
            continue
        for _slide_id, slide_patches in subset.groupby("slide_id"):
            n = len(slide_patches)
            n_relabel = max(1, int(round(n * relabel_top_frac)))
            top_idx = slide_patches.nlargest(n_relabel, "attention").index.tolist()
            changed_indices.extend(top_idx)
            counts[slide_class] += n_relabel

    # Build index→new_label map from nc_df, then apply to df
    idx_to_new_label = nc_df.loc[changed_indices, "_slide_class"].to_dict()
    for idx, new_label in idx_to_new_label.items():
        df.loc[idx, "label"] = new_label

    changed_df = df.loc[changed_indices].copy()

    print(
        f"  Relabeled → benign: {counts['benign']:,} | atypical: {counts['atypical']:,} "
        f"(top {relabel_top_frac:.0%} per slide)"
    )
    return df, changed_df


# ---------------------------------------------------------------------------
# Update organized_patches/ symlinks
# ---------------------------------------------------------------------------

def update_organized_patches(changed_df: pd.DataFrame) -> None:
    """Move symlinks for relabeled patches from non-cancer/ to their new label folder."""
    if changed_df.empty:
        return

    n_updated = n_skipped = 0
    for _, row in changed_df.iterrows():
        fname     = f"{row['slide_id']}_{row['x']}_{row['y']}.png"
        new_label = row["label"]

        old_dst = ORG_DIR / "non-cancer" / fname
        new_dst = ORG_DIR / new_label / fname

        # Resolve the symlink target before unlinking
        if old_dst.is_symlink():
            src = old_dst.resolve()
        elif old_dst.is_file():
            # Organized with --copy; find source in patch_dataset/
            src_path = PATCH_DIR / new_label / fname
            src = src_path if src_path.exists() else None
        else:
            # Already moved or never existed in non-cancer/
            src = None

        if src is None or not Path(src).exists():
            n_skipped += 1
            continue

        # Remove old entry
        if old_dst.is_symlink() or old_dst.exists():
            old_dst.unlink()

        # Create new symlink (overwrite if stale)
        if new_dst.is_symlink() or new_dst.exists():
            new_dst.unlink()
        new_dst.symlink_to(src)
        n_updated += 1

    print(f"  Symlinks updated: {n_updated:,} moved, {n_skipped:,} skipped (source not found).")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def tag_added_patches(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean columns marking patches from the newly added datasets.

    is_added_mid20_40 — benign patches from the benign_mid20_40 subset
    is_added_top30_40 — atypical patches from the top30_40_atypical folder
    """
    df = df.copy()

    # Benign: only the mid20_40 subset (not top1000) from attention_scores_benign_kept.csv
    ben_raw = pd.read_csv(ADDED_BEN_CSV)
    ben_mid = ben_raw[ben_raw["subset"] == "mid20_40"]
    ben_keys = set(zip(ben_mid["slide_id"], ben_mid["x"].astype(str), ben_mid["y"].astype(str)))

    # Atypical: all entries in atypical_attention_scores_top30_40_2000.csv
    aty_raw = pd.read_csv(ADDED_ATY_CSV)
    aty_keys = set(zip(aty_raw["slide_id"], aty_raw["x"].astype(str), aty_raw["y"].astype(str)))

    df["is_added_mid20_40"] = df.apply(
        lambda r: (r["slide_id"], str(int(r["x"])), str(int(r["y"]))) in ben_keys, axis=1
    )
    df["is_added_top30_40"] = df.apply(
        lambda r: (r["slide_id"], str(int(r["x"])), str(int(r["y"]))) in aty_keys, axis=1
    )

    n_ben = df["is_added_mid20_40"].sum()
    n_aty = df["is_added_top30_40"].sum()
    print(f"  Tagged {n_ben:,} added benign (mid20_40) and {n_aty:,} added atypical (top30_40) patches.")
    return df


def filter_existing(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose image exists in organized_patches/."""
    def exists(row) -> bool:
        return (
            ORG_DIR / row["label"] / f"{row['slide_id']}_{row['x']}_{row['y']}.png"
        ).exists()
    mask = df.apply(exists, axis=1)
    print(f"  Dropped {(~mask).sum():,} rows with missing images ({mask.sum():,} kept).")
    return df[mask].reset_index(drop=True)


def resample(
    df: pd.DataFrame,
    targets: dict[str, int],
    rng: np.random.Generator,
    added_fracs: dict[str, tuple[str, float]] | None = None,
) -> pd.DataFrame:
    """Sample exactly targets[cls] rows per class (without replacement).

    added_fracs: optional {cls: (bool_col, target_frac)} — ensures approximately
    target_frac of the sampled rows for that class have bool_col == True.
    If there are not enough tagged rows, all tagged rows are used and the
    remainder is filled from untagged rows.
    """
    parts: list[pd.DataFrame] = []
    for cls in CLASSES:
        subset = df[df["label"] == cls]
        n_tgt  = targets.get(cls, 0)

        if added_fracs and cls in added_fracs:
            col, frac = added_fracs[cls]
            added    = subset[subset[col]]
            original = subset[~subset[col]]

            n_added_want = int(round(n_tgt * frac))
            n_orig_want  = n_tgt - n_added_want

            n_added_use = min(n_added_want, len(added))
            n_orig_use  = min(n_orig_want + (n_added_want - n_added_use), len(original))
            n_total     = n_added_use + n_orig_use

            if n_total < n_tgt:
                print(f"  WARNING: {cls} only {n_total:,} available; requested {n_tgt:,}. Using all.")

            chunk: list[pd.DataFrame] = []
            if n_added_use > 0:
                chunk.append(added.sample(n=n_added_use, replace=False,
                                          random_state=int(rng.integers(0, 1 << 31))))
            if n_orig_use > 0:
                chunk.append(original.sample(n=n_orig_use, replace=False,
                                             random_state=int(rng.integers(0, 1 << 31))))
            parts.append(pd.concat(chunk, ignore_index=True))
        else:
            if len(subset) < n_tgt:
                print(f"  WARNING: {cls} has only {len(subset):,} rows; requested {n_tgt:,}. Using all.")
                parts.append(subset)
            else:
                parts.append(
                    subset.sample(n=n_tgt, replace=False, random_state=int(rng.integers(0, 1 << 31)))
                )
    return pd.concat(parts, ignore_index=True)


def print_dist(df: pd.DataFrame, label: str) -> None:
    total  = len(df)
    counts = df["label"].value_counts()
    print(f"\n  {label}  ({total:,} total)")
    for cls in CLASSES:
        n = counts.get(cls, 0)
        print(f"    {cls:<15}: {n:>6,}  ({100 * n / max(total, 1):5.1f}%)")


def available_counts(df: pd.DataFrame) -> dict[str, int]:
    return df["label"].value_counts().to_dict()


# ---------------------------------------------------------------------------
# Resampling strategies
# ---------------------------------------------------------------------------

def targets_balanced(avail: dict[str, int]) -> dict[str, int]:
    """1:1:1:1 — cap at the smallest available class."""
    cap = min(avail.get(c, 0) for c in CLASSES)
    return {c: cap for c in CLASSES}


def targets_neg_heavy(avail: dict[str, int], multiplier: float = 3.0) -> dict[str, int]:
    """non-cancer is the heavy class: non-cancer = multiplier × mean(other classes).
    Uses the same logic as targets_class_heavy with heavy_class='non-cancer'.
    """
    return targets_class_heavy(avail, "non-cancer", multiplier)


def targets_class_heavy(
    avail: dict[str, int],
    heavy_class: str,
    multiplier: float = 3.0,
) -> dict[str, int]:
    """heavy_class kept at full availability; other classes scaled down so that
    heavy_class = multiplier × mean(other classes).

    Solving:  heavy = multiplier × mean(others)
    We fix heavy = avail[heavy_class] and scale others proportionally.
    If others are already small enough, keep them at natural counts.
    """
    heavy_avail   = avail.get(heavy_class, 0)
    other_classes = [c for c in CLASSES if c != heavy_class]
    others_avail  = {c: avail.get(c, 0) for c in other_classes}

    # Target mean for other classes so that heavy = multiplier * mean(others)
    other_mean_target = heavy_avail / multiplier
    others_total_avail = sum(others_avail.values())
    others_total_target = int(round(other_mean_target * len(other_classes)))

    if others_total_target >= others_total_avail:
        # No downsampling needed — heavy is already dominant
        return {heavy_class: heavy_avail, **others_avail}

    # Scale each other class proportionally
    targets: dict[str, int] = {heavy_class: heavy_avail}
    for cls, n in others_avail.items():
        targets[cls] = max(1, int(round(n * others_total_target / others_total_avail)))

    # Fix rounding drift
    drift = others_total_target - sum(targets[c] for c in other_classes)
    if drift != 0:
        largest_other = max(other_classes, key=lambda c: targets[c])
        targets[largest_other] += drift

    return targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--multiplier",         type=float, default=3.0,
                        help="Heavy-class multiplier (default 3).")
    parser.add_argument("--relabel-frac",       type=float, default=0.20,
                        help="Top fraction of per-slide non-cancer patches to relabel (default 0.20).")
    parser.add_argument("--no-relabel",         action="store_true",
                        help="Skip the boundary relabeling step.")
    parser.add_argument("--max-per-slide",      type=int,   default=300,
                        help="Max patches per (slide_id, label) kept before resampling. "
                             "Prevents mega-slides from dominating a split. Default 300.")
    args = parser.parse_args()

    rng = np.random.default_rng(seed=args.seed)

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 1: Load and combine raw CSVs")
    print("=" * 60)
    pool = load_combined_pool()
    print_dist(pool, "Raw combined pool")

    # ------------------------------------------------------------------
    changed_df = pd.DataFrame()
    if not args.no_relabel:
        print("\n" + "=" * 60)
        print("Step 2: Relabel boundary non-cancer patches")
        print("=" * 60)
        pool, changed_df = relabel_boundary_patches(pool, args.relabel_frac)
        print_dist(pool, "After relabeling")

        if not changed_df.empty:
            print("\n" + "=" * 60)
            print("Step 3: Sync organized_patches/ symlinks")
            print("=" * 60)
            update_organized_patches(changed_df)
    else:
        print("\nStep 2: Relabeling skipped (--no-relabel).")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Filter to existing images")
    print("=" * 60)
    pool = filter_existing(pool)
    print_dist(pool, "After filtering")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4b: Tag added patches for proportional sampling")
    print("=" * 60)
    pool = tag_added_patches(pool)

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4c: Cap patches per slide to prevent split imbalance")
    print("=" * 60)
    if args.max_per_slide > 0:
        before = len(pool)
        pool = (
            pool.groupby(["slide_id", "label"], group_keys=False)
            .apply(lambda g: g.sample(
                n=min(len(g), args.max_per_slide),
                random_state=42,
            ))
            .reset_index(drop=True)
        )
        print(f"  Capped at {args.max_per_slide} patches per (slide, label): "
              f"{before:,} → {len(pool):,} rows.")
        print_dist(pool, "After per-slide cap")
    else:
        print("  Skipped (--max-per-slide=0).")

    avail = available_counts(pool)

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Resample into 5 datasets")
    print("=" * 60)

    strategy_map = {
        "balanced":       lambda: targets_balanced(avail),
        "neg_heavy":      lambda: targets_neg_heavy(avail, args.multiplier),
        "benign_heavy":   lambda: targets_class_heavy(avail, "benign",    args.multiplier),
        "atypical_heavy": lambda: targets_class_heavy(avail, "atypical",  args.multiplier),
        "malignant_heavy":lambda: targets_class_heavy(avail, "malignant", args.multiplier),
    }

    # ~45% of sampled benign rows come from mid20_40, ~45% of atypical from top30_40
    ADDED_FRACS = {
        "benign":   ("is_added_mid20_40", ADDED_FRAC),
        "atypical": ("is_added_top30_40", ADDED_FRAC),
    }
    TAG_COLS = ["is_added_mid20_40", "is_added_top30_40"]

    for dst_name, kind in DATASETS:
        targets  = strategy_map[kind]()
        print(f"\n  [{kind}]  targets: { {c: targets[c] for c in CLASSES} }")
        df_out   = resample(pool, targets, rng, added_fracs=ADDED_FRACS)
        print_dist(df_out, f"After resampling [{kind}]")
        # Report actual added-patch fractions
        for cls, (col, _) in ADDED_FRACS.items():
            cls_rows = df_out[df_out["label"] == cls]
            if len(cls_rows):
                frac_actual = cls_rows[col].mean()
                print(f"    {cls} added fraction: {frac_actual:.1%}  ({col})")
        out_path = BASE_DIR / dst_name
        df_out.drop(columns=TAG_COLS, errors="ignore").to_csv(out_path, index=False)
        print(f"  Saved → {out_path}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Done.  Update configs/config.yaml csv_files to:")
    for dst_name, kind in DATASETS:
        print(f"  {kind}: {dst_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
