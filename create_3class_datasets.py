#!/usr/bin/env python3
"""
Create 3-class versions of the five training CSVs.

Strategy
--------
Every "non-cancer" patch in the existing CSVs originated from a slide in
patch_dataset/{benign,atypical,malignant}/.  The CLAM attention model assigned
it the "non-cancer" label because it had low attention, but the slide-level
ground truth is one of the three positive classes.

This script:
  1. Builds a filename → slide_class index from all three patch_dataset/ folders.
  2. For each of the five clean CSVs, relabels non-cancer rows to their slide's
     true class (benign / atypical / malignant).
  3. Adds symlinks in organized_patches/{benign,atypical,malignant}/ so that
     PatchDataset can find the relabeled images at the new label path.
  4. Saves five new CSVs: dataset_{name}_3class.csv

The result is a 3-class problem: benign / atypical / malignant.
The original 4-class CSVs are not modified.

Usage
-----
  python create_3class_datasets.py [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------

BASE_DIR  = Path(__file__).parent
PATCH_DIR = BASE_DIR / "patch_dataset"
ORG_DIR   = BASE_DIR / "organized_patches"

SOURCE_CLASSES = ["benign", "atypical", "malignant"]

INPUT_CSVS = [
    ("dataset_balanced_clean.csv",       "dataset_balanced_3class.csv"),
    ("dataset_neg_heavy_clean.csv",      "dataset_neg_heavy_3class.csv"),
    ("dataset_benign_heavy_clean.csv",   "dataset_benign_heavy_3class.csv"),
    ("dataset_atypical_heavy_clean.csv", "dataset_atypical_heavy_3class.csv"),
    ("dataset_malignant_heavy_clean.csv","dataset_malignant_heavy_3class.csv"),
]


# ---------------------------------------------------------------------------
# Step 1: Build filename → slide_class index
# ---------------------------------------------------------------------------

def build_slide_class_index() -> dict[str, str]:
    """Map each image filename to its source folder (slide class)."""
    index: dict[str, str] = {}
    for cls in SOURCE_CLASSES:
        folder = PATCH_DIR / cls
        if not folder.exists():
            print(f"  WARNING: {folder} not found — skipping {cls}.")
            continue
        for f in folder.iterdir():
            if f.suffix == ".png":
                index[f.name] = cls
    print(f"  Slide-class index: {len(index):,} files indexed across {SOURCE_CLASSES}.")
    return index


# ---------------------------------------------------------------------------
# Step 2: Relabel and save CSVs
# ---------------------------------------------------------------------------

def relabel_csv(
    df: pd.DataFrame,
    slide_class_index: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relabel all non-cancer rows to their original slide class.

    Returns
    -------
    df_out     : full dataframe with relabeled rows, non-cancer dropped if unmapped
    changed_df : rows whose label changed (needed for symlink creation)
    """
    df = df.copy()
    nc_mask = df["label"] == "non-cancer"
    nc_df   = df[nc_mask].copy()

    # Build filename for lookup
    nc_df["_fname"] = (
        nc_df["slide_id"].astype(str) + "_"
        + nc_df["x"].astype(str) + "_"
        + nc_df["y"].astype(str) + ".png"
    )
    nc_df["_new_label"] = nc_df["_fname"].map(slide_class_index)

    unmapped = nc_df["_new_label"].isna().sum()
    if unmapped:
        print(f"  WARNING: {unmapped:,} non-cancer rows have no source folder match — dropping.")

    mapped = nc_df.dropna(subset=["_new_label"])
    for idx, row in mapped.iterrows():
        df.loc[idx, "label"] = row["_new_label"]

    # Drop any that couldn't be mapped (shouldn't happen if patch_dataset is complete)
    unmapped_idx = nc_df[nc_df["_new_label"].isna()].index
    df = df.drop(index=unmapped_idx).reset_index(drop=True)

    changed_df = df[df["label"].isin(SOURCE_CLASSES)].copy()
    # Only the rows that were originally non-cancer
    changed_df = mapped.rename(columns={"_new_label": "new_label"})[
        ["slide_id", "x", "y", "new_label"]
    ].rename(columns={"new_label": "label"})

    return df, changed_df


# ---------------------------------------------------------------------------
# Step 3: Create organized_patches/ symlinks for relabeled patches
# ---------------------------------------------------------------------------

def create_symlinks(changed_df: pd.DataFrame, dry_run: bool) -> None:
    """For each relabeled patch, add a symlink in organized_patches/{new_label}/."""
    n_created = n_skipped = n_exists = 0

    for _, row in changed_df.iterrows():
        fname     = f"{row['slide_id']}_{row['x']}_{row['y']}.png"
        new_label = row["label"]

        new_dst = ORG_DIR / new_label / fname
        if new_dst.exists() or new_dst.is_symlink():
            n_exists += 1
            continue

        # Resolve the target: prefer existing non-cancer symlink, else patch_dataset/
        nc_src = ORG_DIR / "non-cancer" / fname
        if nc_src.is_symlink():
            target = nc_src.resolve()
        elif nc_src.is_file():
            target = nc_src
        else:
            # Fall back to patch_dataset/ source folder
            target = PATCH_DIR / new_label / fname

        if not target.exists():
            n_skipped += 1
            continue

        if not dry_run:
            new_dst.symlink_to(target)
        n_created += 1

    print(f"  Symlinks: {n_created:,} created, {n_exists:,} already existed, {n_skipped:,} skipped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview counts without writing files or symlinks.")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] No files will be written.\n")

    print("=" * 60)
    print("Step 1: Build slide-class index")
    print("=" * 60)
    slide_class_index = build_slide_class_index()

    for in_name, out_name in INPUT_CSVS:
        in_path  = BASE_DIR / in_name
        out_path = BASE_DIR / out_name

        if not in_path.exists():
            print(f"\nWARNING: {in_name} not found — skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing: {in_name} → {out_name}")
        print("=" * 60)

        df = pd.read_csv(in_path)
        before = df["label"].value_counts().to_dict()
        print(f"  Before: {before}")

        df_out, changed_df = relabel_csv(df, slide_class_index)
        after = df_out["label"].value_counts().to_dict()
        print(f"  After:  {after}")
        print(f"  Relabeled {len(changed_df):,} non-cancer → slide class rows.")

        print("  Creating organized_patches/ symlinks …")
        create_symlinks(changed_df, dry_run=args.dry_run)

        if not args.dry_run:
            df_out.to_csv(out_path, index=False)
            print(f"  Saved → {out_path}")

    print("\nDone.")
    if not args.dry_run:
        print("\nAdd to configs/config_3class.yaml:")
        for _, out_name in INPUT_CSVS:
            key = out_name.replace("dataset_", "").replace("_3class.csv", "")
            print(f"  {key}: {out_name}")


if __name__ == "__main__":
    main()
