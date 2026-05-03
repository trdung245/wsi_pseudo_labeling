#!/usr/bin/env python3
"""
Merge added_data patches into the existing pipeline and regenerate *_clean.csv files.

Steps
-----
1. Copy new patch images from added_data/ subfolders into patch_dataset/
2. Append new entries to dataset_pos_heavy.csv (the raw pool)
3. Re-run organize_dataset.py to update organized_patches/ symlinks
4. Re-run prepare_datasets.py to regenerate all 5 *_clean.csv files

New data:
  added_data/top30_40_atypical/  → patch_dataset/atypical/   (9838 files)
  added_data/benign_top1000/     → patch_dataset/benign/     (3000 files)
  added_data/benign_mid20_40/    → patch_dataset/benign/     (6391 files)

Usage
-----
  python merge_added_data.py [--dry-run] [--no-rerun]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE_DIR    = Path(__file__).parent
PATCH_DIR   = BASE_DIR / "patch_dataset"
ADDED_DIR   = BASE_DIR / "added_data"
POS_HEAVY   = BASE_DIR / "dataset_pos_heavy.csv"
BEN_CSV     = BASE_DIR / "attention_scores_benign_kept.csv"
ATY_CSV     = BASE_DIR / "atypical_attention_scores_top30_40_2000.csv"

FOLDER_MAP = {
    "top30_40_atypical": "atypical",
    "benign_top1000":    "benign",
    "benign_mid20_40":   "benign",
}


def copy_images(dry_run: bool) -> dict[str, int]:
    counts: dict[str, int] = {}
    for src_folder, dst_class in FOLDER_MAP.items():
        src_dir = ADDED_DIR / src_folder
        dst_dir = PATCH_DIR / dst_class
        files = list(src_dir.glob("*.png"))
        print(f"  {src_folder}/ → patch_dataset/{dst_class}/  ({len(files)} files)")
        n = 0
        for f in files:
            dst = dst_dir / f.name
            if not dst.exists():
                if not dry_run:
                    shutil.copy2(f, dst)
                n += 1
        counts[src_folder] = n
        print(f"    Copied {n} new files ({len(files)-n} already existed).")
    return counts


def append_to_raw_pool(dry_run: bool) -> None:
    existing = pd.read_csv(POS_HEAVY)
    existing_keys = set(
        zip(existing["slide_id"], existing["x"].astype(str), existing["y"].astype(str))
    )
    print(f"  dataset_pos_heavy.csv existing rows: {len(existing):,}")

    # Atypical entries
    df_aty = pd.read_csv(ATY_CSV)[["slide_id", "patch_id", "x", "y", "label", "attention"]]
    new_aty = df_aty[
        ~df_aty.apply(
            lambda r: (r["slide_id"], str(r["x"]), str(r["y"])) in existing_keys, axis=1
        )
    ]
    print(f"  New atypical rows to append: {len(new_aty):,}")

    # Benign entries (drop the 'subset' column)
    df_ben = pd.read_csv(BEN_CSV)[["slide_id", "patch_id", "x", "y", "label", "attention"]]
    new_ben = df_ben[
        ~df_ben.apply(
            lambda r: (r["slide_id"], str(r["x"]), str(r["y"])) in existing_keys, axis=1
        )
    ]
    print(f"  New benign rows to append:   {len(new_ben):,}")

    combined_new = pd.concat([new_aty, new_ben], ignore_index=True)
    updated = pd.concat([existing, combined_new], ignore_index=True)
    print(f"  dataset_pos_heavy.csv updated rows: {len(updated):,}")

    if not dry_run:
        updated.to_csv(POS_HEAVY, index=False)
        print(f"  Saved → {POS_HEAVY}")
    else:
        print("  [DRY RUN] Not saved.")


def run_pipeline(dry_run: bool) -> None:
    scripts = ["organize_dataset.py", "prepare_datasets.py"]
    for script in scripts:
        print(f"\n  Running {script} ...")
        if not dry_run:
            result = subprocess.run(
                [sys.executable, str(BASE_DIR / script)],
                cwd=str(BASE_DIR),
                capture_output=False,
            )
            if result.returncode != 0:
                print(f"  ERROR: {script} exited with code {result.returncode}")
                sys.exit(result.returncode)
        else:
            print(f"  [DRY RUN] Would run: python {script}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true", help="Preview without modifying files.")
    parser.add_argument("--no-rerun", action="store_true", help="Skip re-running organize/prepare scripts.")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] No files will be modified.\n")

    print("=" * 60)
    print("Step 1: Copy images to patch_dataset/")
    print("=" * 60)
    copy_images(args.dry_run)

    print("\n" + "=" * 60)
    print("Step 2: Append new entries to dataset_pos_heavy.csv")
    print("=" * 60)
    append_to_raw_pool(args.dry_run)

    if not args.no_rerun:
        print("\n" + "=" * 60)
        print("Step 3: Re-run organize_dataset.py + prepare_datasets.py")
        print("=" * 60)
        run_pipeline(args.dry_run)
    else:
        print("\nStep 3: Skipped (--no-rerun).")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
