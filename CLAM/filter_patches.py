"""
Filter extracted patches by attention score for any label.

Selects per slide:
  - Top-K patches by attention          -> patch_dataset/{label}_top{k}/
  - Mid-band [start%, end%] patches     -> patch_dataset/{label}_mid{start}_{end}/

Copies PNG files from the source patch directory and writes a kept CSV.

Usage
-----
  python filter_patches.py --label atypical
  python filter_patches.py --label benign --top-k 1000 --mid-start 20 --mid-end 40
  python filter_patches.py --label malignant --input-csv my_scores.csv --src-dir raw_patches/malignant
"""
import argparse
import os
import shutil
import pandas as pd
from tqdm import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Filter patches by attention score for any label.")
    parser.add_argument("--label", required=True,
                        help="Label name (e.g. benign, atypical, malignant, non-cancer). "
                             "Used to build default path names.")
    parser.add_argument("--input-csv", default=None,
                        help="Input attention scores CSV. Default: attention_scores_new_{label}.csv")
    parser.add_argument("--src-dir", default=None,
                        help="Directory containing extracted PNG patches. Default: patch_dataset/{label}/")
    parser.add_argument("--top-k", type=int, default=1000,
                        help="Keep the top-K patches by attention per slide (default: 1000).")
    parser.add_argument("--mid-start", type=int, default=20,
                        help="Start of mid-band percentile (default: 20).")
    parser.add_argument("--mid-end", type=int, default=40,
                        help="End of mid-band percentile (default: 40).")
    parser.add_argument("--output-csv", default=None,
                        help="Output kept-patches CSV. Default: attention_scores_{label}_kept.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    label = args.label

    input_csv  = args.input_csv  or os.path.join(BASE, f"attention_scores_new_{label}.csv")
    src_dir    = args.src_dir    or os.path.join(BASE, "patch_dataset", label)
    output_csv = args.output_csv or os.path.join(BASE, f"attention_scores_{label}_kept.csv")

    top_k      = args.top_k
    mid_start  = args.mid_start
    mid_end    = args.mid_end

    top_dir = os.path.join(BASE, "patch_dataset", f"{label}_top{top_k}")
    mid_dir = os.path.join(BASE, "patch_dataset", f"{label}_mid{mid_start}_{mid_end}")

    os.makedirs(top_dir, exist_ok=True)
    os.makedirs(mid_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    kept_rows = []

    for slide_id, g in df.groupby("slide_id"):
        n = len(g)
        sorted_g = g.sort_values("attention", ascending=False).reset_index(drop=True)

        top = sorted_g.iloc[:top_k].copy()
        top["subset"] = f"top{top_k}"
        kept_rows.append(top)

        i_start = int(n * mid_start / 100)
        i_end   = int(n * mid_end   / 100)
        band    = sorted_g.iloc[i_start:i_end].copy()
        band["subset"] = f"mid{mid_start}_{mid_end}"
        kept_rows.append(band)

        print(f"{slide_id}: total={n}, top{top_k}={len(top)}, band_{mid_start}_{mid_end}={len(band)}")

    kept_df = pd.concat(kept_rows).drop_duplicates(subset=["slide_id", "x", "y"]).reset_index(drop=True)
    kept_df.to_csv(output_csv, index=False)
    print(f"\nKept CSV: {len(kept_df)} rows -> {output_csv}")

    errors = 0
    for _, row in tqdm(kept_df.iterrows(), total=len(kept_df), desc="Copying patches"):
        fname   = f"{row.slide_id}_{int(row.x)}_{int(row.y)}.png"
        src     = os.path.join(src_dir, fname)
        dst_dir = top_dir if row.subset.startswith("top") else mid_dir
        dst     = os.path.join(dst_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"[WARN] Not found: {src}")
            errors += 1

    print(f"\nDone!")
    print(f"  {len(os.listdir(top_dir))} images in {top_dir}")
    print(f"  {len(os.listdir(mid_dir))} images in {mid_dir}")
    print(f"  Errors: {errors}")


if __name__ == "__main__":
    main()
