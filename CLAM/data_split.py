"""
Pseudo-label patches by CLAM attention score.

Default strategy (sorted by attention, high → low)
---------------------------------------------------
  0  – top_pct %         →  slide's true label   (most informative tissue)
  top_pct – mid_start %  →  dropped               (ambiguous boundary)
  mid_start – mid_end %  →  slide's true label,   (boundary diversity;
                             benign & atypical only  only for softer classes)
  mid_end – 100 %        →  non-cancer            (background tissue)

The pseudo_label column stores the final class name directly
(e.g. "benign", "atypical", "malignant", "non-cancer") so downstream
cleanup.py can simply promote it to the label column.

Usage
-----
  python data_split.py                          # defaults
  python data_split.py --top-pct 15 --mid-start-pct 25 --mid-end-pct 45
  python data_split.py --mid-labels benign atypical malignant
  python data_split.py --output-csv pseudo_strict.csv --top-pct 5
"""
import argparse
import os

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Pseudo-label patches by CLAM attention score.")
    p.add_argument("--input-csv", default=os.path.join(BASE, "attention_scores_all_slides.csv"),
                   help="Input attention scores CSV (default: attention_scores_all_slides.csv).")
    p.add_argument("--output-csv", default=os.path.join(BASE, "pseudo_labels.csv"),
                   help="Output pseudo-labeled CSV (default: pseudo_labels.csv).")
    p.add_argument("--top-pct", type=float, default=10.0,
                   help="Top-%% of patches (highest attention) assigned slide label. Default: 10")
    p.add_argument("--mid-start-pct", type=float, default=30.0,
                   help="Percentile where mid-band starts (after ignored zone). Default: 30")
    p.add_argument("--mid-end-pct", type=float, default=40.0,
                   help="Percentile where mid-band ends / non-cancer begins. Default: 40")
    p.add_argument("--mid-labels", nargs="+", default=["benign", "atypical"],
                   help="Slide labels that receive mid-band patches. Default: benign atypical")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (unused currently, reserved for future sampling).")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    results = []

    for slide_id, group in df.groupby("slide_id"):
        slide_label = group["label"].iloc[0]
        group = group.sort_values("attention", ascending=False).reset_index(drop=True)
        n = len(group)

        i_top      = int(n * args.top_pct       / 100)
        i_mid_s    = int(n * args.mid_start_pct / 100)
        i_mid_e    = int(n * args.mid_end_pct   / 100)

        # Top region → slide's true label
        top = group.iloc[:i_top].copy()
        top["pseudo_label"] = slide_label
        results.append(top)

        # Mid-band → slide's true label (boundary diversity), selected labels only
        if slide_label in args.mid_labels and i_mid_s < i_mid_e:
            mid = group.iloc[i_mid_s:i_mid_e].copy()
            mid["pseudo_label"] = slide_label
            results.append(mid)

        # Bottom region → non-cancer (background tissue)
        if i_mid_e < n:
            neg = group.iloc[i_mid_e:].copy()
            neg["pseudo_label"] = "non-cancer"
            results.append(neg)

    out_df = pd.concat(results).reset_index(drop=True)
    out_df.to_csv(args.output_csv, index=False)

    print(f"\nSaved {len(out_df)} rows → {args.output_csv}")
    print(f"\nThresholds (% of patches per slide, sorted by attention ↓):")
    print(f"  top {args.top_pct:.0f}%                    → slide label")
    print(f"  {args.top_pct:.0f}–{args.mid_start_pct:.0f}%                    → dropped (ignored zone)")
    if args.mid_labels:
        print(f"  {args.mid_start_pct:.0f}–{args.mid_end_pct:.0f}% ({', '.join(args.mid_labels)}) → slide label (boundary)")
    print(f"  {args.mid_end_pct:.0f}–100%                   → non-cancer")
    print(f"\nPseudo-label distribution:")
    print(out_df["pseudo_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
