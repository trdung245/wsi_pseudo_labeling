"""
Promote pseudo_label → label in merged dataset CSVs.

pseudo_label now stores the final class name directly
("benign", "atypical", "malignant", "non-cancer"), so this
script simply overwrites label with pseudo_label and drops the column.

Usage
-----
  python cleanup.py                         # default CSV list
  python cleanup.py --csvs my_dataset.csv   # custom files
"""
import argparse
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CSVS = [
    "dataset_balanced.csv",
    "dataset_pos_heavy.csv",
    "dataset_neg_heavy.csv",
]


def parse_args():
    p = argparse.ArgumentParser(description="Promote pseudo_label to label in dataset CSVs.")
    p.add_argument("--csvs", nargs="+", default=DEFAULT_CSVS,
                   help="CSV files to clean up (searched relative to cwd).")
    p.add_argument("--suffix", default="new_",
                   help="Prefix added to output filename (default: 'new_').")
    return p.parse_args()


def main():
    args = parse_args()

    for csv_name in args.csvs:
        if not os.path.exists(csv_name):
            print(f"[SKIP] Not found: {csv_name}")
            continue

        df = pd.read_csv(csv_name)

        if "pseudo_label" in df.columns:
            df["label"] = df["pseudo_label"]
            df = df.drop(columns=["pseudo_label"])
            print(f"[OK]   {csv_name} — promoted pseudo_label → label")
        else:
            print(f"[OK]   {csv_name} — no pseudo_label column, left unchanged")

        out_name = args.suffix + os.path.basename(csv_name)
        df.to_csv(out_name, index=False)
        print(f"       → {out_name}")


if __name__ == "__main__":
    main()
