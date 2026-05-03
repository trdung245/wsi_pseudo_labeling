"""
Extract attention scores and patch images for new slides of any label.

Usage
-----
  python extract_new_slides.py --label atypical --slides BRACS_001 BRACS_002
  python extract_new_slides.py --label malignant --slides BRACS_999 --patch-id-start 300000
  python extract_new_slides.py --label benign --slides BRACS_3277 BRACS_3279 BRACS_3310 --patch-id-start 241493

All paths default to the conventional layout relative to this script's directory.
Override any of them with the corresponding flag.
"""
import argparse
import os
import torch
import torch.nn.functional as F
import h5py
import pandas as pd
import openslide
from tqdm import tqdm
from types import SimpleNamespace

from utils.eval_utils import initiate_model

BASE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Extract attention scores and patches for new slides.")
    parser.add_argument("--label", required=True,
                        help="Label for all slides (e.g. benign, atypical, malignant, non-cancer).")
    parser.add_argument("--slides", required=True, nargs="+",
                        help="Slide IDs to process (without .svs extension).")
    parser.add_argument("--patch-id-start", type=int, default=0,
                        help="Starting patch_id counter (default: 0).")
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint path. Default: results/bracs_subtype_clam_s1/s_1_checkpoint.pt")
    parser.add_argument("--heatmap-dir", default=None,
                        help="Directory containing per-slide h5 subdirs. "
                             "Default: heatmaps/heatmap_raw_results/HEATMAP_OUTPUT/{label}")
    parser.add_argument("--slide-dir", default=None,
                        help="Directory containing .svs files. Default: slide_data/")
    parser.add_argument("--output-csv", default=None,
                        help="Output CSV path. Default: attention_scores_new_{label}.csv")
    parser.add_argument("--patch-save-dir", default=None,
                        help="Directory to save extracted PNG patches. Default: patch_dataset/{label}/")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="Patch size in pixels (default: 256).")
    # Model config
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--model-type", default="clam_sb")
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--drop-out", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    label = args.label

    # Resolve paths
    heatmap_dir   = args.heatmap_dir   or os.path.join(BASE, "heatmaps", "heatmap_raw_results", "HEATMAP_OUTPUT", label)
    ckpt_path     = args.ckpt          or os.path.join(BASE, "results", "bracs_subtype_clam_s1", "s_1_checkpoint.pt")
    slide_dir     = args.slide_dir     or os.path.join(BASE, "slide_data")
    output_csv    = args.output_csv    or os.path.join(BASE, f"attention_scores_new_{label}.csv")
    patch_save_dir = args.patch_save_dir or os.path.join(BASE, "patch_dataset", label)

    os.makedirs(patch_save_dir, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_args = SimpleNamespace(
        n_classes=args.n_classes,
        model_type=args.model_type,
        model_size=args.model_size,
        drop_out=args.drop_out,
        embed_dim=args.embed_dim,
    )

    model = initiate_model(model_args, ckpt_path)
    model = model.to(device)
    model.eval()

    all_rows = []
    patch_id = args.patch_id_start

    for slide_id in tqdm(args.slides, desc="Processing slides"):
        h5_path = os.path.join(heatmap_dir, slide_id, f"{slide_id}.h5")
        if not os.path.exists(h5_path):
            print(f"[SKIP] h5 not found: {h5_path}")
            continue

        with h5py.File(h5_path, "r") as f:
            features = torch.tensor(f["features"][:]).float().to(device)
            coords   = f["coords"][:]

        with torch.no_grad():
            _, _, _, A_raw, _ = model(features)
            A = F.softmax(A_raw, dim=1).squeeze(0).cpu().numpy()

        svs_path = os.path.join(slide_dir, slide_id + ".svs")
        try:
            slide = openslide.OpenSlide(svs_path)
            slide_open = True
        except Exception as e:
            print(f"[WARN] Cannot open SVS {svs_path}: {e}")
            slide_open = False

        for i in range(len(A)):
            x, y = int(coords[i][0]), int(coords[i][1])
            all_rows.append({
                "slide_id":  slide_id,
                "patch_id":  patch_id,
                "x":         x,
                "y":         y,
                "label":     label,
                "attention": float(A[i]),
            })

            if slide_open:
                try:
                    patch = slide.read_region((x, y), level=0, size=(args.patch_size, args.patch_size)).convert("RGB")
                    patch.save(os.path.join(patch_save_dir, f"{slide_id}_{x}_{y}.png"))
                except Exception as e:
                    print(f"[WARN] Patch error at {slide_id} ({x},{y}): {e}")

            patch_id += 1

        if slide_open:
            slide.close()

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"\nDone! Saved {len(df)} rows to {output_csv}")
    print(f"patch_id range: {args.patch_id_start} - {patch_id - 1}")
    print(f"Patch images saved to: {patch_save_dir}/")


if __name__ == "__main__":
    main()
