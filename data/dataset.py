"""
Dataset classes for WSI patch data.

CSV schema: slide_id, patch_id, x, y, label, attention, attention_pct

Image path convention (organized_patches/):
    organized_patches/{label}/{slide_id}_{x}_{y}.png

Run organize_dataset.py once before training to create this directory from the
three raw CSVs.  It creates symlinks (no disk duplication) and assigns each
patch the CLAM-derived attention label, not the original BRACS slide label.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, PngImagePlugin
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Raise PIL's iCCP decompression limit once at import time.
# Some BRACS PNG tiles embed large ICC colour profiles that exceed the default.
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)   # 100 MB

# Canonical label order – index is the integer class id
LABEL_TO_IDX: dict[str, int] = {
    "non-cancer": 0,
    "benign":     1,
    "atypical":   2,
    "malignant":  3,
}
IDX_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_IDX.items()}
NUM_CLASSES = len(LABEL_TO_IDX)


class PatchDataset(Dataset):
    """Single-CSV patch dataset backed by the organized_patches/ directory.

    Parameters
    ----------
    csv_path:
        Path to one of the three CSV files.
    patch_dir:
        Root of the organized 4-class directory (organized_patches/).
    transform:
        Optional torchvision transform applied to each loaded PIL image.
    split:
        'train', 'val', or 'test'.  Patches are assigned to splits by a
        deterministic hash of slide_id (slide-level split, no leakage).
    val_split / test_split:
        Fraction of slides held out for validation / test.
    label_map:
        Optional dict mapping label string → integer class index.
        Defaults to the module-level LABEL_TO_IDX (4-class).
        Pass a custom map for e.g. 3-class training.
    """

    def __init__(
        self,
        csv_path: str | Path,
        patch_dir: str | Path,
        transform: Optional[Callable] = None,
        split: str = "train",
        val_split: float = 0.15,
        test_split: float = 0.10,
        label_map: Optional[dict[str, int]] = None,
    ) -> None:
        assert split in ("train", "val", "test")
        self.patch_dir = Path(patch_dir)
        self.transform = transform
        self._label_map = label_map if label_map is not None else LABEL_TO_IDX

        df = pd.read_csv(csv_path)
        df = self._assign_splits(df, val_split, test_split)
        df = df[df["split"] == split].reset_index(drop=True)

        self.records: list[dict] = []
        skipped = 0
        for _, row in df.iterrows():
            if row["label"] not in self._label_map:
                skipped += 1
                continue
            img_path = self._image_path(row["label"], row["slide_id"], row["x"], row["y"])
            if not img_path.exists():
                skipped += 1
                continue
            self.records.append({
                "img_path":  img_path,
                "label":     self._label_map[row["label"]],
                "attention": float(row["attention_pct"]),
                "slide_id":  row["slide_id"],
                "patch_id":  row["patch_id"],
            })

        if skipped:
            logger.warning(
                "%s [%s]: skipped %d/%d patches (image not in organized_patches/).",
                Path(csv_path).name, split, skipped, len(df),
            )
        logger.info(
            "%s [%s]: %d usable patches. Label dist: %s",
            Path(csv_path).name, split, len(self.records),
            self._label_counts(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _image_path(self, label: str, slide_id: str, x, y) -> Path:
        return self.patch_dir / label / f"{slide_id}_{x}_{y}.png"

    @staticmethod
    def _assign_splits(df: pd.DataFrame, val_split: float, test_split: float) -> pd.DataFrame:
        """Deterministic stratified slide-level train / val / test split.

        Slides are grouped by the label that is most common on that slide, then
        within each label-group they are shuffled and apportioned to test / val /
        train by the requested fractions.  This guarantees that every class
        present in the CSV is represented in every split (provided there are
        enough slides), regardless of random seed luck.
        """
        rng = np.random.default_rng(seed=42)

        # Map each slide to its dominant label
        slide_dominant = (
            df.groupby("slide_id")["label"]
            .agg(lambda s: s.value_counts().idxmax())
        )

        slide_to_split: dict[str, str] = {}
        for label_class, group in slide_dominant.groupby(slide_dominant):
            slides = sorted(group.index.tolist())
            rng.shuffle(slides)                     # deterministic shuffle
            n = len(slides)
            n_test = max(1, round(test_split * n))  # at least 1 slide per split
            n_val  = max(1, round(val_split  * n))  # if enough slides exist
            # If only 1 or 2 slides in a class, allocate to train to avoid
            # completely empty train for that class
            if n == 1:
                n_test, n_val = 0, 0
            elif n == 2:
                n_test, n_val = 0, 1
            for sid in slides[:n_test]:
                slide_to_split[sid] = "test"
            for sid in slides[n_test:n_test + n_val]:
                slide_to_split[sid] = "val"
            for sid in slides[n_test + n_val:]:
                slide_to_split[sid] = "train"

        df = df.copy()
        df["split"] = df["slide_id"].map(slide_to_split)
        return df

    def _label_counts(self) -> dict[str, int]:
        from collections import Counter
        idx_to_label = {v: k for k, v in self._label_map.items()}
        return dict(Counter(idx_to_label[r["label"]] for r in self.records))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        image = Image.open(rec["img_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image":     image,
            "label":     torch.tensor(rec["label"],     dtype=torch.long),
            "attention": torch.tensor(rec["attention"], dtype=torch.float32),
        }


class CombinedPatchDataset(Dataset):
    """Concatenation of patches from all three CSV files.

    Used during Phase 2 (stacking) to collect predictions from all VMoE
    models over a unified held-out split.
    """

    def __init__(
        self,
        csv_paths: dict[str, str | Path],
        patch_dir: str | Path,
        transform: Optional[Callable] = None,
        split: str = "val",
        val_split: float = 0.15,
        test_split: float = 0.10,
        label_map: Optional[dict[str, int]] = None,
    ) -> None:
        self.datasets = {
            name: PatchDataset(
                csv_path=path,
                patch_dir=patch_dir,
                transform=transform,
                split=split,
                val_split=val_split,
                test_split=test_split,
                label_map=label_map,
            )
            for name, path in csv_paths.items()
        }
        self._offsets: list[tuple[str, int]] = []
        for name, ds in self.datasets.items():
            self._offsets.extend((name, i) for i in range(len(ds)))

    @property
    def records(self) -> list[dict]:
        """Flat list of all records across constituent datasets (for class-weight computation)."""
        result = []
        for ds in self.datasets.values():
            result.extend(ds.records)
        return result

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> dict:
        name, local_idx = self._offsets[idx]
        sample = self.datasets[name][local_idx]
        sample["source"] = name
        return sample
