import os
import pandas as pd
import numpy as np
from PIL import Image
import openslide
from tqdm import tqdm

# ===== CONFIG =====
CSV_PATH = "attention_scores_all_slides.csv"
WSI_DIR = "slide_data"
SAVE_DIR = "patch_dataset"
PATCH_SIZE = 256

os.makedirs(SAVE_DIR, exist_ok=True)

# ===== LOAD CSV =====
df = pd.read_csv(CSV_PATH)

# ===== CACHE SLIDES =====
slide_cache = {}

def get_slide(slide_id):
    if slide_id not in slide_cache:
        slide_path = os.path.join(WSI_DIR, slide_id + ".svs")
        slide_cache[slide_id] = openslide.OpenSlide(slide_path)
    return slide_cache[slide_id]

# ===== MAIN LOOP =====
for idx, row in tqdm(df.iterrows(), total=len(df)):
    slide_id = row["slide_id"]
    x, y = int(row["x"]), int(row["y"])
    label = row["label"]

    try:
        slide = get_slide(slide_id)

        patch = slide.read_region(
            (x, y),
            level=0,
            size=(PATCH_SIZE, PATCH_SIZE)
        ).convert("RGB")

        # ===== SAVE PATH =====
        label_dir = os.path.join(SAVE_DIR, label)
        os.makedirs(label_dir, exist_ok=True)

        filename = f"{slide_id}_{x}_{y}.png"
        save_path = os.path.join(label_dir, filename)

        patch.save(save_path)

    except Exception as e:
        print(f"Error at {slide_id} ({x},{y}): {e}")

# ===== CLOSE SLIDES =====
for slide in slide_cache.values():
    slide.close()

print("Done extracting patches")