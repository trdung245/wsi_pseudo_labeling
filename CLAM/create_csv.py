import os
import torch
import torch.nn.functional as F
import h5py
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace

from utils.eval_utils import initiate_model

# ========================
# CONFIG
# ========================
ROOT_DIR = "heatmaps/heatmap_raw_results/HEATMAP_OUTPUT"
CKPT_PATH = "results/bracs_subtype_clam_s1/s_1_checkpoint.pt"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

MODEL_ARGS = SimpleNamespace(
    n_classes=3,
    model_type="clam_sb",
    model_size="small",
    drop_out=0.,
    embed_dim=1024
)

# ========================
# LOAD MODEL
# ========================
model = initiate_model(MODEL_ARGS, CKPT_PATH)
model = model.to(DEVICE)
model.eval()

# ========================
# PROCESS ALL SLIDES
# ========================
all_rows = []

for label in ["benign", "atypical", "malignant"]:
    label_dir = os.path.join(ROOT_DIR, label)

    for slide_id in tqdm(os.listdir(label_dir)):
        slide_path = os.path.join(label_dir, slide_id)

        if not os.path.isdir(slide_path):
            continue

        # find the correct h5 file (features)
        h5_path = os.path.join(slide_path, f"{slide_id}.h5")

        if not os.path.exists(h5_path):
            continue

        # ========================
        # LOAD FEATURES
        # ========================
        with h5py.File(h5_path, "r") as f:
            features = torch.tensor(f["features"][:]).float()
            coords = f["coords"][:]

        # move to device
        features = features.to(DEVICE)

        # ========================
        # COMPUTE ATTENTION
        # ========================
        with torch.no_grad():
            logits, Y_prob, Y_hat, A_raw, _ = model(features)

            A = F.softmax(A_raw, dim=1)   # normalize across patches

            A = A.squeeze(0).cpu().numpy()

        # ========================
        # STORE RESULTS
        # ========================
        for i in range(len(A)):
            all_rows.append({
                "slide_id": slide_id,
                "patch_id": i,
                "x": int(coords[i][0]),
                "y": int(coords[i][1]),
                "label": label,
                "attention": float(A[i])
            })

# ========================
# SAVE CSV
# ========================
df = pd.DataFrame(all_rows)

df.to_csv("attention_scores_all_slides.csv", index=False)

print("✅ Done! Saved to attention_scores_all_slides.csv")

print(features.shape)   # (N, 1024)
print(A.shape)     # (N, 3)
print(A.sum())