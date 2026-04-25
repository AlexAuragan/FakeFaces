import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import kagglehub

## human-face
# kagglehub.dataset_download("ashwingupta3012/human-faces", output_dir="data/human-faces")

## celeb-a
# kagglehub.dataset_download("jessicali9530/celeba-dataset", output_dir="data/celeba")

## pre-resize celeba to (112, 128) — (H, W) matching INPUT_SHAPE
TARGET_SIZE = (128, 112)  # PIL uses (W, H)
SRC_DIR = Path("data/celeba/img_align_celeba")
DST_DIR = Path("data/celeba/img_align_celeba_resized")

images = list(SRC_DIR.rglob("*.jpg")) + list(SRC_DIR.rglob("*.png"))
print(f"Resizing {len(images)} images to {TARGET_SIZE[1]}x{TARGET_SIZE[0]}...")

for src in tqdm(images):
    dst = DST_DIR / src.relative_to(SRC_DIR)
    if dst.exists():
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).resize(TARGET_SIZE, Image.LANCZOS).save(dst)

print(f"Done. Resized images saved to {DST_DIR}")
