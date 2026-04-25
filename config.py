import torch
import numpy as np
print(f"{torch.cuda.is_available()=}")       # True
print(f"{torch.cuda.get_device_name(0)=}")   # AMD Radeon RX 7600

INPUT_SHAPE = (3, 112, 128)
DATE = "24-03-2026"
DATASET_SIZE = 202599
DATA_DIR = "data/celeba/img_align_celeba_resized"
LP_PATH = "models/latent_points_dim100"

# Generate latent points seeds to have a point of comparison across epochs.
# LATENT_POINTS = torch.normal(torch.zeros([20, 100]), 1)
# np.save(LP_PATH+".npy",LATENT_POINTS.numpy())
LATENT_POINTS = np.load(LP_PATH+".npy")