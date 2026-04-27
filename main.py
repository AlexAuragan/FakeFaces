import os
import torch
import wandb
from config import LATENT_POINTS
from src.core import train, dl_gif

os.makedirs("models", exist_ok=True)

model_config = {
    "latent_dim": 100,
    "batch_size": 64,
    "gen_struc": 0,
    "disc_struc": 0,
}

training_config = {
    "starting_epoch": 0,
    "number_epochs": 100,
    "learning_rate_disc": 0.0002,
    "learning_rate_gan": 0.0002,
    "latent_points": torch.tensor(LATENT_POINTS, dtype=torch.float32),
    "model_directory": "models/run_8",
    "sanity_check": False,
    "verbose": 1,
    "learning_rate_weight_strength": 0.5,
    "r1_batch_mod": 8,
}

if __name__ == "__main__":
    from datetime import datetime
    wandb.init(project="fakeface", config={**model_config, **training_config},
               # mode="disabled"
               )
    train(model_config, training_config)
    wandb.finish()
