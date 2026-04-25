import os
import torch
import wandb
from config import LATENT_POINTS
from src.core import train

"""
● Other things to try:

  - Before a run, check temps and watch for spikes:
  watch -n1 sensors   # or: watch -n1 cat /sys/class/hwmon/hwmon*/temp1_input
  - Add HSA_ENABLE_SDMA=0 — disables a DMA path that commonly causes ROCm hangs on RDNA3:
  os.environ["HSA_ENABLE_SDMA"] = "0"   # add to config.py
  - Reduce num_workers in core.py:26 from 4 to 2 — less memory bus pressure alongside active GPU workloads.
  - Check the actual crash log after a crash:
  journalctl -b -1 | grep -iE "amdgpu|reset|hang|killed"
  - If you see amdgpu: GPU reset, it's driver recovery failing. If you see nothing, it's a hard power event (thermal or PSU).

  The fix that will most likely help is the ROCR_VISIBLE_DEVICES=0 correction and HSA_ENABLE_SDMA=0.

"""
os.makedirs("models", exist_ok=True)

model_config = {
    "latent_dim": 100,
    "batch_size": 64,
    "gen_struc": 0,
    "disc_struc": 0,
}

training_config = {
    "starting_epoch": 0,
    "number_epochs": 30,
    "learning_rate_disc": 0.0002,
    "learning_rate_gan": 0.0002,
    "latent_points": torch.tensor(LATENT_POINTS, dtype=torch.float32),
    "model_directory": "models/run_4",
    "sanity_check": False,
    "verbose": 1,
    "learning_rate_weight_strength": 0.5,
}

if __name__ == "__main__":
    from datetime import datetime
    print(datetime.now())
    wandb.init(project="fakeface", config={**model_config, **training_config},
               # mode="disabled"
               )
    train(model_config, training_config)
    wandb.finish()
