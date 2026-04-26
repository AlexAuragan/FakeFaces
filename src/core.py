import math
import re
import os
from time import time

import imageio
import torch
import wandb
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from config import DATA_DIR, LATENT_POINTS
from src.model import Generator, Discriminator


def make_dataset(batch_size = 64) -> DataLoader:
    transform = transforms.Compose([
        # transforms.Resize((112, 128)), # No need to resize if we use the already resized version
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalize colors to [-1, 1]
    ])

    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)

def custom_lr_scheduler(epoch, starting_lr) -> float:
  if epoch < 10 :
    return starting_lr * epoch + starting_lr
  if epoch < 30 :
    return starting_lr * 10
  return starting_lr * 10 * ( 1 - ((epoch-30)/19 ))

def generate_latent_points(latent_dim: int, n_samples: int) -> Tensor:
    return torch.normal(torch.zeros([n_samples, latent_dim]), 1).cuda()

def generate_fake_samples(generator, latent_dim, n_samples) -> Tensor:
    x_input = generate_latent_points(latent_dim, n_samples)
    with torch.no_grad():
        samples = generator(x_input)
    return samples

def make_model(config):
    torch.backends.cudnn.benchmark = True

    g_model = Generator(latent_size=config["latent_dim"]).cuda()
    d_model = Discriminator().cuda()

    g_optimizer = torch.optim.Adam(g_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(d_model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    g_scaler = torch.amp.GradScaler("cuda")
    d_scaler = torch.amp.GradScaler("cuda")

    return g_model, d_model, g_optimizer, d_optimizer, g_scaler, d_scaler

def save_gan(g_model, d_model, g_optimizer, d_optimizer, g_scaler, d_scaler, model_directory, config, epoch):
    torch.save({
        "generator": g_model.state_dict(),
        "discriminator": d_model.state_dict(),
        "g_optimizer": g_optimizer.state_dict(),
        "d_optimizer": d_optimizer.state_dict(),
        "g_scaler": g_scaler.state_dict(),
        "d_scaler": d_scaler.state_dict(),
        "config": config,
        "epoch": epoch,
    }, f"{model_directory}/checkpoint.pt")


def load_gan(model_directory, config):
    g_model, d_model, g_optimizer, d_optimizer, g_scaler, d_scaler = make_model(config)
    checkpoint = torch.load(f"{model_directory}/checkpoint.pt")
    g_model.load_state_dict(checkpoint["generator"])
    d_model.load_state_dict(checkpoint["discriminator"])
    g_optimizer.load_state_dict(checkpoint["g_optimizer"])
    d_optimizer.load_state_dict(checkpoint["d_optimizer"])
    if "g_scaler" in checkpoint:
        g_scaler.load_state_dict(checkpoint["g_scaler"])
    if "d_scaler" in checkpoint:
        d_scaler.load_state_dict(checkpoint["d_scaler"])
    epoch = checkpoint.get("epoch", 0)
    return g_model, d_model, g_optimizer, d_optimizer, g_scaler, d_scaler, epoch

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def _add_progress_bar(frames, bar_height=4, color=(148, 0, 211)):
    import numpy as np
    result = []
    n = len(frames)
    for idx, frame in enumerate(frames):
        frame = np.array(frame)
        h, w = frame.shape[:2]
        bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
        filled = int((idx + 1) / n * w)
        bar[:, :filled] = color
        result.append(np.vstack([frame, bar]))
    return result

def dl_gif(model_directory, fps=3):
    samples_dir = os.path.join(model_directory, "samples")
    if not os.path.exists(samples_dir):
        raise FileNotFoundError(f"No samples directory found at {samples_dir}")

    for i in range(1, 6):
        pattern = re.compile(rf'^sample_{i}_epoch_\d+\.png$')
        filenames = sorted(
            [f for f in os.listdir(samples_dir) if pattern.match(f)],
            key=natural_keys,
        )
        if not filenames:
            continue
        frames = [imageio.imread(os.path.join(samples_dir, f)) for f in filenames]
        frames = _add_progress_bar(frames)
        imageio.mimsave(os.path.join(model_directory, f"evolution_{i}.gif"), frames, fps=fps, loop=0)
        print(f"Saved evolution_{i}.gif ({len(frames)} frames)")



def compute_lr_scale(d_loss_real, d_loss_fake, g_loss, base_lr_d, base_lr_g, strength=0.5):
    LN2 = math.log(2)  # ~0.693, equilibrium loss value

    # D is winning when EITHER term is too low
    d_best = min(d_loss_real, d_loss_fake)

    d_advantage = LN2 - d_best  # large when D is crushing fakes
    # G is winning when its loss is low
    g_advantage = LN2 - g_loss

    # scale lr down for whoever is ahead, clamp to [0.1, 1.0]
    d_scale = (1.0 - strength * max(0, d_advantage) / LN2)
    g_scale = (1.0 - strength * max(0, g_advantage) / LN2)

    d_scale = max(0.1, min(1.0, d_scale))
    g_scale = max(0.1, min(1.0, g_scale))

    return base_lr_d * d_scale, base_lr_g * g_scale

def train_epoch(g_model, d_model, g_optimizer, d_optimizer, g_scaler, d_scaler, dataset, model_config, training_config, num_epoch, global_step):
    latent_dim = model_config["latent_dim"]
    criterion = torch.nn.BCEWithLogitsLoss()

    g_model.train()
    d_model.train()

    ds_iterator = enumerate(tqdm(dataset)) if training_config["verbose"] >= 1 else enumerate(dataset)
    if training_config["sanity_check"]:
        ds_iterator = ((i, x) for i, x in ds_iterator if i < 10)

    for j, (batch, _) in ds_iterator:
        batch = batch.cuda()
        real_labels = torch.ones(len(batch), 1).cuda() - 0.15 * torch.rand(len(batch), 1, device="cuda")
        fake_labels = torch.zeros(len(batch), 1).cuda()

        # --- discriminator ---
        batch.requires_grad_(True)
        X_fake = generate_fake_samples(g_model, latent_dim, len(batch))
        d_optimizer.zero_grad()
        d_real_out = d_model(batch)
        d_loss_real = criterion(d_real_out, real_labels)
        d_loss_fake = criterion(d_model(X_fake), fake_labels)

        # R1 gradient penalty every r1_batch_mod steps
        if global_step % training_config["r1_batch_mod"] == 0:
            grad = torch.autograd.grad(
                d_real_out.sum(), batch, create_graph=True
            )[0]
            r1 = grad.pow(2).flatten(1).sum(1).mean()
            d_loss = d_loss_real + d_loss_fake + (10.0/2) * 16 * r1
        else:
            d_loss = d_loss_real + d_loss_fake

        d_scaler.scale(d_loss).backward()
        d_scaler.step(d_optimizer)
        d_scaler.update()

        # --- generator ---
        z = generate_latent_points(latent_dim, len(batch))
        g_optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):
            g_loss = criterion(d_model(g_model(z)), real_labels)
        g_scaler.scale(g_loss).backward()
        g_scaler.step(g_optimizer)
        g_scaler.update()

        global_step += 1

        wandb.log({
            "batch/d_loss_real": d_loss_real.item(),
            "batch/d_loss_fake": d_loss_fake.item(),
            "batch/g_loss": g_loss.item(),
            "batch/lr_disc": d_optimizer.param_groups[0]["lr"],
            "batch/lr_gen": g_optimizer.param_groups[0]["lr"],
        }, step=global_step)
        if training_config["verbose"] == 2:
            print(f"Epoch>{num_epoch+1}, batch={j}, d_real={d_loss_real.item():.4f}, d_fake={d_loss_fake.item():.4f}, g={g_loss.item():.4f}")

    wandb.log({
        "d_loss_real": d_loss_real.item(),
        "d_loss_fake": d_loss_fake.item(),
        "g_loss": g_loss.item(),
    })

    if training_config["verbose"] == 1:
        print(f"Epoch>{num_epoch+1}, d_real={d_loss_real.item():.4f}, d_fake={d_loss_fake.item():.4f}, g={g_loss.item():.4f}")
    return d_loss_real.item(), d_loss_fake.item(), g_loss.item(), global_step

def model_save_directory(base_dir, epoch):
    path = os.path.join(base_dir, f"epoch_{epoch}")
    os.makedirs(path, exist_ok=True)
    return path

def train(model_config, training_config):
    latent_points = training_config["latent_points"].cuda()
    dataset = make_dataset(model_config["batch_size"])

    if training_config["starting_epoch"] == 0:
        g_model, d_model, g_optimizer, d_optimizer, g_scaler, d_scaler = make_model(model_config)
    else:
        model_dir = model_save_directory(training_config["model_directory"], training_config["starting_epoch"])
        g_model, d_model, g_optimizer, d_optimizer, g_scaler, d_scaler, _ = load_gan(model_dir, model_config)

    avg_time = 0
    global_step = 0
    for epoch in range(training_config["starting_epoch"], training_config["starting_epoch"] + training_config["number_epochs"]):
        starting_time = time()

        d_loss_real, d_loss_fake, g_loss, global_step = train_epoch(g_model, d_model, g_optimizer, d_optimizer, g_scaler, d_scaler, dataset, model_config, training_config, epoch, global_step)

        new_lr_d, new_lr_g = compute_lr_scale(
            d_loss_real, d_loss_fake, g_loss,
            training_config["learning_rate_disc"],
            training_config["learning_rate_gan"],
            training_config["learning_rate_weight_strength"]
        )
        for pg in d_optimizer.param_groups:
            pg["lr"] = new_lr_d
        for pg in g_optimizer.param_groups:
            pg["lr"] = new_lr_g

        delta_time = time() - starting_time
        avg_time += delta_time

        # sample images for wandb + GIF
        g_model.eval()
        with torch.no_grad():
            X = g_model(latent_points).cpu()
        X = ((X + 1) / 2 * 255).clamp(0, 255).byte()
        # X shape: (N, 3, H, W) → wandb expects (H, W, 3)
        samples_dir = os.path.join(training_config["model_directory"], "samples")
        os.makedirs(samples_dir, exist_ok=True)
        for i in range(min(5, len(latent_points))):
            img = X[i].permute(1, 2, 0).numpy()
            imageio.imwrite(os.path.join(samples_dir, f"sample_{i+1}_epoch_{epoch+1:04d}.png"), img)
        samples = {f"samples_{i+1}": wandb.Image(X[i].permute(1, 2, 0).numpy(), caption=f"sample {i+1}, epoch {epoch+1}") for i in range(len(LATENT_POINTS))}
        wandb.log({
            **samples,
            "duration": delta_time,
            "epoch": epoch + 1,
            "learning_rate_discriminator": training_config["learning_rate_disc"],
            "learning_rate_gan": training_config["learning_rate_gan"],
            "lr_disc_effective": new_lr_d,
            "lr_gen_effective": new_lr_g,
        }, step=global_step)

        save_gan(g_model, d_model, g_optimizer, d_optimizer, g_scaler, d_scaler,
                 model_save_directory(training_config["model_directory"], epoch + 1),
                 model_config, epoch + 1)

    wandb.summary["avg_time"] = avg_time / training_config["number_epochs"]
