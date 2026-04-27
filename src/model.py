from config import INPUT_SHAPE
import torch

"""
# Norms layer
BatchNorm on the generator → smoother training, better gradient flow through upsampling layers
SpectralNorm on the discriminator → prevents it from becoming too powerful and producing exploding/vanishing gradients for the generator
BatchNorm normalizes the activations — it rescales the output of a layer to have zero mean and unit variance across the batch. This helps gradients flow and speeds up training. It's applied to the layer's output.
Spectral Norm constrains the weights themselves — specifically it divides the weight matrix by its largest singular value (the spectral norm), keeping it ≤ 1. This limits how much the discriminator's output can change in response to small changes in input, which is called Lipschitz continuity. GANs need this because an unconstrained discriminator can produce arbitrarily large gradients that destabilize the generator.

# Model size
"""

"""
# GAN weight sizing guide for CelebA

## Rule of thumb

```
params ≈ dataset_size × 20–50
```

200k images → **4M–10M params total** (G + D combined)

---

## Conv layer formula

```
Conv2d weights    = C_in × C_out × K × K  +  C_out (bias)
ConvTranspose2d   = same formula, reversed direction
Linear weights    = C_in × C_out          +  C_out (bias)
```

**Example:**
```
Conv2d(64, 128, k=3)   →  64 × 128 × 3 × 3  +  128  =  73,856 params
Linear(100, 28672)     →  100 × 28672        +  28672 =  2,895,872 params
```

---

## Generator (latent_size=100)

| Layer | Formula | Params |
|---|---|---|
| `Linear(100, 512×7×8)` | 100 × 28672 + 28672 | 2,895,872 |
| `ConvTranspose2d(512→256, k=4)` | 512 × 256 × 4 × 4 | 2,097,152 |
| `ConvTranspose2d(256→128, k=4)` | 256 × 128 × 4 × 4 | 524,288 |
| `ConvTranspose2d(128→64, k=4)` | 128 × 64 × 4 × 4 | 131,072 |
| `ConvTranspose2d(64→32, k=4)` | 64 × 32 × 4 × 4 | 32,768 |
| `ConvTranspose2d(32→3, k=7)` | 32 × 3 × 7 × 7 | 4,704 |
| `BatchNorm × 4` | (256+128+64+32) × 2 | 960 |
| **Generator total** | | **~5.69M** |

---

## Discriminator (input 112×128)

| Layer | Formula | Params |
|---|---|---|
| `Conv2d(3→64, k=3)` | 3 × 64 × 3 × 3 | 1,728 |
| `Conv2d(64→128, k=3)` | 64 × 128 × 3 × 3 | 73,728 |
| `Conv2d(128→256, k=3)` | 128 × 256 × 3 × 3 | 294,912 |
| `Conv2d(256→512, k=3)` | 256 × 512 × 3 × 3 | 1,179,648 |
| `Linear(512×7×8 → 1)` | 28672 × 1 + 1 | 28,673 |
| **Discriminator total** | | **~1.58M** |

---

"""
class Discriminator(torch.nn.Module): # 2.38M parameters
    @staticmethod
    def down_block(dims_in, dims_out): # 9 * dims_out * (dims_in + dims_out) + 2 * dims_out
        return torch.nn.Sequential(
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(dims_in, dims_out, kernel_size=3, stride=2, padding=1) # dims_in * dims_out * 9 + dims_out
            ),
            torch.nn.LeakyReLU(0.2),
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(dims_out, dims_out, kernel_size=3, padding=1) # dims_out^2 * 9 + dims_out
            ),
            torch.nn.LeakyReLU(0.2),
        )

    def __init__(self):
        super().__init__()
        self.input = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)),
            torch.nn.LeakyReLU(0.2),
        )
        self.features = torch.nn.Sequential(
            self.down_block(64, 64),     # 74k
            self.down_block(64, 128),    # 221k
            self.down_block(128, 256),   # 885k
            self.down_block(256, 256),   # 1.18M
        )

        flat_size = 256 * 7 * 8  # 14336
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(0.4),
            torch.nn.utils.spectral_norm(
                torch.nn.Linear(flat_size, 1),  # 14337
            )
        )

    def forward(self, x):
        return self.classifier(self.features(self.input(x)))


class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class Generator(torch.nn.Module): # 4.3M parameters
    @staticmethod
    def up_block(dims_in, dims_out): # 9 * c_out * (c_in + c_out) + 6 * c_out
        return torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv2d(dims_in, dims_out, kernel_size=3, padding=1), # 9 * dims_in * dims_out + dims_out
            torch.nn.BatchNorm2d(dims_out), # 2 * c_out
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(dims_out, dims_out, kernel_size=3, padding=1), # 9 * dims_out^2 + dims_out
            torch.nn.BatchNorm2d(dims_out), # 2 * c_out
            torch.nn.LeakyReLU(0.2)
        )

    def __init__(self, latent_size: int): # 4.3M
        super().__init__()
        n_nodes = 256 * (112 // (2**4)) * (128 // (2**4))  # 256 * 7 * 8 = 28,672
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(latent_size, n_nodes),  # 1.43M
            Reshape(256, 7, 8),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2),
        )

        self.features = torch.nn.Sequential(
            self.up_block(256, 256), # 1.18M
            self.up_block(256, 256), # 1.18M
            self.up_block(256, 256), # 1.8M
            self.up_block(256, 128), # 444k
        )
        self.generator = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1), # 72k
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 3, kernel_size=1),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.generator(self.features(self.projection(x)))
