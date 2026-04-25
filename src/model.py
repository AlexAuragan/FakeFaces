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
class Discriminator(torch.nn.Module): # 1.58M parameters
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
            ), # 3 * 64 * 3 * 3 + 64 = 1,792
            torch.nn.LeakyReLU(0.2),
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            ), # 64 * 128 * 3 * 3 + 128 = 73,856
            torch.nn.LeakyReLU(0.2),
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
            ), # 128 * 256 * 3 * 3 + 256 = 295,168
            torch.nn.LeakyReLU(0.2),
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
            ), # 256 * 512 * 3 * 3 + 512 = 1,180,160
            torch.nn.LeakyReLU(0.2),
            torch.nn.Flatten(),
        )
        # 4 stride-2 convs on 112x128: 112->56->28->14->7, 128->64->32->16->8
        # flat_size = 512 * 7 * 8 = 28,672
        dummy = torch.zeros(1, *INPUT_SHAPE)
        flat_size = self.features(dummy).shape[1]
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(flat_size, 1), # 28,672 * 1 + 1 = 28,673
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class Generator(torch.nn.Module): # 5.69M parameters
    def __init__(self, latent_size: int):
        super().__init__()
        n_nodes = 512 * (112 // (2**4)) * (128 // (2**4))  # 512 * 7 * 8 = 28,672
        self.features = torch.nn.Sequential(
            torch.nn.Linear(latent_size, n_nodes),              # latent_size * 28,672 + 28,672
                                                                # e.g. latent=100: 2,895,872
            torch.nn.LeakyReLU(0.2),
            Reshape(512, 7, 8),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 512 * 256 * 4 * 4 + 256 = 2,097,408
            torch.nn.BatchNorm2d(256),                          # 256 * 2 = 512
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 256 * 128 * 4 * 4 + 128 = 524,416
            torch.nn.BatchNorm2d(128),                          # 128 * 2 = 256
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 128 * 64 * 4 * 4 + 64 = 131,136
            torch.nn.BatchNorm2d(64),                           # 64 * 2 = 128
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 64 * 32 * 4 * 4 + 32 = 32,800
            torch.nn.BatchNorm2d(32),                           # 32 * 2 = 64
            torch.nn.LeakyReLU(0.2),
        )
        self.generator = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 3, kernel_size=7, padding=3),               # 32 * 3 * 7 * 7 + 3 = 4,707
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.generator(self.features(x))
