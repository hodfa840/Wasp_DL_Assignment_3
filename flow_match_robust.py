#!/usr/bin/env python3
"""
Rectified Flow (RF) for Fashion-MNIST - Robust Version
Coherent with DDPM pipeline and report structure.

- Training objective: learn velocity field v_theta(x_t, t) to match (x1 - x0)
  where x_t = (1 - t) * x0 + t * x1 and x0 ~ N(0, I), x1 ~ p_data
- Sampling: integrate dx/dt = v_theta(x, t) from t=0..1 (Euler integration)
- Logs: training loss, average grad norm, parameter change rate (coherent with DDPM)
- Evaluation: Compatible with DDPM feature extractor for FID/KID comparison
"""

import os
import time
import random
from typing import Tuple, List, Optional
import urllib.request
import gzip
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ---------------------------
# Reproducibility & Device
# ---------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Torch version: {torch.__version__}")

# ---------------------------
# Manual Fashion-MNIST Dataset (avoiding torchvision)
# ---------------------------
class FashionMNISTDataset(Dataset):
    """Manual Fashion-MNIST loader to avoid torchvision issues."""
    
    def __init__(self, root="data", train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        
        # Download and load data
        self.data, self.targets = self._load_data()
        
    def _load_data(self):
        """Download and load Fashion-MNIST data manually."""
        os.makedirs(self.root, exist_ok=True)
        
        if self.train:
            images_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
            labels_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
            images_file = os.path.join(self.root, "train-images-idx3-ubyte.gz")
            labels_file = os.path.join(self.root, "train-labels-idx1-ubyte.gz")
        else:
            images_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"
            labels_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"
            images_file = os.path.join(self.root, "t10k-images-idx3-ubyte.gz")
            labels_file = os.path.join(self.root, "t10k-labels-idx1-ubyte.gz")
        
        # Download if not exists
        if not os.path.exists(images_file):
            print(f"Downloading {images_url}")
            urllib.request.urlretrieve(images_url, images_file)
        if not os.path.exists(labels_file):
            print(f"Downloading {labels_url}")
            urllib.request.urlretrieve(labels_url, labels_file)
        
        # Load images
        with gzip.open(images_file, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        
        # Load labels
        with gzip.open(labels_file, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        return torch.from_numpy(images).float(), torch.from_numpy(labels).long()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        
        # Normalize to [0, 1] then transform to [-1, 1] (coherent with DDPM)
        img = img / 255.0
        img = img * 2.0 - 1.0
        
        # Add channel dimension
        img = img.unsqueeze(0)  # [1, 28, 28]
        
        if self.transform:
            img = self.transform(img)
        
        return img, target

# ---------------------------
# Data Loading
# ---------------------------
BATCH_SIZE = 128
IMG_CHW = (1, 28, 28)

print("Loading Fashion-MNIST data...")
train_set = FashionMNISTDataset(root="data", train=True)
test_set = FashionMNISTDataset(root="data", train=False)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print("Data loaded successfully!")
x0_, _ = train_set[0]
print(f"Sample pixel range: [{x0_.min().item():.3f}, {x0_.max().item():.3f}]")
print(f"Sample shape: {x0_.shape}")

# ---------------------------
# Utilities: Time Embeddings
# ---------------------------
def fourier_time_embedding(t: torch.Tensor, dim: int = 64, max_freq: float = 10.0) -> torch.Tensor:
    """
    t: shape [N] or [N, 1] in [0,1]
    returns: [N, 2*dim]
    """
    if t.ndim == 2 and t.shape[1] == 1:
        t = t.squeeze(1)
    device = t.device
    freqs = torch.linspace(1.0, max_freq, dim, device=device)  # [dim]
    angles = t[:, None] * freqs[None, :] * 2.0 * torch.pi      # [N, dim]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [N, 2*dim]

# ---------------------------
# Model: UNet-lite with FiLM time conditioning (coherent with DDPM UNet style)
# ---------------------------
class FiLM(nn.Module):
    """Simple FiLM layer: gamma,beta from time embedding -> affine modulate a feature map."""
    def __init__(self, emb_dim: int, num_channels: int):
        super().__init__()
        self.to_gamma = nn.Linear(emb_dim, num_channels)
        self.to_beta  = nn.Linear(emb_dim, num_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W], t_emb: [N, emb_dim]
        gamma = self.to_gamma(t_emb).unsqueeze(-1).unsqueeze(-1)  # [N, C, 1, 1]
        beta  = self.to_beta(t_emb).unsqueeze(-1).unsqueeze(-1)   # [N, C, 1, 1]
        return x * (1.0 + gamma) + beta

class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, emb_dim: int, use_norm: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(8, out_c) if use_norm else nn.Identity()
        self.norm2 = nn.GroupNorm(8, out_c) if use_norm else nn.Identity()
        self.film1 = FiLM(emb_dim, out_c)
        self.film2 = FiLM(emb_dim, out_c)

    def forward(self, x, t_emb):
        x = self.conv1(x); x = self.norm1(x); x = self.act(self.film1(x, t_emb))
        x = self.conv2(x); x = self.norm2(x); x = self.act(self.film2(x, t_emb))
        return x

class Down(nn.Module):
    def __init__(self, in_c: int, out_c: int, emb_dim: int):
        super().__init__()
        self.block = ConvBlock(in_c, out_c, emb_dim)
        self.down  = nn.Conv2d(out_c, out_c, 4, stride=2, padding=1)  # 2x downsample

    def forward(self, x, t_emb):
        x = self.block(x, t_emb)
        skip = x
        x = self.down(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, in_c: int, out_c: int, emb_dim: int, skip_c: int = None):
        super().__init__()
        if skip_c is None:
            skip_c = out_c
        self.up    = nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1)
        self.block = ConvBlock(out_c + skip_c, out_c, emb_dim)  # concat skip with correct channels

    def forward(self, x, skip, t_emb):
        x = self.up(x)  # [N, out_c, H, W]
        x = torch.cat([x, skip], dim=1)  # [N, out_c + skip_c, H, W]
        x = self.block(x, t_emb)  # [N, out_c, H, W]
        return x

class UNetRF(nn.Module):
    """
    Predicts velocity field v_theta(x_t, t) with input x_t in [-1,1], t in [0,1].
    Architecture similar to DDPM UNet for coherent comparison.
    """
    def __init__(self, in_channels: int = 1, base: int = 32, time_dim: int = 128):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(2 * (time_dim // 2), time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        # Encoder
        self.in_conv = nn.Conv2d(in_channels, base, 3, padding=1)
        self.down1 = Down(base, base*2, time_dim)    # 28 -> 14
        self.down2 = Down(base*2, base*4, time_dim)  # 14 -> 7

        # Bottleneck
        self.mid = ConvBlock(base*4, base*4, time_dim)

        # Decoder
        self.up1 = Up(base*4, base*2, time_dim, skip_c=base*4)  # 7 -> 14, skip from down2 has base*4 channels
        self.up2 = Up(base*2, base, time_dim, skip_c=base*2)    # 14 -> 28, skip from down1 has base*2 channels
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t: [N] or [N,1] in [0,1], embed via Fourier -> MLP
        t = t.view(-1, 1)
        t_emb = fourier_time_embedding(t, dim=self.time_dim // 2, max_freq=10.0)
        t_emb = self.time_mlp(t_emb)

        x = self.in_conv(x_t)
        x, s1 = self.down1(x, t_emb)   # s1: [N, base, 28, 28]
        x, s2 = self.down2(x, t_emb)   # s2: [N, 2*base, 14, 14]

        x = self.mid(x, t_emb)

        x = self.up1(x, s2, t_emb)
        x = self.up2(x, s1, t_emb)

        v = self.out_conv(x)  # velocity field, shape [N, 1, 28, 28]
        return v

# ---------------------------
# Rectified Flow Trainer
# ---------------------------
class RectifiedFlow(nn.Module):
    """
    Rectified Flow implementation coherent with DDPM pipeline:
    - Training objective for RF (MSE between predicted velocity and (x1 - x0))
    - Sampling via Euler integration of dx/dt = v_theta(x,t)
    - Same logging style as DDPM for comparison
    """
    def __init__(self, network: nn.Module, device: torch.device):
        super().__init__()
        self.net = network.to(device)
        self.device = device

    @staticmethod
    def _linear_interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t = (1 - t) x0 + t x1 (straight-line interpolation)
        while t.ndim < x0.ndim:
            t = t.unsqueeze(-1)
        return (1.0 - t) * x0 + t * x1

    def training_step(self, x1: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        RF training step:
        Sample x0 ~ N(0, I), t ~ U(0,1), build x_t, target v = x1 - x0
        """
        n = x1.size(0)
        x0 = torch.randn_like(x1)                         # base noise
        t  = torch.rand(n, device=x1.device)              # [0,1]
        x_t = self._linear_interpolate(x0, x1, t.view(n, 1, 1, 1))
        v_target = x1 - x0  # constant velocity field

        v_pred = self.net(x_t, t)                         # predict velocity
        loss = F.mse_loss(v_pred, v_target)

        return loss, t.mean().item()

    @torch.no_grad()
    def sample(self, n_samples: int, steps: int = 100, clamp_to_range: bool = True) -> torch.Tensor:
        """
        Euler integrate dx/dt = v_theta(x, t), t in [0,1]
        Returns images in approximately [-1,1] (coherent with DDPM).
        """
        self.eval()
        x = torch.randn(n_samples, *IMG_CHW, device=self.device)
        ts = torch.linspace(0.0, 1.0, steps+1, device=self.device)
        dt = ts[1] - ts[0]
        
        for i in range(steps):
            t_mid = (ts[i] + ts[i+1]) / 2.0  # midpoint time for better accuracy
            t_vec = torch.full((n_samples,), t_mid.item(), device=self.device)
            v = self.net(x, t_vec)
            x = x + dt * v
            
        if clamp_to_range:
            x = x.clamp(-1.0, 1.0)
        return x

# ---------------------------
# Training loop (coherent with DDPM logging style)
# ---------------------------
def train_rf(
    model: RectifiedFlow,
    train_loader: DataLoader,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    save_path: str = "weights_rf.pth",
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    train_losses: List[float] = []
    grad_norms: List[float] = []
    param_change_rates: List[float] = []

    prev_params: Optional[List[torch.Tensor]] = None
    best_loss = float("inf")

    print(f"\n🚀 Starting Rectified Flow training for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        batch_grad_norms = []

        for x1, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False):
            x1 = x1.to(device)

            optimizer.zero_grad(set_to_none=True)

            # RF training step
            loss, _ = model.training_step(x1)
            loss.backward()

            # gradient norm (L2) - coherent with DDPM logging
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    gnorm = p.grad.data.norm(2).item()
                    total_norm_sq += gnorm * gnorm
            batch_grad_norms.append((total_norm_sq + 1e-16) ** 0.5)

            optimizer.step()

            running_loss += loss.item() * x1.size(0) / len(train_loader.dataset)

        # epoch logs
        epoch_loss = running_loss
        avg_grad_norm = float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0
        train_losses.append(epoch_loss)
        grad_norms.append(avg_grad_norm)

        # parameter change rate (coherent with DDPM stability metrics)
        with torch.no_grad():
            current_params = [p.detach().clone() for p in model.parameters()]
            if prev_params is None:
                param_change_rates.append(0.0)
            else:
                sq = 0.0
                for p, q in zip(current_params, prev_params):
                    sq += torch.norm(p - q).item() ** 2
                param_change_rates.append(float(sq ** 0.5))
            prev_params = current_params

        # save best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"[Epoch {epoch}] New best loss: {best_loss:.6f}  → saved to {save_path}")

        # user-facing log (coherent with DDPM format)
        print(f"[Epoch {epoch}] Loss: {epoch_loss:.6f} | GradNorm: {avg_grad_norm:.4f} | ParamΔ: {param_change_rates[-1]:.4f}")

    return train_losses, grad_norms, param_change_rates

# ---------------------------
# Main: train, sample, evaluate (coherent with DDPM pipeline)
# ---------------------------
if __name__ == "__main__":
    # Hyperparameters
    NUM_EPOCHS = 200
    LR = 1e-3
    SAMPLE_DIR = "rf_generated_fmnist"
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    # Model & Optimizer
    net = UNetRF(in_channels=1, base=32, time_dim=128)
    rf = RectifiedFlow(net, DEVICE)
    opt = Adam(rf.parameters(), lr=LR)

    print(f"\n📊 Model Info:")
    print(f"Device: {DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in rf.parameters()):,}")
    print(f"Training samples: {len(train_set):,}")
    print(f"Batch size: {BATCH_SIZE}")

    # Train
    losses, gnorms, param_rates = train_rf(
        rf, train_loader, NUM_EPOCHS, opt, DEVICE, save_path="weights/rf_weights.pth"
    )

    # Reload best and sample
    print("\n🎨 Loading best model and generating samples...")
    best = RectifiedFlow(UNetRF(in_channels=1, base=32, time_dim=128), DEVICE)
    best.load_state_dict(torch.load("weights/rf_weights.pth", map_location=DEVICE))
    best.eval()

    # Generate and save samples (coherent with DDPM output format)
    with torch.no_grad():
        print("Generating 64 samples...")
        samples = best.sample(n_samples=64, steps=100)  # [64,1,28,28] in [-1,1]
        # map to [0,1] for saving as images
        samples_01 = (samples + 1.0) / 2.0
        
        # Save individual sample images
        for i in range(min(64, samples_01.size(0))):
            img_array = samples_01[i].squeeze().cpu().numpy()
            plt.figure(figsize=(3, 3))
            plt.imshow(img_array, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.savefig(os.path.join(SAMPLE_DIR, f"sample_{i:02d}.png"), bbox_inches='tight', dpi=100)
            plt.close()
        
        # Save 8x8 grid visualization (coherent with DDPM)
        plt.figure(figsize=(12, 12))
        for i in range(64):
            plt.subplot(8, 8, i+1)
            plt.imshow(samples_01[i].squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
        plt.suptitle('Rectified Flow Generated Fashion-MNIST Samples', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(SAMPLE_DIR, "grid_8x8.png"), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✅ Generated samples saved to {SAMPLE_DIR}/")

    # Plot training curves (coherent with DDPM plotting style)
    epochs_axis = list(range(1, len(losses) + 1))

    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs_axis, losses, 'b-', marker='o', markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("RF: Training Loss")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs_axis, gnorms, 'g-', marker='o', markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("L2 Grad Norm")
    plt.title("RF: Gradient Norm")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs_axis, param_rates, 'r-', marker='o', markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("L2 Param Change")
    plt.title("RF: Parameter Change Rate")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("rf_training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("📈 Training curves saved to rf_training_curves.png")
    
    # ================== EXTRA PLOTS TO MATCH DDPM FILENAMES ==================
    # Output folder (change to "figures/diffusion" if your LaTeX expects that path)
    FIG_DIR = "figures/flow"
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---- (Optional) tiny classifier for testing_cnn_on_gen_data.png ----
    class FeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.fc1 = nn.Linear(64*12*12, 128)
            self.fc2 = nn.Linear(128, 10)
        def forward(self, x, return_features=False):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.adaptive_avg_pool2d(x, (12, 12))
            x = torch.flatten(x, 1)
            feats = self.fc1(x)
            if return_features:
                return feats
            out = self.fc2(F.relu(feats))
            return out

    def get_or_train_feature_extractor(weights="feature_extractor_fashionmnist.pth", epochs=5, lr=1e-3):
        feat = FeatureExtractor().to(DEVICE)
        if os.path.exists(weights):
            print(f"📥 Loading existing feature extractor: {weights}")
            feat.load_state_dict(torch.load(weights, map_location=DEVICE))
            feat.eval(); return feat
        print("[Info] Training a quick feature extractor (no torchvision)...")
        crit = nn.CrossEntropyLoss(); opt = Adam(feat.parameters(), lr=lr)
        for ep in range(epochs):
            feat.train()
            for xb, yb in tqdm(train_loader, desc=f"FeatExt {ep+1}/{epochs}", leave=False):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                logits = feat(xb)
                loss = crit(logits, yb)
                loss.backward(); opt.step()
        torch.save(feat.state_dict(), weights)
        print(f"💾 Feature extractor saved: {weights}")
        feat.eval(); return feat

    # ---- 1) forward.png: linear interpolation x_t = (1-t)x0 + t x1 (training path) ----
    def save_forward_interpolations(loader, n_rows=2,
                                    t_list=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
                                    outpath=os.path.join(FIG_DIR, "forward.png")):
        print("📊 Creating forward interpolation plot...")
        best.eval()
        for xb, _ in loader:
            xb = xb.to(DEVICE); break
        n = min(n_rows, xb.size(0))
        x1 = xb[:n]
        x0 = torch.randn_like(x1)
        fig, axes = plt.subplots(n, len(t_list), figsize=(2.2*len(t_list), 2.2*n))
        if n == 1: axes = np.expand_dims(axes, 0)
        for i in range(n):
            for j, t in enumerate(t_list):
                t_tensor = torch.full((1,1,1,1), t, device=DEVICE)
                xt = (1.0 - t_tensor) * x0[i:i+1] + t_tensor * x1[i:i+1]
                img = ((xt[0,0].clamp(-1,1) + 1)/2).cpu().numpy()
                ax = axes[i, j]
                if i == 0: ax.set_title(f"t={t:.1f}")
                ax.imshow(img, cmap='gray', vmin=0, vmax=1); ax.axis("off")
        plt.tight_layout(); plt.savefig(outpath, dpi=150, bbox_inches='tight'); plt.close()
        print(f"✅ Forward plot saved: {outpath}")

    save_forward_interpolations(train_loader)

    # ---- 2) backward.png & backwork_trained.png: sampling snapshots (noise -> image) ----
    def save_sampling_trajectories(model, n_rows=2, steps=100,
                                   show_ts=(0.0, 0.1, 0.3, 0.5, 0.7, 1.0),
                                   out_back=os.path.join(FIG_DIR, "backward.png"),
                                   out_back_alt=os.path.join(FIG_DIR, "backwork_trained.png")):
        print("📊 Creating backward sampling trajectory plot...")
        model.eval()
        n_cols = len(show_ts)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2*n_cols, 2.2*n_rows))
        if n_rows == 1: axes = np.expand_dims(axes, 0)
        with torch.no_grad():
            x = torch.randn(n_rows, *IMG_CHW, device=DEVICE)
            ts = torch.linspace(0.0, 1.0, steps+1, device=DEVICE)
            dt = ts[1] - ts[0]
            capture_idx = [int(round(t * steps)) for t in show_ts]
            for i in range(steps+1):
                if i in capture_idx:
                    col = capture_idx.index(i)
                    for r in range(n_rows):
                        img = ((x[r,0].clamp(-1,1) + 1)/2).cpu().numpy()
                        ax = axes[r, col]
                        if r == 0: ax.set_title(f"t={show_ts[col]:.1f}")
                        ax.imshow(img, cmap='gray', vmin=0, vmax=1); ax.axis("off")
                if i < steps:
                    t_mid = (ts[i] + ts[i+1]) / 2.0
                    v = best.net(x, torch.full((n_rows,), t_mid.item(), device=DEVICE))
                    x = x + dt * v
        plt.tight_layout(); plt.savefig(out_back, dpi=150, bbox_inches='tight'); plt.close()
        import shutil; shutil.copyfile(out_back, out_back_alt)  # legacy filename for your LaTeX
        print(f"✅ Backward plots saved: {out_back} & {out_back_alt}")

    save_sampling_trajectories(best)

    # ---- 3) fake_vs_real.png: generated (top) vs real (bottom) ----
    def save_fake_vs_real(model, loader, n_show=10,
                          outpath=os.path.join(FIG_DIR, "fake_vs_real.png")):
        print("📊 Creating fake vs real comparison plot...")
        model.eval()
        with torch.no_grad():
            fake = model.sample(n_samples=n_show, steps=100).clamp(-1,1)
        for xb, _ in loader:
            real = xb[:n_show].to(DEVICE).clamp(-1,1); break
        fig, axes = plt.subplots(2, n_show, figsize=(2.0*n_show, 4))
        for i in range(n_show):
            axes[0,i].imshow(((fake[i,0]+1)/2).cpu().numpy(), cmap='gray', vmin=0, vmax=1); axes[0,i].axis('off')
            axes[1,i].imshow(((real[i,0]+1)/2).cpu().numpy(), cmap='gray', vmin=0, vmax=1); axes[1,i].axis('off')
            if i == 0: axes[0,i].set_title("Fake", fontweight='bold'); axes[1,i].set_title("Real", fontweight='bold')
        plt.tight_layout(); plt.savefig(outpath, dpi=150, bbox_inches='tight'); plt.close()
        print(f"✅ Fake vs real plot saved: {outpath}")

    save_fake_vs_real(best, test_loader)

    # ---- 4) testing_cnn_on_gen_data.png: classifier predictions on generated images ----
    def save_classifier_on_gen(model, n_show=10,
                               outpath=os.path.join(FIG_DIR, "testing_cnn_on_gen_data.png")):
        print("📊 Creating classifier predictions on generated data plot...")
        class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
                       "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
        feat = get_or_train_feature_extractor()
        model.eval()
        with torch.no_grad():
            fake = model.sample(n_samples=n_show, steps=100).clamp(-1,1)
            logits = feat(fake.to(DEVICE))
            preds = logits.argmax(dim=1).cpu().tolist()
        fig, axes = plt.subplots(1, n_show, figsize=(2.0*n_show, 2.5))
        if n_show == 1: axes = [axes]
        for i in range(n_show):
            axes[i].imshow(((fake[i,0]+1)/2).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(class_names[preds[i]], fontsize=9); axes[i].axis('off')
        plt.tight_layout(); plt.savefig(outpath, dpi=150, bbox_inches='tight'); plt.close()
        print(f"✅ Classifier predictions plot saved: {outpath}")

    save_classifier_on_gen(best, n_show=10)

    # ---- 5) Separate curves with DDPM filenames ----
    def save_separate_curves(losses, gnorms, param_rates, outdir=FIG_DIR):
        print("📊 Creating separate training curve plots...")
        xs = list(range(1, len(losses)+1))
        
        plt.figure(figsize=(8,4))
        plt.plot(xs, losses, marker='o', color='blue', linewidth=2, markersize=4)
        plt.title("Training Loss per Epoch", fontweight='bold')
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "train_loss.png"), dpi=150, bbox_inches='tight'); plt.close()

        plt.figure(figsize=(8,4))
        plt.plot(xs, gnorms, marker='o', color='green', linewidth=2, markersize=4)
        plt.title("Average Gradient Norm", fontweight='bold')
        plt.xlabel("Epoch"); plt.ylabel("Grad Norm"); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "grad_norm.png"), dpi=150, bbox_inches='tight'); plt.close()

        plt.figure(figsize=(8,4))
        plt.plot(xs, param_rates, marker='o', color='red', linewidth=2, markersize=4)
        plt.title("Parameter Change Rate", fontweight='bold')
        plt.xlabel("Epoch"); plt.ylabel("L2 Norm of Change"); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "param_change.png"), dpi=150, bbox_inches='tight'); plt.close()
        
        print(f"✅ Individual curve plots saved to: {outdir}/")

    save_separate_curves(losses, gnorms, param_rates)
    # ================== END EXTRA PLOTS ==================
    
    # Ensure final model is saved
    print("\n💾 Final model save...")
    final_model_path = "weights/rf_weights_final.pth"
    torch.save(best.state_dict(), final_model_path)
    print(f"✅ Final model saved: {final_model_path}")
    
    # Summary (coherent with DDPM report format)
    print("\n🎉 Rectified Flow training completed!")
    print("="*50)
    print(f"📊 Final metrics:")
    print(f"   Loss: {losses[-1]:.6f}")
    print(f"   Grad Norm: {gnorms[-1]:.4f}")
    print(f"   Param Change: {param_rates[-1]:.4f}")
    print(f"💾 Outputs:")
    print(f"   Best model: weights/rf_weights.pth")
    print(f"   Final model: {final_model_path}")
    print(f"   Samples: {SAMPLE_DIR}/")
    print(f"   Training curves: rf_training_curves.png")
    print(f"   DDPM-style plots: {FIG_DIR}/")
    print("="*50)
