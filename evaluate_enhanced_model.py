import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import gzip
import struct

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set Matplotlib backend to Agg
plt.switch_backend('Agg')

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ---------------------------
# DATASET
# ---------------------------
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # Add .copy() to make the array writable
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols).copy()
    return images

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num_items = struct.unpack('>II', f.read(8))
        # Add .copy() to make the array writable
        labels = np.frombuffer(f.read(), dtype=np.uint8).copy()
    return labels

class FashionMNISTDataset(Dataset):
    def __init__(self, train=True, transform=None, data_dir="data"):
        if train:
            image_file = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
            label_file = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        else:
            image_file = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
            label_file = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')

        self.data = load_mnist_images(image_file)
        self.targets = load_mnist_labels(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].reshape(28, 28, 1)
        label = int(self.targets[idx])
        if self.transform:
            image = self.transform(image)
        return image, label

# ---------------------------
# CORRECT Enhanced UNet & RF Classes (from flow_match_enhanced.py)
# ---------------------------
def fourier_time_embedding(t: torch.Tensor, dim: int = 64, max_freq: float = 10.0) -> torch.Tensor:
    if t.ndim == 2 and t.shape[1] == 1: t = t.squeeze(1)
    device = t.device
    freqs = torch.linspace(1.0, max_freq, dim, device=device)
    angles = t[:, None] * freqs[None, :] * 2.0 * torch.pi
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

class FiLM(nn.Module):
    def __init__(self, emb_dim: int, num_channels: int):
        super().__init__()
        self.to_gamma = nn.Linear(emb_dim, num_channels)
        self.to_beta  = nn.Linear(emb_dim, num_channels)
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(t_emb).unsqueeze(-1).unsqueeze(-1)
        beta  = self.to_beta(t_emb).unsqueeze(-1).unsqueeze(-1)
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
        self.down  = nn.Conv2d(out_c, out_c, 4, stride=2, padding=1)
    def forward(self, x, t_emb):
        x = self.block(x, t_emb)
        skip = x
        x = self.down(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, in_c: int, out_c: int, emb_dim: int, skip_c: int = None):
        super().__init__()
        if skip_c is None: skip_c = out_c
        self.up    = nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1)
        self.block = ConvBlock(out_c + skip_c, out_c, emb_dim)
    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, t_emb)
        return x

class UNetRFEnhanced(nn.Module):
    def __init__(self, in_channels: int = 1, base: int = 64, time_dim: int = 128, num_classes: int = 10, depth: int = 2):
        super().__init__()
        self.time_dim = time_dim
        self.depth = depth
        self.time_mlp = nn.Sequential(
            nn.Linear(2 * (time_dim // 2), time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
        )
        self.y_emb = nn.Embedding(num_classes, time_dim)
        self.null_token = nn.Parameter(torch.randn(time_dim))
        self.in_conv = nn.Conv2d(in_channels, base, 3, padding=1)
        self.down1 = Down(base, base*2, time_dim)
        self.down2 = Down(base*2, base*4, time_dim)
        self.mid = ConvBlock(base*4, base*4, time_dim)
        self.up1 = Up(base*4, base*2, time_dim, skip_c=base*4)
        self.up2 = Up(base*2, base, time_dim, skip_c=base*2)
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y=None) -> torch.Tensor:
        t = t.view(-1, 1)
        t_emb = fourier_time_embedding(t, dim=self.time_dim // 2, max_freq=10.0)
        t_emb = self.time_mlp(t_emb)
        if y is not None:
            mask = (torch.rand(y.size(0), device=y.device) > 0.1).float()
            y_emb = self.y_emb(y) * mask.unsqueeze(1) + self.null_token * (1 - mask).unsqueeze(1)
            t_emb = t_emb + y_emb
        else:
            t_emb = t_emb + self.null_token.unsqueeze(0).expand(t_emb.size(0), -1)
        x = self.in_conv(x_t)
        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x = self.mid(x, t_emb)
        x = self.up1(x, s2, t_emb)
        x = self.up2(x, s1, t_emb)
        return self.out_conv(x)

class RectifiedFlowEnhanced(nn.Module):
    def __init__(self, network: nn.Module, device: torch.device):
        super().__init__()
        self.net = network.to(device)
        self.device = device

    @staticmethod
    def _linear_interpolate(x0, x1, t):
        while t.ndim < x0.ndim: t = t.unsqueeze(-1)
        return (1.0 - t) * x0 + t * x1

    @torch.no_grad()
    def sample(self, n_samples: int, steps: int = 150, guidance_scale: float = 1.5, y=None):
        self.eval()
        x = torch.randn(n_samples, 1, 28, 28, device=self.device)
        ts = torch.linspace(0.0, 1.0, steps + 1, device=self.device)
        for i in range(steps):
            t0, t1 = ts[i], ts[i+1]
            dt = t1 - t0
            t0v = torch.full((n_samples,), t0.item(), device=self.device)
            t1v = torch.full((n_samples,), t1.item(), device=self.device)
            if y is not None and guidance_scale > 1.0:
                v_cond   = self.net(x, t0v, y)
                v_uncond = self.net(x, t0v, None)
                v0 = v_uncond + guidance_scale * (v_cond - v_uncond)
                
                x_euler = x + dt * v0
                v_cond   = self.net(x_euler, t1v, y)
                v_uncond = self.net(x_euler, t1v, None)
                v1 = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v0 = self.net(x, t0v, y)
                x_euler = x + dt * v0
                v1 = self.net(x_euler, t1v, y)
            x = x + 0.5 * dt * (v0 + v1)
        return x.clamp(-1.0, 1.0)

# ---------------------------
# FEATURE EXTRACTOR
# ---------------------------
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
        if return_features: return feats
        out = self.fc2(F.relu(feats))
        return out

def get_or_train_feature_extractor(
    train_loader: DataLoader, 
    path="feature_extractor_fashionmnist.pth",
    force_train=False
):
    fe = FeatureExtractor().to(DEVICE)
    if os.path.exists(path) and not force_train:
        print(f"Loading pre-trained feature extractor from {path}")
        fe.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        return fe
    
    print(f"Training feature extractor and saving to {path}...")
    optimizer = torch.optim.Adam(fe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(5):
        for images, labels in tqdm(train_loader, desc=f"FE Train Epoch {epoch+1}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = fe(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    torch.save(fe.state_dict(), path)
    return fe

# ---------------------------
# FID and KID METRICS
# ---------------------------
def compute_fid_kid(real_feats, fake_feats):
    # FID
    mu1, sigma1 = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    # KID (Corrected Implementation)
    m, n = real_feats.shape[0], fake_feats.shape[0]
    
    # Polynomial kernel, d=num_features, p=3
    d = real_feats.shape[1]
    K_XX = (np.dot(real_feats, real_feats.T) / d + 1) ** 3
    K_YY = (np.dot(fake_feats, fake_feats.T) / d + 1) ** 3
    K_XY = (np.dot(real_feats, fake_feats.T) / d + 1) ** 3

    # Unbiased estimator for MMD^2
    # Ensure diagonal elements are zero for unbiased estimation
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    
    kid_mean = np.sum(K_XX) / (m * (m - 1)) + np.sum(K_YY) / (n * (n - 1)) - 2 * np.sum(K_XY) / (m * n)

    # Bootstrapping for CI (simplified and corrected)
    kid_vars = []
    for _ in range(100):
        idx_real = np.random.choice(m, m, replace=True)
        idx_fake = np.random.choice(n, n, replace=True)
        
        K_XX_boot = K_XX[np.ix_(idx_real, idx_real)]
        K_YY_boot = K_YY[np.ix_(idx_fake, idx_fake)]
        K_XY_boot = K_XY[np.ix_(idx_real, idx_fake)]
        
        # Unbiased estimator on the bootstrapped sample
        m_boot, n_boot = len(idx_real), len(idx_fake)
        term1 = np.sum(K_XX_boot) / (m_boot * (m_boot - 1)) if m_boot > 1 else 0
        term2 = np.sum(K_YY_boot) / (n_boot * (n_boot - 1)) if n_boot > 1 else 0
        term3 = -2 * np.sum(K_XY_boot) / (m_boot * n_boot)
        
        kid_vars.append(term1 + term2 + term3)

    kid_ci = 1.96 * np.std(kid_vars)

    return fid, (kid_mean, kid_ci)

# ---------------------------
# LOAD ENHANCED MODEL (DEFINITIVELY CORRECTED)
# ---------------------------
def load_enhanced_model():
    print("📦 Loading Enhanced RF model...")
    model_path = "weights/rf_enhanced_weights_final.pth"
    if not os.path.exists(model_path):
        print(f"Error: Enhanced model weights not found at {model_path}")
        return None
    
    # Instantiate the CORRECT model architectures from the training script
    # The trained model used depth=2
    unet = UNetRFEnhanced(in_channels=1, base=64, time_dim=128, depth=2).to(DEVICE)
    rf_model = RectifiedFlowEnhanced(unet, device=DEVICE)
    
    # Load the state dict for the entire RectifiedFlowEnhanced object
    state_dict = torch.load(model_path, map_location=DEVICE)
    rf_model.load_state_dict(state_dict)
    
    rf_model.eval()
    return rf_model

# ---------------------------
# MAIN EVALUATION
# ---------------------------
def main():
    print("🚀 Starting Enhanced Model Evaluation with FID/KID Metrics")
    print("="*60)
    
    # Data
    print("📊 Loading datasets from 'data/' directory...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = FashionMNISTDataset(train=True, transform=transform)
    test_dataset = FashionMNISTDataset(train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    train_images = np.array([img.numpy() for img, _ in train_dataset])
    test_images = np.array([img.numpy() for img, _ in test_dataset])

    # Feature Extractor
    print("🔧 Loading/training feature extractor...")
    feature_extractor = get_or_train_feature_extractor(train_loader)
    
    # Load Model
    enhanced_model = load_enhanced_model()
    if enhanced_model is None:
        return
        
    # Generate Samples
    print("🎨 Generating 10,000 samples from Enhanced RF model...")
    generated_samples = []
    with torch.no_grad():
        for _ in tqdm(range(10000 // 100), desc="Generating samples"):
            samples = enhanced_model.sample(100, steps=150, guidance_scale=2.0)
            generated_samples.append(samples.cpu().numpy())
    generated_images = np.concatenate(generated_samples, axis=0)

    # Extract Features (Corrected logic for Train vs Train)
    print("🔍 Extracting features from all datasets...")
    # Use 20k images for train-train comparison, 10k for others
    train_images_for_feats = train_images[:20000] 
    
    @torch.no_grad()
    def get_features(images):
        features = []
        img_tensor = torch.from_numpy(images).to(DEVICE)
        for i in tqdm(range(0, len(img_tensor), 128), desc="Extracting features"):
            batch = img_tensor[i:i+128]
            feats = feature_extractor(batch, return_features=True)
            features.append(feats.cpu().numpy())
        return np.concatenate(features, axis=0)

    # Get features for all 20k train images
    all_train_feats = get_features(train_images_for_feats)
    # Split into two non-overlapping sets for baseline
    train_feats_1 = all_train_feats[:10000]
    train_feats_2 = all_train_feats[10000:]
    
    test_feats = get_features(test_images)
    generated_feats = get_features(generated_images)

    # Calculate Metrics
    print("🧮 Calculating FID/KID scores...")
    results = {}
    
    fid_train, kid_train = compute_fid_kid(train_feats_1, train_feats_2)
    results["Train vs Train (baseline)"] = (fid_train, kid_train)

    fid_gen, kid_gen = compute_fid_kid(train_feats_1, generated_feats)
    results["Train vs Generated"] = (fid_gen, kid_gen)
    
    fid_test, kid_test = compute_fid_kid(test_feats, generated_feats)
    results["Test vs Generated"] = (fid_test, kid_test)
    
    # Print and Save Results
    print("\n✅ Evaluation Complete!")
    print("="*60)
    header = f"| {'Comparison':<25} | {'FID Score':<15} | {'KID (mean ± 95% CI)':<25} |"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    
    output_lines = [separator, header, separator]
    for name, (fid, (kid_mean, kid_ci)) in results.items():
        line = f"| {name:<25} | {fid:<15.4f} | {f'{kid_mean:.4f} ± {kid_ci:.4f}':<25} |"
        print(line)
        output_lines.append(line)
    print(separator)
    output_lines.append(separator)

    with open("evaluation_results_enhanced.txt", "w") as f:
        f.write("\n".join(output_lines))
    print("\n💾 Results saved to evaluation_results_enhanced.txt")

if __name__ == "__main__":
    main()
