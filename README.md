# Wasp Deep Learning Module 3: Rectified Flow on Fashion-MNIST

## Project Overview
This repository contains two implementations of Rectified Flow (RF) models for generating images on the Fashion-MNIST dataset:
1. A robust implementation (`flow_match_robust.py`)
2. An enhanced implementation (`flow_match_enhanced.py`)

## Model Architectures
### Robust Model
- UNet-based velocity field network
- Basic implementation of Rectified Flow
- Straightforward training and sampling

### Enhanced Model
- Improved UNet architecture
- Advanced training techniques:
  - Exponential Moving Average (EMA) of model weights
  - AdamW optimizer with weight decay
  - Gradient clipping
  - Heun (RK2) ODE integrator for sampling
  - Beta(2,2) time sampling
  - Classifier-Free Guidance (CFG)

## Environment Setup
### Conda Environment
```bash
conda env create -f flow_env.yml
conda activate flow_env
```

### Pip Environment
```bash
python -m venv flow_env_pip
source flow_env_pip/bin/activate
pip install -r requirements.txt
```

## Training
### Robust Model
```bash
python flow_match_robust.py
```

### Enhanced Model
```bash
python flow_match_enhanced.py
```

## Evaluation
```bash
python evaluate_enhanced_model.py
```

## Metrics
- Fréchet Inception Distance (FID)
- Kernel Inception Distance (KID)

## Artifacts
- Model weights saved in `weights/`
- Generated images in `rf_enhanced_generated_fmnist/`
- Training and evaluation logs

## Dependencies
- PyTorch
- NumPy
- Matplotlib
- SciPy
- tqdm

## License
[Insert appropriate license]

## Authors
[Your Name]
WASP Deep Learning Course
