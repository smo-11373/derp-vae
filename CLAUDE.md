# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Derp-VAE (Distribution Enforcement via Random Probe for Variational Auto-Encoder) is a specialized VAE implementation for quantitative finance applications with extremely low signal-to-noise ratios. It extends standard VAE with marginal distribution enforcement, bimodal label priors, and random probe testing via Cramér-Wold theorem.

## Commands

### Installation
```bash
pip install torch numpy scikit-learn
```

### Training
```bash
python train.py --epochs 20 --batch_size 64 --n_samples 1000
```
Arguments: `--latent_dim`, `--epochs`, `--batch_size`, `--lr`, `--n_samples`, `--device`

### Prediction
```bash
python predict.py --checkpoint derp_vae_model.pt --n_samples 100
```

### Run Examples (Integration Tests)
```bash
python examples.py
```

## Architecture

### Data Flow
```
img -> Encoder(img, px) -> (m, s) -> sample z0 -> Decoder(z0) -> (Timg, Tpx)
                                                        |
                                              Sample x1 -> Re-encoder -> z1
```

### Key Components

**models/encoder.py** - `DerpEncoder`: Takes (img, px) and outputs latent mean/std (m, s). Uses 3-layer MLP with BatchNorm, LeakyReLU, Dropout.

**models/decoder.py** - `DerpDecoder`: Takes latent z and outputs (Timg, Tpx). Shared backbone with two heads - image head (unbounded) and label head (sigmoid to [0,1]).

**training/engine.py** - `DerpEngine`: Complete training loop with ELBO + DERP loss. Handles adaptive scaling, prediction with log-sum-exp stability.

**training/random_probe.py** - DERP loss implementation testing z0~N(0,I), z1~N(0,I), dz~N(0,I), px1~Bimodal via Cramér-Wold theorem.

**training/monitor.py** - `Monitor`: Tracks MSI (Manifold Stability Index), calibration beta, px separation.

**config/hyperparameters.py** - `HyperParameters` dataclass with two presets: `finance_low_snr()` and `finance_high_freq()`.

### Loss Components (ELBO)
1. Image reconstruction (MSE)
2. Complete KL divergence: `0.5 * (m0² + s0² - 2*ln(s0) - 1)`
3. Classification loss (bimodal likelihood on px1)
4. Recovery loss (z0 ≈ encoder(decoder(z0)))
5. DERP loss (weighted by `lambda_rp`)

## Key Hyperparameters

- `latent_dim`: 64 (latent space dimension)
- `pxBias`: 0.05 (peaks at 0.45 and 0.55)
- `pxStdev`: 0.10 (peak width for bimodal distribution)
- `Rscale`: 0.9 (recovery constraint multiplier)
- `Gscale`: 0.1 (generative noise scale)
- `lambda_rp`: 1.0 (DERP loss weight)
- `S_pred`: 8 (importance samples for prediction)

## Health Metrics

- **MSI (Manifold Stability Index)**: `1 - Var(z1-z0)/Var(z0)`. Target > 0.80, critical if < 0.60
- **Calibration Beta**: Slope from linear regression. Target 0.10-0.50 for finance (shrinkage expected)
- **px Separation**: Distance between class modes. Target > 0.02

## Implementation Notes

- Uses `torch.logsumexp()` for numerical stability in prediction
- Xavier normal initialization for weight stability
- Gradient clipping default threshold: 5.0
- Input image shape: (batch, 1, 39, 39)
- Regression shrinkage (β < 1) is expected due to information loss in encoder-decoder cycle
