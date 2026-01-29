# Derp-VAE: Complete Implementation

**Distribution Enforcement via Random Probe for Variational Auto-Encoder**

A complete implementation of Derp-VAE designed for quantitative finance applications with extremely low signal-to-noise ratios (SNR).

## Overview

Derp-VAE extends the standard Variational Autoencoder with:

1. **Marginal Distribution Framework**: Tests three marginal normality constraints (z0, dz, z1) while allowing correlation structure to emerge naturally
2. **Bimodal Label Prior**: Continuous probability space with overlapping peaks at 0.45 and 0.55
3. **Random Probe (DERP) Testing**: Enforces high-dimensional normality via Cramér-Wold theorem
4. **Recovery Constraints**: Ensures encoder-decoder-encoder cycle preserves latent structure
5. **Adaptive Scaling**: RMSE-based scale updates for data-driven constraint adjustment

## Installation

```bash
# Clone or download the repository
cd derp_vae_complete

# Install dependencies
pip install torch numpy scikit-learn
```

## Quick Start

### Training

```bash
python train.py --epochs 20 --batch_size 64 --n_samples 1000
```

### Prediction

```bash
python predict.py --checkpoint derp_vae_model.pt --n_samples 100
```

## Project Structure

```
derp_vae_complete/
├── models/
│   ├── encoder.py          # DerpEncoder: (img, px) -> (m, s)
│   └── decoder.py          # DerpDecoder: z -> (Timg, Tpx)
├── training/
│   ├── engine.py           # Complete training engine
│   ├── random_probe.py     # DERP loss implementation
│   └── monitor.py          # Health metrics (MSI, beta, etc.)
├── config/
│   └── hyperparameters.py  # Configuration dataclass
├── train.py                # Main training script
├── predict.py              # Inference script
└── README.md               # This file
```

## Key Features

### 1. Complete ELBO Loss

The training loss includes:

- **Image Reconstruction**: MSE between input and decoded image
- **Complete KL Divergence**: `0.5 * (m0² + s0² - 2*ln(s0) - 1)`
- **Classification Loss**: Bimodal likelihood on px1
- **Recovery Loss**: Ensures z0 ≈ encoder(decoder(z0))

### 2. DERP (Random Probe) Loss

Tests four distributional constraints:
- `p(z0) = N(0,I)` - Latent normality
- `p(z1) = N(0,I)` - Ergodicity
- `p(dz) = N(0,I)` - Recovery normality
- `p(px1) = Bimodal` - Label space structure

### 3. Adaptive Scaling

Automatically adjusts constraint strictness based on empirical RMSE:

```python
z_rmse = mean(std(z1 - z0))
z_Rscale = hp.Rz * z_rmse  # typically hp.Rz = 0.9
```

### 4. Health Monitoring

Tracks essential metrics:
- **MSI** (Manifold Stability Index): `1 - Var(z1-z0)/Var(z0)` > 0.80
- **Calibration β**: Slope of predicted vs actual, target 0.10-0.50
- **px Separation**: Distance between class modes > 0.02

### 5. Numerical Stability

Uses log-sum-exp for prediction to prevent underflow:

```python
log_avg_0 = torch.logsumexp(log_likelihoods_0, dim=0) - log(S_pred)
log_avg_1 = torch.logsumexp(log_likelihoods_1, dim=0) - log(S_pred)
prob = sigmoid(log_avg_1 - log_avg_0)
```

## Hyperparameters

### Default Configuration (Low-SNR Finance)

```python
HyperParameters.finance_low_snr()
```

- **latent_dim**: 64
- **pxBias**: 0.05 (peaks at 0.45 and 0.55)
- **pxStdev**: 0.10 (peak width)
- **Rscale**: 0.9 (recovery constraint strictness)
- **Gscale**: 0.1 (generative noise)
- **lambda_rp**: 1.0 (DERP loss weight)
- **S_pred**: 8 (importance samples for prediction)

### High-Frequency Trading Configuration

```python
HyperParameters.finance_high_freq()
```

Tighter constraints for higher-frequency signals:
- **pxBias**: 0.03
- **Rscale**: 0.95
- **lambda_rp**: 2.0

## Usage Examples

### Custom Training

```python
from models.encoder import DerpEncoder
from models.decoder import DerpDecoder
from training.engine import DerpEngine
from config.hyperparameters import HyperParameters

# Setup
hp = HyperParameters.finance_low_snr()
encoder = DerpEncoder(latent_dim=hp.latent_dim)
decoder = DerpDecoder(latent_dim=hp.latent_dim)
engine = DerpEngine(encoder, decoder, hp, device='cuda')

# Train
for epoch in range(hp.epochs):
    for img, labels in train_loader:
        loss_dict = engine.training_step(img, labels)
    
    engine.update_scales()  # Update adaptive scales
    summary = engine.monitor.print_summary(epoch)
```

### Prediction

```python
# Load model
engine, hp = load_model('derp_vae_model.pt')

# Predict
prob = engine.predict_stable(test_image)
print(f"P(label=1) = {prob:.4f}")
```

### Monitoring Health

```python
# During training
summary = engine.monitor.get_summary()

if summary['msi'] < 0.60:
    print("WARNING: Manifold collapse detected!")
    
if summary['beta'] < 0.05:
    print("WARNING: No signal detected!")
```

## Health Metrics Interpretation

### Manifold Stability Index (MSI)

- **MSI > 0.80**: ✓ Healthy - good recovery
- **MSI 0.60-0.80**: ⚠️ Warning - recovery degrading
- **MSI < 0.60**: ✗ Critical - manifold collapse

### Calibration Beta (β)

- **β = 1.0**: Perfect calibration
- **β = 0.10-0.50**: ✓ Healthy for finance (shrinkage expected)
- **β < 0.05**: ✗ No signal - model not predictive

### px Separation

- **sep > 0.05**: ✓ Strong class separation
- **sep > 0.02**: ✓ Adequate separation
- **sep < 0.02**: ⚠️ Weak separation
- **sep ≈ 0.00**: ✗ Regression to mean

## Theoretical Background

### Marginal vs Conditional Distributions

The framework tests **marginal** distributions:
- `p(z0) = N(0,I)` - marginal over data
- `p(dz) = N(0,I)` - marginal over cycles
- `p(z1) = N(0,I)` - marginal over generation

These are **not redundant** - they constrain different aspects of the joint distribution while leaving the correlation structure `Cov(z0, dz)` free to emerge.

### Regression Shrinkage

Due to information loss in the cycle `z0 -> x1 -> z1`, we expect:
```
E[z1|z0] ≈ λ·z0 where λ < 1
```

This is **not a bug** - it's a feature that reflects realistic signal degradation. The marginal constraints ensure this shrinkage is statistically well-behaved.

### Bimodal Prior for Low-SNR

With peaks at 0.45 and 0.55 (separation = 0.10):
- Model can output 0.51, 0.49, 0.53 (weak edges)
- Avoids forcing to 0 or 1 (unrealistic in finance)
- Allows calibration: predicted 0.53 → actual 0.5045 (β = 0.15)

## Common Issues

### 1. MSI Dropping

**Symptom**: MSI < 0.60
**Cause**: Encoder/decoder losing coherence
**Fix**: 
- Increase `lambda_rp` (strengthen DERP)
- Decrease `Gscale` (reduce noise)
- Check if input features are purely random

### 2. px Collapsing to 0.5

**Symptom**: px separation ≈ 0
**Cause**: No classification supervision
**Fix**: Verify classification loss is included (it is in this implementation!)

### 3. Prediction Underflow

**Symptom**: NaN in predictions
**Cause**: exp(-large_nll) → 0
**Fix**: Use log-sum-exp (already implemented)

## Citation

If you use this implementation, please cite:

```
Derp-VAE: Distribution Enforcement via Random Probe for Variational Auto-Encoder
Implementation for Quantitative Finance Applications
```

## License

MIT License

## Acknowledgments

- Based on the DERP methodology for distribution enforcement
- Designed for low-SNR quantitative finance applications
- Implements complete ELBO with Random Probe testing
