"""
Example Usage of Derp-VAE
Quick demonstration of key features
"""

import torch
from models.encoder import DerpEncoder
from models.decoder import DerpDecoder
from training.engine import DerpEngine
from training.monitor import Monitor
from config.hyperparameters import HyperParameters


def example_1_basic_forward_pass():
    """Example 1: Basic forward pass through the model"""
    print("\n" + "="*60)
    print("Example 1: Basic Forward Pass")
    print("="*60)
    
    hp = HyperParameters.finance_low_snr()
    
    # Initialize models
    encoder = DerpEncoder(latent_dim=hp.latent_dim)
    decoder = DerpDecoder(latent_dim=hp.latent_dim)
    
    # Create sample input
    batch_size = 4
    img = torch.randn(batch_size, 1, 39, 39)
    px = torch.rand(batch_size, 1) * 0.2 + 0.4  # Random in [0.4, 0.6]
    
    # Encode
    m, s = encoder(img, px)
    print(f"\nEncoder output:")
    print(f"  Mean shape: {m.shape}")
    print(f"  Std shape:  {s.shape}")
    print(f"  Mean range: [{m.min():.2f}, {m.max():.2f}]")
    print(f"  Std range:  [{s.min():.2f}, {s.max():.2f}]")
    
    # Sample latent
    z = m + s * torch.randn_like(m)
    
    # Decode
    Timg, Tpx = decoder(z)
    print(f"\nDecoder output:")
    print(f"  Image shape: {Timg.shape}")
    print(f"  px shape:    {Tpx.shape}")
    print(f"  px range:    [{Tpx.min():.3f}, {Tpx.max():.3f}]")


def example_2_training_step():
    """Example 2: Single training step"""
    print("\n" + "="*60)
    print("Example 2: Training Step")
    print("="*60)
    
    hp = HyperParameters.finance_low_snr()
    hp.epochs = 1  # Just for demo
    
    encoder = DerpEncoder(latent_dim=hp.latent_dim)
    decoder = DerpDecoder(latent_dim=hp.latent_dim)
    engine = DerpEngine(encoder, decoder, hp, device='cpu')
    
    # Sample batch
    batch_size = 8
    img = torch.randn(batch_size, 1, 39, 39)
    labels = torch.randint(0, 2, (batch_size,))
    
    print(f"\nBatch:")
    print(f"  Images: {img.shape}")
    print(f"  Labels: {labels}")
    
    # Training step
    loss_dict = engine.training_step(img, labels)
    
    print(f"\nLosses:")
    for key, val in loss_dict.items():
        if key != 'total':
            print(f"  {key:12s}: {val:.4f}")
    print(f"  {'total':12s}: {loss_dict['total']:.4f}")


def example_3_prediction():
    """Example 3: Prediction with Bayesian marginalization"""
    print("\n" + "="*60)
    print("Example 3: Prediction")
    print("="*60)
    
    hp = HyperParameters.finance_low_snr()
    hp.S_pred = 4  # Fewer samples for demo
    
    encoder = DerpEncoder(latent_dim=hp.latent_dim)
    decoder = DerpDecoder(latent_dim=hp.latent_dim)
    engine = DerpEngine(encoder, decoder, hp, device='cpu')
    
    # Test image
    img_test = torch.randn(1, 1, 39, 39)
    
    print(f"\nPredicting on test image...")
    print(f"  Using {hp.S_pred} importance samples")
    
    prob = engine.predict_stable(img_test)
    
    print(f"\nPrediction:")
    print(f"  P(label=1) = {prob:.4f}")
    print(f"  P(label=0) = {1-prob:.4f}")
    print(f"  Predicted label: {int(prob > 0.5)}")
    print(f"  Edge: {abs(prob - 0.5):.4f}")


def example_4_monitoring():
    """Example 4: Health monitoring"""
    print("\n" + "="*60)
    print("Example 4: Health Monitoring")
    print("="*60)
    
    hp = HyperParameters.finance_low_snr()
    
    encoder = DerpEncoder(latent_dim=hp.latent_dim)
    decoder = DerpDecoder(latent_dim=hp.latent_dim)
    engine = DerpEngine(encoder, decoder, hp, device='cpu')
    
    # Simulate training data
    for _ in range(3):
        img = torch.randn(8, 1, 39, 39)
        labels = torch.randint(0, 2, (8,))
        engine.training_step(img, labels)
    
    # Get summary
    print("\nCollected metrics from 3 batches:")
    summary = engine.monitor.get_summary()
    
    # Check health
    print(f"\nHealth Check:")
    if 'msi' in summary:
        status = "✓ HEALTHY" if summary.get('msi_healthy', False) else "✗ UNHEALTHY"
        print(f"  MSI:     {summary['msi']:.4f} {status}")
    
    if 'beta' in summary:
        status = "✓ HEALTHY" if summary.get('beta_healthy', False) else "✗ UNHEALTHY"
        print(f"  Beta:    {summary['beta']:.4f} {status}")
    
    if 'px_separation' in summary:
        status = "✓ HEALTHY" if summary.get('px_sep_healthy', False) else "✗ UNHEALTHY"
        print(f"  px Sep:  {summary['px_separation']:.4f} {status}")


def example_5_custom_hyperparameters():
    """Example 5: Custom hyperparameter configuration"""
    print("\n" + "="*60)
    print("Example 5: Custom Hyperparameters")
    print("="*60)
    
    # Create custom configuration
    hp = HyperParameters(
        latent_dim=32,
        pxBias=0.03,
        pxStdev=0.08,
        Rscale=0.95,
        lambda_rp=2.0,
        batch_size=128,
        lr=5e-4
    )
    
    print("\nCustom configuration:")
    print(f"  Latent dim:    {hp.latent_dim}")
    print(f"  px peaks:      {0.5 - hp.pxBias:.2f}, {0.5 + hp.pxBias:.2f}")
    print(f"  px stdev:      {hp.pxStdev:.2f}")
    print(f"  Rscale:        {hp.Rscale}")
    print(f"  lambda_rp:     {hp.lambda_rp}")
    print(f"  batch_size:    {hp.batch_size}")
    print(f"  learning rate: {hp.lr}")
    
    # Use preset configurations
    print("\nPreset configurations available:")
    print("  1. HyperParameters.finance_low_snr()")
    print("  2. HyperParameters.finance_high_freq()")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("Derp-VAE Examples")
    print("="*60)
    
    example_1_basic_forward_pass()
    example_2_training_step()
    example_3_prediction()
    example_4_monitoring()
    example_5_custom_hyperparameters()
    
    print("\n" + "="*60)
    print("All Examples Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
