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
from utils.logging import get_training_logger

logger = get_training_logger(experiment_name="examples")


def example_1_basic_forward_pass():
    """Example 1: Basic forward pass through the model"""
    logger.info("=" * 60)
    logger.info("Example 1: Basic Forward Pass")
    logger.info("=" * 60)

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
    logger.info("Encoder output:")
    logger.info(f"  Mean shape: {m.shape}")
    logger.info(f"  Std shape:  {s.shape}")
    logger.info(f"  Mean range: [{m.min():.2f}, {m.max():.2f}]")
    logger.info(f"  Std range:  [{s.min():.2f}, {s.max():.2f}]")

    # Sample latent
    z = m + s * torch.randn_like(m)

    # Decode
    Timg, Tpx = decoder(z)
    logger.info("Decoder output:")
    logger.info(f"  Image shape: {Timg.shape}")
    logger.info(f"  px shape:    {Tpx.shape}")
    logger.info(f"  px range:    [{Tpx.min():.3f}, {Tpx.max():.3f}]")


def example_2_training_step():
    """Example 2: Single training step"""
    logger.info("=" * 60)
    logger.info("Example 2: Training Step")
    logger.info("=" * 60)

    hp = HyperParameters.finance_low_snr()
    hp.epochs = 1  # Just for demo

    encoder = DerpEncoder(latent_dim=hp.latent_dim)
    decoder = DerpDecoder(latent_dim=hp.latent_dim)
    engine = DerpEngine(encoder, decoder, hp, device='cpu')

    # Sample batch
    batch_size = 8
    img = torch.randn(batch_size, 1, 39, 39)
    labels = torch.randint(0, 2, (batch_size,))

    logger.info("Batch:")
    logger.info(f"  Images: {img.shape}")
    logger.info(f"  Labels: {labels}")

    # Training step
    loss_dict = engine.training_step(img, labels)

    logger.info("Losses:")
    for key, val in loss_dict.items():
        if key != 'total':
            logger.info(f"  {key:12s}: {val:.4f}")
    logger.info(f"  {'total':12s}: {loss_dict['total']:.4f}")


def example_3_prediction():
    """Example 3: Prediction with Bayesian marginalization"""
    logger.info("=" * 60)
    logger.info("Example 3: Prediction")
    logger.info("=" * 60)

    hp = HyperParameters.finance_low_snr()
    hp.S_pred = 4  # Fewer samples for demo

    encoder = DerpEncoder(latent_dim=hp.latent_dim)
    decoder = DerpDecoder(latent_dim=hp.latent_dim)
    engine = DerpEngine(encoder, decoder, hp, device='cpu')

    # Test image
    img_test = torch.randn(1, 1, 39, 39)

    logger.info("Predicting on test image...")
    logger.info(f"  Using {hp.S_pred} importance samples")

    prob = engine.predict_stable(img_test)

    logger.info("Prediction:")
    logger.info(f"  P(label=1) = {prob:.4f}")
    logger.info(f"  P(label=0) = {1-prob:.4f}")
    logger.info(f"  Predicted label: {int(prob > 0.5)}")
    logger.info(f"  Edge: {abs(prob - 0.5):.4f}")


def example_4_monitoring():
    """Example 4: Health monitoring"""
    logger.info("=" * 60)
    logger.info("Example 4: Health Monitoring")
    logger.info("=" * 60)

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
    logger.info("Collected metrics from 3 batches:")
    summary = engine.monitor.get_summary()

    # Check health
    logger.info("Health Check:")
    if 'msi' in summary:
        status = "[OK]" if summary.get('msi_healthy', False) else "[WARN]"
        logger.info(f"  MSI:     {summary['msi']:.4f} {status}")

    if 'beta' in summary:
        status = "[OK]" if summary.get('beta_healthy', False) else "[WARN]"
        logger.info(f"  Beta:    {summary['beta']:.4f} {status}")

    if 'px_separation' in summary:
        status = "[OK]" if summary.get('px_sep_healthy', False) else "[WARN]"
        logger.info(f"  px Sep:  {summary['px_separation']:.4f} {status}")


def example_5_custom_hyperparameters():
    """Example 5: Custom hyperparameter configuration"""
    logger.info("=" * 60)
    logger.info("Example 5: Custom Hyperparameters")
    logger.info("=" * 60)

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

    logger.info("Custom configuration:")
    logger.info(f"  Latent dim:    {hp.latent_dim}")
    logger.info(f"  px peaks:      {0.5 - hp.pxBias:.2f}, {0.5 + hp.pxBias:.2f}")
    logger.info(f"  px stdev:      {hp.pxStdev:.2f}")
    logger.info(f"  Rscale:        {hp.Rscale}")
    logger.info(f"  lambda_rp:     {hp.lambda_rp}")
    logger.info(f"  batch_size:    {hp.batch_size}")
    logger.info(f"  learning rate: {hp.lr}")

    # Use preset configurations
    logger.info("Preset configurations available:")
    logger.info("  1. HyperParameters.finance_low_snr()")
    logger.info("  2. HyperParameters.finance_high_freq()")


def main():
    """Run all examples"""
    logger.info("=" * 60)
    logger.info("Derp-VAE Examples")
    logger.info("=" * 60)

    example_1_basic_forward_pass()
    example_2_training_step()
    example_3_prediction()
    example_4_monitoring()
    example_5_custom_hyperparameters()

    logger.info("=" * 60)
    logger.info("All Examples Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
