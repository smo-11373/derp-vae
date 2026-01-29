"""
Prediction Script for Derp-VAE
Load trained model and make predictions
"""

import torch
import numpy as np
import argparse

from models.encoder import DerpEncoder
from models.decoder import DerpDecoder
from training.engine import DerpEngine
from utils.logging import get_training_logger

logger = get_training_logger(experiment_name="predict")


def load_model(checkpoint_path, device='cuda'):
    """
    Load trained Derp-VAE model
    
    Args:
        checkpoint_path: Path to saved model
        device: torch device
    
    Returns:
        engine: DerpEngine with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    hp = checkpoint['hp']
    
    # Initialize architecture
    encoder = DerpEncoder(
        img_size=hp.img_size,
        latent_dim=hp.latent_dim,
        hidden_dim=hp.hidden_dim
    )
    
    decoder = DerpDecoder(
        latent_dim=hp.latent_dim,
        img_size=hp.img_size,
        hidden_dim=hp.hidden_dim
    )
    
    # Load weights
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    # Create engine
    engine = DerpEngine(encoder, decoder, hp, device)
    engine.encoder.eval()
    engine.decoder.eval()
    
    return engine, hp


def predict_batch(engine, images):
    """
    Predict probabilities for a batch of images
    
    Args:
        engine: DerpEngine instance
        images: (batch, 1, H, W) tensor
    
    Returns:
        probs: (batch,) predicted probabilities
    """
    probs = []
    
    images = images.to(engine.device)
    
    for i in range(images.size(0)):
        prob = engine.predict_stable(images[i:i+1])
        probs.append(prob)
    
    return np.array(probs)


def analyze_predictions(probs, labels=None):
    """
    Analyze prediction distribution

    Args:
        probs: (n,) predicted probabilities
        labels: (n,) true labels (optional)
    """
    logger.info("=" * 60)
    logger.info("Prediction Analysis")
    logger.info("=" * 60)

    logger.info("Probability Statistics:")
    logger.info(f"  Mean:   {probs.mean():.4f}")
    logger.info(f"  Median: {np.median(probs):.4f}")
    logger.info(f"  Std:    {probs.std():.4f}")
    logger.info(f"  Min:    {probs.min():.4f}")
    logger.info(f"  Max:    {probs.max():.4f}")

    # Edge distribution
    edges = np.abs(probs - 0.5)
    logger.info("Edge Statistics:")
    logger.info(f"  Mean edge:     {edges.mean():.4f}")
    logger.info(f"  Edge > 0.02:   {(edges > 0.02).sum()} / {len(edges)} ({(edges > 0.02).mean()*100:.1f}%)")
    logger.info(f"  Edge > 0.05:   {(edges > 0.05).sum()} / {len(edges)} ({(edges > 0.05).mean()*100:.1f}%)")
    logger.info(f"  Edge > 0.10:   {(edges > 0.10).sum()} / {len(edges)} ({(edges > 0.10).mean()*100:.1f}%)")

    # Histogram
    logger.info("Probability Histogram:")
    bins = np.linspace(0, 1, 11)
    counts, _ = np.histogram(probs, bins=bins)
    for i in range(len(counts)):
        bar = "#" * int(counts[i] / counts.max() * 50)
        logger.info(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}]: {bar} ({counts[i]})")

    if labels is not None:
        # Accuracy
        predictions = (probs > 0.5).astype(int)
        accuracy = (predictions == labels).mean()

        logger.info(f"Accuracy: {accuracy:.4f}")

        # Calibration by decile
        logger.info("Calibration by Decile:")
        deciles = np.percentile(probs, np.arange(0, 101, 10))
        for i in range(10):
            mask = (probs >= deciles[i]) & (probs < deciles[i+1])
            if mask.sum() > 0:
                mean_prob = probs[mask].mean()
                mean_label = labels[mask].mean()
                logger.info(
                    f"  Decile {i+1}: Pred={mean_prob:.3f}, Actual={mean_label:.3f}, "
                    f"n={mask.sum()}"
                )

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Predict with Derp-VAE')
    parser.add_argument('--checkpoint', type=str, default='derp_vae_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of samples to predict')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    engine, hp = load_model(args.checkpoint, device)

    logger.info("Model loaded successfully!")
    logger.info(f"  Latent dim: {hp.latent_dim}")
    logger.info(f"  Image size: {hp.img_size}")

    # Generate test data
    logger.info(f"Generating {args.n_samples} test samples...")
    test_images = torch.randn(args.n_samples, 1, hp.img_size[0], hp.img_size[1])

    # Create weak labels for evaluation
    img_mean = test_images.mean(dim=(1, 2, 3))
    test_labels = (img_mean > img_mean.median()).long().numpy()

    # Predict
    logger.info("Predicting...")
    probs = predict_batch(engine, test_images)

    # Analyze
    analyze_predictions(probs, test_labels)

    # Example predictions
    logger.info("Example Predictions:")
    for i in range(min(10, args.n_samples)):
        logger.info(
            f"  Sample {i}: Prob={probs[i]:.4f}, Label={test_labels[i]}, "
            f"Pred={int(probs[i] > 0.5)}"
        )


if __name__ == "__main__":
    main()
