"""
Main Training Script for Derp-VAE
Complete training loop with validation and monitoring
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import argparse

from models.encoder import DerpEncoder
from models.decoder import DerpDecoder
from training.engine import DerpEngine
from config.hyperparameters import HyperParameters


def create_synthetic_data(n_samples=1000, img_size=(39, 39), seed=42):
    """
    Create synthetic financial data for testing
    
    Args:
        n_samples: Number of samples
        img_size: Image dimensions
        seed: Random seed
    
    Returns:
        images: (n_samples, 1, H, W)
        labels: (n_samples,)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate random images
    images = torch.randn(n_samples, 1, img_size[0], img_size[1])
    
    # Generate labels with slight correlation to image statistics
    # (to simulate low-SNR financial signals)
    img_mean = images.mean(dim=(1, 2, 3))
    img_std = images.std(dim=(1, 2, 3))
    
    # Weak signal: label depends on mean + noise
    signal = img_mean + 0.1 * img_std
    labels = (signal > signal.median()).long()
    
    return images, labels


def train_epoch(engine, dataloader, epoch):
    """
    Train for one epoch
    
    Args:
        engine: DerpEngine instance
        dataloader: DataLoader
        epoch: Current epoch number
    
    Returns:
        summary: dict with epoch statistics
    """
    engine.monitor.reset()
    
    for batch_idx, (img, labels) in enumerate(dataloader):
        img = img.to(engine.device)
        labels = labels.to(engine.device)
        
        # Training step
        loss_dict = engine.training_step(img, labels)
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: "
                  f"Loss={loss_dict['total']:.4f}, "
                  f"KL={loss_dict['kl']:.4f}, "
                  f"Class={loss_dict['class']:.4f}, "
                  f"DERP={loss_dict['derp']:.4f}")
    
    # Update adaptive scales at end of epoch
    engine.update_scales()
    
    # Get epoch summary
    summary = engine.monitor.print_summary(epoch + 1)
    
    return summary


def validate(engine, val_loader):
    """
    Validate model on validation set
    
    Args:
        engine: DerpEngine instance
        val_loader: Validation DataLoader
    
    Returns:
        accuracy: validation accuracy
        avg_prob: average predicted probability
    """
    engine.encoder.eval()
    engine.decoder.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for img, labels in val_loader:
            img = img.to(engine.device)
            labels = labels.to(engine.device)
            
            # Predict each sample
            for i in range(img.size(0)):
                prob = engine.predict_stable(img[i:i+1])
                all_probs.append(prob)
                all_labels.append(labels[i].item())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Compute accuracy
    predictions = (all_probs > 0.5).astype(int)
    accuracy = (predictions == all_labels).mean()
    
    # Collect for calibration analysis
    engine.monitor.reset()
    engine.monitor.collect_predictions(all_probs, all_labels)
    calib = engine.monitor.compute_calibration()
    
    print(f"\nValidation Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Avg Prob: {all_probs.mean():.4f}")
    if calib is not None:
        print(f"  Calib beta:  {calib['beta']:.4f}")
        print(f"  Calib alpha: {calib['alpha']:.4f}")
    
    return accuracy, all_probs.mean()


def train(hp, train_loader, val_loader, device):
    """
    Complete training loop
    
    Args:
        hp: HyperParameters instance
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: torch device
    
    Returns:
        engine: Trained DerpEngine
        history: Training history
    """
    # Initialize model
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
    
    # Initialize engine
    engine = DerpEngine(encoder, decoder, hp, device)
    
    print("\n" + "="*60)
    print("DERP-VAE Training")
    print("="*60)
    print(f"Device: {device}")
    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print("="*60 + "\n")
    
    history = []
    
    for epoch in range(hp.epochs):
        print(f"\nEpoch {epoch+1}/{hp.epochs}")
        print("-" * 60)
        
        # Train
        summary = train_epoch(engine, train_loader, epoch)
        summary['epoch'] = epoch
        history.append(summary)
        
        # Validate
        if (epoch + 1) % hp.val_every == 0 or epoch == hp.epochs - 1:
            val_acc, val_prob = validate(engine, val_loader)
            summary['val_accuracy'] = val_acc
            summary['val_prob'] = val_prob
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60 + "\n")
    
    return engine, history


def main():
    parser = argparse.ArgumentParser(description='Train Derp-VAE')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    hp = HyperParameters.finance_low_snr()
    hp.latent_dim = args.latent_dim
    hp.epochs = args.epochs
    hp.batch_size = args.batch_size
    hp.lr = args.lr
    
    # Generate data
    print("Generating synthetic data...")
    images, labels = create_synthetic_data(n_samples=args.n_samples)
    
    # Create dataset
    dataset = TensorDataset(images, labels)
    
    # Split train/val
    val_size = int(len(dataset) * hp.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=hp.batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False
    )
    
    # Train
    engine, history = train(hp, train_loader, val_loader, device)
    
    # Save model
    torch.save({
        'encoder': engine.encoder.state_dict(),
        'decoder': engine.decoder.state_dict(),
        'hp': hp,
        'history': history
    }, 'derp_vae_model.pt')
    
    print("Model saved to derp_vae_model.pt")


if __name__ == "__main__":
    main()
