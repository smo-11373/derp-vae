"""
Monitoring Module for Derp-VAE
Tracks health metrics: MSI, calibration beta, ergodicity
"""

import torch
import numpy as np
from sklearn.linear_model import LinearRegression


class Monitor:
    """
    Monitor for tracking Derp-VAE health metrics
    
    Tracks:
    - Manifold Stability Index (MSI)
    - Calibration slope (beta)
    - Ergodicity measures
    - Recovery statistics
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulators"""
        self.z0_list = []
        self.z1_list = []
        self.img0_list = []
        self.img1_list = []
        self.px0_list = []
        self.px1_list = []
        self.labels_list = []
        self.probs_list = []
        
        self.losses = {
            'total': [],
            'img': [],
            'kl': [],
            'class': [],
            'rec': [],
            'derp': [],
            'rp_z0': [],
            'rp_z1': [],
            'rp_dz': [],
            'rp_px': []
        }
    
    def collect_batch(self, z0, z1, img0, img1, px0, px1, labels, losses):
        """
        Collect data from a training batch
        
        Args:
            z0, z1: (batch, latent_dim)
            img0, img1: (batch, 1, H, W)
            px0, px1: (batch, 1)
            labels: (batch,)
            losses: dict with loss components
        """
        self.z0_list.append(z0.detach().cpu())
        self.z1_list.append(z1.detach().cpu())
        self.img0_list.append(img0.detach().cpu())
        self.img1_list.append(img1.detach().cpu())
        self.px0_list.append(px0.detach().cpu())
        self.px1_list.append(px1.detach().cpu())
        self.labels_list.append(labels.detach().cpu())
        
        for key, val in losses.items():
            self.losses[key].append(val)
    
    def collect_predictions(self, probs, labels):
        """
        Collect predictions for calibration analysis
        
        Args:
            probs: (batch,) predicted probabilities
            labels: (batch,) true labels
        """
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        self.probs_list.extend(probs.flatten())
        self.labels_list.extend(labels.flatten())
    
    def compute_msi(self):
        """
        Compute Manifold Stability Index
        
        MSI = 1 - Var(z1 - z0) / Var(z0)
        
        Target: > 0.80
        Interpretation:
        - MSI ≈ 1: Perfect recovery
        - MSI < 0.60: Manifold collapse
        
        Returns:
            msi: scalar
        """
        if len(self.z0_list) == 0:
            return None
        
        z0 = torch.cat(self.z0_list, dim=0)
        z1 = torch.cat(self.z1_list, dim=0)
        
        dz = z1 - z0
        
        var_dz = torch.var(dz)
        var_z0 = torch.var(z0)
        
        msi = 1.0 - (var_dz / (var_z0 + 1e-8))
        
        return msi.item()
    
    def compute_rmse_stats(self):
        """
        Compute RMSE statistics for adaptive scaling
        
        Returns:
            dict with z_rmse, img_rmse, px_rmse
        """
        if len(self.z0_list) == 0:
            return None
        
        z0 = torch.cat(self.z0_list, dim=0)
        z1 = torch.cat(self.z1_list, dim=0)
        img0 = torch.cat(self.img0_list, dim=0)
        img1 = torch.cat(self.img1_list, dim=0)
        px0 = torch.cat(self.px0_list, dim=0)
        px1 = torch.cat(self.px1_list, dim=0)
        
        # Compute differences
        z_diff = z1 - z0
        img_diff = img1 - img0
        px_diff = px1 - px0
        
        # Compute RMSE (std across samples, mean across dimensions)
        z_rmse = torch.std(z_diff, dim=0).mean().item()
        img_rmse = torch.std(img_diff.flatten(1), dim=0).mean().item()
        px_rmse = torch.std(px_diff, dim=0).mean().item()
        
        return {
            'z_rmse': z_rmse,
            'img_rmse': img_rmse,
            'px_rmse': px_rmse
        }
    
    def compute_calibration(self):
        """
        Compute calibration slope (beta) and intercept (alpha)
        
        Fit: labels ~ beta * probs + alpha
        
        Target: 0.10 < beta < 0.50 for finance
        Interpretation:
        - beta ≈ 1: Perfect calibration
        - beta < 1: Shrinkage (common in low-SNR)
        - beta ≈ 0: No signal
        
        Returns:
            dict with beta, alpha, r2
        """
        if len(self.probs_list) < 10:
            return None
        
        probs = np.array(self.probs_list).reshape(-1, 1)
        labels = np.array(self.labels_list)
        
        # Linear regression
        reg = LinearRegression().fit(probs, labels)
        
        beta = reg.coef_[0]
        alpha = reg.intercept_
        r2 = reg.score(probs, labels)
        
        return {
            'beta': beta,
            'alpha': alpha,
            'r2': r2
        }
    
    def compute_px_separation(self):
        """
        Compute separation between class modes in px1 distribution
        
        Returns:
            separation: scalar (mean_class1 - mean_class0)
        """
        if len(self.px1_list) == 0 or len(self.labels_list) == 0:
            return None
        
        px1 = torch.cat(self.px1_list, dim=0).squeeze()
        labels = torch.cat(self.labels_list, dim=0)
        
        # Mean px for each class
        px_class0 = px1[labels == 0].mean()
        px_class1 = px1[labels == 1].mean()
        
        separation = (px_class1 - px_class0).item()
        
        return separation
    
    def get_summary(self):
        """
        Get comprehensive summary of health metrics
        
        Returns:
            dict with all metrics
        """
        summary = {}
        
        # Loss statistics
        for key, vals in self.losses.items():
            if len(vals) > 0:
                summary[f'loss_{key}_mean'] = np.mean(vals)
                summary[f'loss_{key}_std'] = np.std(vals)
        
        # MSI
        msi = self.compute_msi()
        if msi is not None:
            summary['msi'] = msi
            summary['msi_healthy'] = msi > 0.80
        
        # RMSE
        rmse = self.compute_rmse_stats()
        if rmse is not None:
            summary.update(rmse)
        
        # Calibration
        calib = self.compute_calibration()
        if calib is not None:
            summary.update(calib)
            summary['beta_healthy'] = 0.05 < calib['beta'] < 0.50
        
        # px separation
        sep = self.compute_px_separation()
        if sep is not None:
            summary['px_separation'] = sep
            summary['px_sep_healthy'] = abs(sep) > 0.02
        
        return summary
    
    def print_summary(self, epoch=None):
        """Print formatted summary"""
        summary = self.get_summary()
        
        if epoch is not None:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} Summary")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print("Training Summary")
            print(f"{'='*60}")
        
        # Losses
        print("\nLosses:")
        if 'loss_total_mean' in summary:
            print(f"  Total:  {summary['loss_total_mean']:.4f} +/- {summary.get('loss_total_std', 0):.4f}")
        if 'loss_img_mean' in summary:
            print(f"  Image:  {summary['loss_img_mean']:.4f}")
        if 'loss_kl_mean' in summary:
            print(f"  KL:     {summary['loss_kl_mean']:.4f}")
        if 'loss_class_mean' in summary:
            print(f"  Class:  {summary['loss_class_mean']:.4f}")
        if 'loss_rec_mean' in summary:
            print(f"  Recov:  {summary['loss_rec_mean']:.4f}")
        if 'loss_derp_mean' in summary:
            print(f"  DERP:   {summary['loss_derp_mean']:.4f}")
        
        # Health metrics
        print("\nHealth Metrics:")
        if 'msi' in summary:
            status = "[OK]" if summary.get('msi_healthy', False) else "[WARN]"
            print(f"  MSI:         {summary['msi']:.4f} {status} (target: > 0.80)")

        if 'beta' in summary:
            status = "[OK]" if summary.get('beta_healthy', False) else "[WARN]"
            print(f"  Calib beta:  {summary['beta']:.4f} {status} (target: 0.05-0.50)")

        if 'px_separation' in summary:
            status = "[OK]" if summary.get('px_sep_healthy', False) else "[WARN]"
            print(f"  px Sep:      {summary['px_separation']:.4f} {status} (target: > 0.02)")
        
        # RMSE
        if 'z_rmse' in summary:
            print("\nAdaptive Scales (RMSE):")
            print(f"  z_rmse:      {summary['z_rmse']:.4f}")
            print(f"  img_rmse:    {summary['img_rmse']:.4f}")
            print(f"  px_rmse:     {summary['px_rmse']:.4f}")
        
        print(f"{'='*60}\n")
        
        return summary
