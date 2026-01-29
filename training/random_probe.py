"""
Random Probe (RP) for distributional testing
Implements Cramér-Wold theorem via random 1D projections
"""

import torch
import numpy as np


class RandomProbe:
    """
    Random Probe for testing high-dimensional normality
    Uses Cramér-Wold theorem: if all 1D projections are Gaussian,
    then the multivariate distribution is Gaussian.
    """
    
    def __init__(self, n_probes=5):
        """
        Args:
            n_probes: Number of random projections to test
        """
        self.n_probes = n_probes
    
    def get_random_projections(self, z, n_probes=None):
        """
        Generate random 1D projections of z
        
        Args:
            z: (batch, dim) tensor to project
            n_probes: Number of probes (default: self.n_probes)
        
        Returns:
            projs: (batch, n_probes) - random 1D projections
        """
        if n_probes is None:
            n_probes = self.n_probes
        
        batch_size, dim = z.shape
        device = z.device
        
        # Generate random unit vectors
        theta = torch.randn(dim, n_probes, device=device)
        theta = theta / (torch.norm(theta, dim=0, keepdim=True) + 1e-8)
        
        # Project: proj_i = <z, theta_i>
        projs = z @ theta  # (batch, n_probes)
        
        return projs
    
    def ks_test(self, samples):
        """
        Modified Kolmogorov-Smirnov test (average-based for differentiability)
        Tests if samples ~ N(0,1)
        
        Args:
            samples: (batch,) tensor of 1D samples
        
        Returns:
            ks_dist: scalar KS distance
        """
        n = samples.shape[0]
        
        if n < 2:
            return torch.tensor(0.0, device=samples.device)
        
        # Sort samples
        sorted_samples, _ = torch.sort(samples)
        
        # Empirical CDF: F_emp(x_i) = i/n
        i = torch.arange(1, n + 1, device=samples.device, dtype=samples.dtype)
        F_emp = i / n
        
        # Theoretical CDF for N(0,1)
        F_theory = 0.5 * (1 + torch.erf(sorted_samples / np.sqrt(2)))
        
        # Average-based distance (differentiable, unlike max)
        diff = torch.abs(F_emp - F_theory)
        avg_dist = diff.mean() * np.sqrt(n)
        
        return avg_dist
    
    def test_normality(self, z):
        """
        Test if z ~ N(0,I) via random projections
        
        Args:
            z: (batch, dim) tensor to test
        
        Returns:
            loss: scalar - average KS distance across probes
        """
        if z.dim() == 1:
            z = z.unsqueeze(1)
        
        batch_size, dim = z.shape
        
        if batch_size < 2:
            return torch.tensor(0.0, device=z.device)
        
        if dim == 1:
            # 1D case: test directly
            return self.ks_test(z.squeeze())
        
        # Multi-dimensional: use random projections
        projs = self.get_random_projections(z, self.n_probes)
        
        # Test each projection
        ks_distances = []
        for i in range(self.n_probes):
            proj_i = projs[:, i]
            ks_dist = self.ks_test(proj_i)
            ks_distances.append(ks_dist)
        
        # Average KS distance across probes
        avg_ks = sum(ks_distances) / len(ks_distances)
        
        return avg_ks
    
    def test_bimodal(self, px, px_bias=0.05, px_stdev=0.10):
        """
        Test if px follows bimodal distribution with peaks at 0.5 ± px_bias
        
        Uses EMD (Earth Mover's Distance) / Wasserstein distance approximation
        
        Args:
            px: (batch, 1) or (batch,) probability values
            px_bias: Distance of peaks from 0.5
            px_stdev: Standard deviation of each mode
        
        Returns:
            loss: scalar - distance from bimodal distribution
        """
        if px.dim() == 2:
            px = px.squeeze(1)
        
        batch_size = px.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=px.device)
        
        # Sort observed values
        px_sorted, _ = torch.sort(px)
        
        # Generate reference bimodal samples
        # Half from mode 0, half from mode 1
        n_half = batch_size // 2
        
        mode0 = torch.randn(n_half, device=px.device) * px_stdev + (0.5 - px_bias)
        mode1 = torch.randn(batch_size - n_half, device=px.device) * px_stdev + (0.5 + px_bias)
        
        ref_samples = torch.cat([mode0, mode1])
        ref_samples = torch.clamp(ref_samples, 0.0, 1.0)
        ref_sorted, _ = torch.sort(ref_samples)
        
        # 1D Wasserstein distance (sorted samples)
        emd = torch.mean(torch.abs(px_sorted - ref_sorted))
        
        return emd


def compute_derp_loss(z0, z1, px1, dz, model_scales, rp, hp):
    """
    Compute complete DERP (Random Probe) loss
    
    Tests:
    - z0 ~ N(0,I)
    - z1 ~ N(0,I)
    - dz ~ N(0,I)
    - px1 ~ Bimodal
    
    Args:
        z0: (batch, latent_dim)
        z1: (batch, latent_dim)
        px1: (batch, 1)
        dz: (batch, latent_dim) - normalized recovery error
        model_scales: dict with 'z_rmse', 'px_rmse'
        rp: RandomProbe instance
        hp: HyperParameters instance
    
    Returns:
        total_loss: scalar
        metrics: dict with individual losses
    """
    # Test marginal normality of z0
    loss_z0 = rp.test_normality(z0)
    
    # Test marginal normality of z1 (ergodicity)
    loss_z1 = rp.test_normality(z1)
    
    # Test marginal normality of dz (recovery)
    loss_dz = rp.test_normality(dz)
    
    # Test bimodal distribution of px1 (ergodicity)
    loss_px = rp.test_bimodal(px1, hp.pxBias, hp.pxStdev)
    
    # Total DERP loss
    total_loss = loss_z0 + loss_z1 + loss_dz + loss_px
    
    metrics = {
        'rp_z0': loss_z0.item(),
        'rp_z1': loss_z1.item(),
        'rp_dz': loss_dz.item(),
        'rp_px': loss_px.item(),
    }
    
    return total_loss, metrics
