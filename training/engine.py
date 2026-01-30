"""
Training Engine for Derp-VAE
Complete implementation with ELBO + DERP losses
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .random_probe import RandomProbe, compute_derp_loss
from .monitor import Monitor


class DerpEngine:
    """
    Complete training engine for Derp-VAE
    
    Implements:
    - Full ELBO (reconstruction + complete KL + classification)
    - DERP loss (Random Probe testing)
    - Recovery loss
    - Adaptive scale updates
    - Health monitoring
    """
    
    def __init__(self, encoder, decoder, hp, device='cuda'):
        """
        Args:
            encoder: DerpEncoder instance
            decoder: DerpDecoder instance
            hp: HyperParameters instance
            device: 'cuda' or 'cpu'
        """
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.hp = hp
        self.device = device
        
        # Adaptive scales (initialized, will update during training)
        self.img_rmse = hp.init_img_rmse
        self.z_rmse = hp.init_z_rmse
        self.px_rmse = hp.init_px_rmse
        
        # Random Probe for DERP loss
        self.random_probe = RandomProbe(n_probes=hp.nRP)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=hp.lr
        )
        
        # Monitor
        self.monitor = Monitor()
    
    def sampler(self, img_in, labels):
        """
        Sampler: x_input -> x0
        Creates bimodal px0 based on labels
        
        Args:
            img_in: (batch, 1, H, W)
            labels: (batch,) with values 0 or 1
        
        Returns:
            img0: (batch, 1, H, W)
            px0: (batch, 1)
        """
        batch_size = img_in.size(0)
        
        # Create bimodal px0
        # If label=0: px0 ~ N(0.5 - pxBias, pxStdev²)
        # If label=1: px0 ~ N(0.5 + pxBias, pxStdev²)
        labels = labels.view(-1, 1).float()
        
        px0 = torch.randn(batch_size, 1, device=self.device) * self.hp.pxStdev + \
              torch.where(labels == 1, 
                         0.5 + self.hp.pxBias, 
                         0.5 - self.hp.pxBias)
        
        # Add small noise to img (from x_input to x0)
        img0 = img_in + torch.randn_like(img_in) * (self.hp.Gscale * self.img_rmse * 0.1)
        
        return img0, px0
    
    def forward_chain(self, img_in, labels):
        """
        Complete forward chain: x_input -> x0 -> z0 -> x1 -> z1

        Args:
            img_in: (batch, 1, H, W)
            labels: (batch,)

        Returns:
            dict with all intermediate values
        """
        # 1. Sampler: x_input -> x0
        img0, px0 = self.sampler(img_in, labels)

        # 2. Encoder: x0 -> theta_z0 -> z0
        m0, s0 = self.encoder(img0, px0)
        z0 = m0 + s0 * torch.randn_like(m0)

        # 3. Decoder: z0 -> theta_x
        Timg, Tpx = self.decoder(z0)

        # 4. Sample x1 from theta_x (with noise for manifold volume)
        img1 = Timg + torch.randn_like(Timg) * (self.hp.Gscale * self.img_rmse)
        px1 = Tpx + torch.randn_like(Tpx) * (self.hp.Gscale * self.px_rmse)
        px1 = torch.clamp(px1, 0.0, 1.0)

        # 5. Re-encode noisy x1 to get z1 (consistent for train and predict)
        m1, s1 = self.encoder(img1, px1)
        z1 = m1 + s1 * torch.randn_like(m1)

        return {
            'img0': img0,
            'px0': px0,
            'm0': m0,
            's0': s0,
            'z0': z0,
            'Timg': Timg,
            'Tpx': Tpx,
            'img1': img1,
            'px1': px1,
            'm1': m1,
            's1': s1,
            'z1': z1
        }
    
    def compute_losses(self, chain, img_in, labels):
        """
        Compute all loss components
        
        Args:
            chain: dict from forward_chain
            img_in: (batch, 1, H, W)
            labels: (batch,)
        
        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses
        """
        batch_size = img_in.size(0)
        labels = labels.view(-1, 1).float()
        
        # ========== 1. Image Reconstruction Loss ==========
        e_img = 0.5 * torch.mean(((img_in - chain['Timg']) / self.img_rmse) ** 2)
        
        # ========== 2. Complete KL Divergence ==========
        # KL(N(m0, s0²) || N(0, I)) = 0.5 * (m0² + s0² - 2*ln(s0) - 1)
        kl_loss = 0.5 * torch.mean(
            chain['m0'] ** 2 + 
            chain['s0'] ** 2 - 
            2 * torch.log(chain['s0'] + 1e-8) - 
            1.0
        )
        
        # ========== 3. Classification Loss (Bimodal Likelihood) ==========
        # Compute p(label | px1) using bimodal prior
        phi0 = torch.exp(-0.5 * ((chain['px1'] - (0.5 - self.hp.pxBias)) / self.hp.pxStdev) ** 2)
        phi1 = torch.exp(-0.5 * ((chain['px1'] - (0.5 + self.hp.pxBias)) / self.hp.pxStdev) ** 2)
        
        # p(label=1|px1) = phi1/(phi0+phi1)
        p_label = torch.where(labels == 1,
                             phi1 / (phi0 + phi1 + 1e-8),
                             phi0 / (phi0 + phi1 + 1e-8))
        
        e_class = -torch.mean(torch.log(p_label + 1e-8))
        
        # ========== 4. Recovery Loss ==========
        e_rec = 0.5 * torch.mean(((chain['z0'] - chain['m1']) / self.z_rmse) ** 2)
        
        # ========== 5. DERP Loss (Random Probe) ==========
        # Compute normalized recovery error
        dz = (chain['z1'] - chain['z0']) / (self.z_rmse + 1e-8)
        
        derp_loss, derp_metrics = compute_derp_loss(
            chain['z0'],
            chain['z1'],
            chain['px1'],
            dz,
            {'z_rmse': self.z_rmse, 'px_rmse': self.px_rmse},
            self.random_probe,
            self.hp
        )
        
        # ========== Total Loss ==========
        total_loss = e_img + kl_loss + e_class + e_rec + self.hp.lambda_rp * derp_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'img': e_img.item(),
            'kl': kl_loss.item(),
            'class': e_class.item(),
            'rec': e_rec.item(),
            'derp': derp_loss.item(),
            **derp_metrics
        }
        
        return total_loss, loss_dict
    
    def training_step(self, img_in, labels):
        """
        Single training step
        
        Args:
            img_in: (batch, 1, H, W)
            labels: (batch,)
        
        Returns:
            loss_dict: dict with all losses
        """
        self.encoder.train()
        self.decoder.train()
        
        self.optimizer.zero_grad()
        
        # Forward chain
        chain = self.forward_chain(img_in, labels)
        
        # Compute losses
        total_loss, loss_dict = self.compute_losses(chain, img_in, labels)
        
        # Backward
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            self.hp.grad_clip
        )
        
        self.optimizer.step()
        
        # Collect for monitoring
        self.monitor.collect_batch(
            chain['z0'], chain['z1'],
            chain['img0'], chain['img1'],
            chain['px0'], chain['px1'],
            labels, loss_dict
        )
        
        return loss_dict
    
    def update_scales(self):
        """
        Update adaptive scales based on epoch statistics
        Called at the end of each epoch
        """
        rmse_stats = self.monitor.compute_rmse_stats()
        
        if rmse_stats is not None:
            # Apply Rscale multiplier [equations 6a-c]
            self.z_rmse = max(rmse_stats['z_rmse'] * self.hp.Rscale, 0.01)
            self.img_rmse = max(rmse_stats['img_rmse'] * self.hp.Rscale, 0.01)
            self.px_rmse = max(rmse_stats['px_rmse'] * self.hp.Rscale, 0.001)
    
    def predict_stable(self, img_test):
        """
        Stabilized Bayesian prediction using log-sum-exp.
        Based on [14a] from the document - Full NLL without recovery term.

        Args:
            img_test: (1, 1, H, W) single test image

        Returns:
            prob: scalar probability of label=1
        """
        self.encoder.eval()
        self.decoder.eval()

        log_l_accum = {0: [], 1: []}
        img_gscale = self.hp.Gscale * self.img_rmse
        px_gscale = self.hp.Gscale * self.px_rmse

        with torch.no_grad():
            for r in [0, 1]:
                # Hypothesis mean for px0 [11a]
                m_px0 = 0.5 + (self.hp.pxBias if r == 1 else -self.hp.pxBias)

                for _ in range(self.hp.S_pred):
                    # === 1. x_input => x0 (sampler) [11b-d] ===
                    px0 = torch.randn(1, 1, device=self.device) * self.hp.pxStdev + m_px0
                    px0 = torch.clamp(px0, 0.1, 0.9)
                    img0 = img_test + torch.randn_like(img_test) * img_gscale

                    # === 2. x0 => z0 (encode) [7a] ===
                    m0, s0 = self.encoder(img0, px0)
                    z0 = m0 + s0 * torch.randn_like(m0)

                    # === 3. z0 => theta_x (decode) ===
                    Timg, Tpx = self.decoder(z0)

                    # === 4. theta_x => x1 (sample with noise) [8a-b] ===
                    img1 = Timg + torch.randn_like(Timg) * img_gscale
                    px1 = Tpx + torch.randn_like(Tpx) * px_gscale
                    px1 = torch.clamp(px1, 0.0, 1.0)

                    # === Compute NLL from [14a] ===

                    # Term 1: x_input => x0
                    e_x0_img = 0.5 * torch.sum(((img0 - img_test) / img_gscale) ** 2)
                    e_x0_px = 0.5 * torch.sum(((px0 - m_px0) / self.hp.pxStdev) ** 2)

                    # Term 2: x0 => z0 (encoder)
                    e_enc = 0.5 * torch.sum(((z0 - m0) / s0) ** 2) + torch.sum(torch.log(s0 + 1e-8))

                    # Term 3: z0 prior
                    e_z0 = 0.5 * torch.sum(z0 ** 2)

                    # Term 4: theta_x => x1
                    e_theta_img = 0.5 * torch.sum(((img1 - Timg) / img_gscale) ** 2)
                    e_theta_px = 0.5 * torch.sum(((px1 - Tpx) / px_gscale) ** 2)

                    # Term 5: x1 => x_input (reconstruction + classification)
                    e_img = 0.5 * torch.sum(((img_test - img1) / img_gscale) ** 2)

                    # binaryLoss(label=r, px1) [10b]
                    phi0 = torch.exp(-0.5 * ((px1 - (0.5 - self.hp.pxBias)) / self.hp.pxStdev) ** 2)
                    phi1 = torch.exp(-0.5 * ((px1 - (0.5 + self.hp.pxBias)) / self.hp.pxStdev) ** 2)
                    p_r = phi1 / (phi0 + phi1 + 1e-8) if r == 1 else phi0 / (phi0 + phi1 + 1e-8)
                    e_class = -torch.log(p_r + 1e-8)

                    # Total NLL [14a]
                    nll = e_x0_img + e_x0_px + e_enc + e_z0 + e_theta_img + e_theta_px + e_img + e_class
                    log_l_accum[r].append(-nll)

        # Aggregate using log-sum-exp (prevents underflow)
        l0 = torch.stack(log_l_accum[0])
        l1 = torch.stack(log_l_accum[1])

        log_avg_0 = torch.logsumexp(l0, dim=0) - np.log(self.hp.S_pred)
        log_avg_1 = torch.logsumexp(l1, dim=0) - np.log(self.hp.S_pred)

        # Sigmoid for numerical stability
        prob = torch.sigmoid(log_avg_1 - log_avg_0).item()

        return prob
