"""
Encoder Module for Derp-VAE
Maps (img, px) -> (m, s) where z ~ N(m, s²·I)
"""

import torch
import torch.nn as nn


class DerpEncoder(nn.Module):
    """
    Encoder: Maps (img, px) to latent distribution parameters (m, s)
    
    Input: 
        - img: (batch, 1, H, W) image features
        - px: (batch, 1) label probability parameter
    
    Output:
        - m: (batch, latent_dim) mean of latent distribution
        - s: (batch, latent_dim) standard deviation of latent distribution
    """
    
    def __init__(self, img_size=(39, 39), latent_dim=64, hidden_dim=512):
        super().__init__()
        
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.input_dim = img_size[0] * img_size[1] + 1  # Flattened img + px
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, latent_dim * 2)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, img, px):
        """
        Forward pass
        
        Args:
            img: (batch, 1, H, W) or (batch, H*W)
            px: (batch, 1)
        
        Returns:
            m: (batch, latent_dim) - mean
            s: (batch, latent_dim) - standard deviation (positive)
        """
        batch_size = img.size(0)
        
        # Flatten image if needed
        if img.dim() == 4:
            img_flat = img.view(batch_size, -1)
        else:
            img_flat = img
        
        # Concatenate image features and px parameter
        x = torch.cat([img_flat, px], dim=1)
        
        # Forward through network
        out = self.net(x)
        
        # Split into mean and log-variance
        m, logvar = torch.chunk(out, 2, dim=1)
        
        # Convert log-variance to standard deviation
        # s = exp(0.5 * logvar) with numerical stability
        s = torch.exp(0.5 * logvar.clamp(-10, 10))
        
        return m, s
    
    def encode(self, img, px):
        """
        Encode and sample z ~ N(m, s²·I)
        
        Returns:
            z: (batch, latent_dim) - sampled latent
            m: (batch, latent_dim) - mean
            s: (batch, latent_dim) - std
        """
        m, s = self.forward(img, px)
        z = m + s * torch.randn_like(m)
        return z, m, s
