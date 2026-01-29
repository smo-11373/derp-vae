"""
Decoder Module for Derp-VAE
Maps z -> (Timg, Tpx) which are parameters defining distribution of x
"""

import torch
import torch.nn as nn


class DerpDecoder(nn.Module):
    """
    Decoder: Maps latent z to parameter space (Timg, Tpx)
    
    Input:
        - z: (batch, latent_dim) latent features
    
    Output:
        - Timg: (batch, 1, H, W) image parameters
        - Tpx: (batch, 1) label probability parameter in [0, 1]
    """
    
    def __init__(self, latent_dim=64, img_size=(39, 39), hidden_dim=512):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.img_dim = img_size[0] * img_size[1]
        
        # Shared decoder backbone
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        
        # Image head (unbounded continuous values)
        self.img_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.img_dim)
        )
        
        # Label probability head (bounded to [0, 1])
        self.px_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in [0, 1]
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
    
    def forward(self, z):
        """
        Forward pass
        
        Args:
            z: (batch, latent_dim)
        
        Returns:
            Timg: (batch, 1, H, W) - image parameters
            Tpx: (batch, 1) - label probability parameter
        """
        batch_size = z.size(0)
        
        # Shared features
        h = self.shared(z)
        
        # Image parameters
        Timg = self.img_head(h)
        Timg = Timg.view(batch_size, 1, self.img_size[0], self.img_size[1])
        
        # Label probability parameter
        Tpx = self.px_head(h)
        
        return Timg, Tpx
    
    def decode_with_noise(self, z, img_scale, px_scale):
        """
        Decode and add noise to generate x ~ p(x|z)
        
        Args:
            z: (batch, latent_dim)
            img_scale: scalar or tensor for image noise
            px_scale: scalar or tensor for px noise
        
        Returns:
            img: (batch, 1, H, W) - sampled image
            px: (batch, 1) - sampled probability
        """
        Timg, Tpx = self.forward(z)
        
        # Add Gaussian noise
        img = Timg + torch.randn_like(Timg) * img_scale
        px = Tpx + torch.randn_like(Tpx) * px_scale
        
        # Clip px to valid range
        px = torch.clamp(px, 0.0, 1.0)
        
        return img, px
