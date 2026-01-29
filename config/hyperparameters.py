"""
Hyperparameters for Derp-VAE
Complete configuration for quantitative finance applications
"""

from dataclasses import dataclass


@dataclass
class HyperParameters:
    """Complete hyperparameter specification for Derp-VAE"""
    
    # ==================== Architecture ====================
    latent_dim: int = 64
    """Dimension of latent space z"""
    
    hidden_dim: int = 512
    """Hidden layer dimension for encoder/decoder"""
    
    img_size: tuple = (39, 39)
    """Input image dimensions"""
    
    # ==================== Bimodal Prior ====================
    pxBias: float = 0.05
    """Distance of class peaks from 0.5 center (peaks at 0.45 and 0.55)"""
    
    pxStdev: float = 0.10
    """Standard deviation of each bimodal mode - wider allows stable likelihoods near 0.5"""
    
    # ==================== Adaptive Scales ====================
    Rscale: float = 0.9
    """Recovery scale multiplier for RMSE (typically 0.9)"""
    
    Gscale: float = 0.1
    """Generative noise scale multiplier (typically 0.1)"""
    
    # Initial RMSE values (will be updated during training)
    init_img_rmse: float = 0.5
    init_z_rmse: float = 0.5
    init_px_rmse: float = 0.1
    
    # ==================== Random Probe (DERP) ====================
    nRP: int = 5
    """Number of random projections for Cramér-Wold testing"""
    
    lambda_rp: float = 1.0
    """Weight for Random Probe loss in total loss"""
    
    # ==================== Training ====================
    lr: float = 1e-3
    """Learning rate"""
    
    batch_size: int = 64
    """Batch size for training"""
    
    epochs: int = 100
    """Number of training epochs"""
    
    grad_clip: float = 5.0
    """Gradient clipping threshold"""
    
    # ==================== Prediction ====================
    S_pred: int = 8
    """Number of importance samples for prediction"""
    
    # ==================== Monitoring ====================
    msi_threshold: float = 0.80
    """Manifold Stability Index threshold - below this indicates collapse"""
    
    beta_min: float = 0.05
    """Minimum calibration beta - below this indicates no signal"""
    
    beta_max: float = 0.50
    """Maximum calibration beta for healthy model"""
    
    # ==================== Validation ====================
    val_split: float = 0.2
    """Fraction of data for validation"""
    
    val_every: int = 5
    """Validate every N epochs"""
    
    def __post_init__(self):
        """Validate hyperparameters"""
        assert 0 < self.pxBias < 0.5, "pxBias must be in (0, 0.5)"
        assert self.pxStdev > 0, "pxStdev must be positive"
        assert 0 < self.Rscale <= 1.0, "Rscale should be in (0, 1]"
        assert self.nRP >= 3, "Need at least 3 random probes"
        assert self.lambda_rp > 0, "lambda_rp must be positive"
        
    @classmethod
    def finance_low_snr(cls):
        """Preset for low-SNR quantitative finance"""
        return cls(
            latent_dim=64,
            pxBias=0.05,
            pxStdev=0.10,
            Rscale=0.9,
            Gscale=0.1,
            lambda_rp=1.0,
            S_pred=8
        )
    
    @classmethod
    def finance_high_freq(cls):
        """Preset for high-frequency trading (tighter constraints)"""
        return cls(
            latent_dim=32,
            pxBias=0.03,
            pxStdev=0.08,
            Rscale=0.95,
            Gscale=0.05,
            lambda_rp=2.0,
            S_pred=16
        )
