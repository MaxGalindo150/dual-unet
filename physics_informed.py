import torch as th
import torch.nn as nn
import torch.nn.functional as F

from attention_unet_model import AttentionUNet

# models/forward_model.py
class ForwardModel(nn.Module):
    """Physics-based forward model for PAT"""
    def __init__(self, in_channels=1, hidden_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, 3, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

# models/uncertainty.py
class UncertaintyEstimator(nn.Module):
    """Uncertainty estimation module"""
    def __init__(self, in_channels=1):
        super().__init__()
        self.uncertainty = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        return F.sigmoid(self.uncertainty(x))

# models/physics_informed.py
class PhysicsInformedWrapper(nn.Module):
    """Wrapper that adds physics-informed capabilities to any reconstruction model"""
    def __init__(self, base_model, forward_model=None, uncertainty_estimator=None):
        super().__init__()
        self.base_model = base_model
        self.forward_model = forward_model or ForwardModel()
        self.uncertainty_estimator = uncertainty_estimator or UncertaintyEstimator()
    
    def forward(self, x):
        # Base model reconstruction
        refined = self.base_model(x)
        
        # Physics forward simulation
        simulated = self.forward_model(refined)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_estimator(refined)
        
        return refined, simulated, uncertainty

# losses/pat_loss.py
class PATLoss(nn.Module):
    """Combined loss for physics-informed reconstruction"""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.image_loss = nn.L1Loss()
        self.physics_loss = nn.MSELoss()
    
    def forward(self, outputs, targets):
        pred_img, simulated, uncertainty = outputs
        gt_img, real_signal = targets
        
        img_loss = self.image_loss(pred_img, gt_img)
        physics_loss = self.physics_loss(simulated, real_signal)
        
        total_loss = self.alpha * img_loss + (1-self.alpha) * physics_loss
        
        return total_loss, img_loss, physics_loss

# utils/testing.py
def test_model(model, img_size=128, batch_size=2, device='cuda'):
    """Comprehensive model testing"""
    print(f"Using device: {device}")
    model = model.to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test dimensions
    x = th.randn((batch_size, 1, img_size, img_size)).to(device)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass test
    try:
        refined, simulated, uncertainty = model(x)
        print("\nOutput shapes:")
        print(f"- Refined image: {refined.shape}")
        print(f"- Simulated signal: {simulated.shape}")
        print(f"- Uncertainty map: {uncertainty.shape}")
        print("\n✓ Forward pass successful")
        
        # Test loss calculation
        gt_img = th.randn_like(refined)
        real_signal = th.randn_like(simulated)
        
        criterion = PATLoss()
        total_loss, img_loss, physics_loss = criterion(
            (refined, simulated, uncertainty),
            (gt_img, real_signal)
        )
        
        print("\nLoss values:")
        print(f"- Total loss: {total_loss.item():.4f}")
        print(f"- Image loss: {img_loss.item():.4f}")
        print(f"- Physics loss: {physics_loss.item():.4f}")
        print("\n✓ Loss calculation successful")
        
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
    
    finally:
        # Memory cleanup
        if hasattr(th.cuda, 'empty_cache'):
            th.cuda.empty_cache()

# Usage example
if __name__ == '__main__':
    # Assuming AttentionUNet is imported from your existing code
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    
    # Create base model
    base_model = AttentionUNet(in_channels=1, out_channels=1)
    
    # Wrap it with physics-informed capabilities
    model = PhysicsInformedWrapper(base_model)
    
    # Test the model
    test_model(model, device=device)