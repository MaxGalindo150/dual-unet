import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_dct import dct, idct



from attention_unet_model import AttentionUNet

def nextpow2(x):
    return int(np.ceil(np.log2(np.abs(x))))

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
    
class PhysicsForwardModel(nn.Module):
    def __init__(self, in_channels=1, Nz=128, Nx=128, c0=1):
        super().__init__()
        self.Nz = Nz
        self.Nx = Nx
        self.c0 = c0
        
        # Parámetros físicos aprendibles
        self.tau = nn.Parameter(th.tensor(77e-12))  # Tiempo de relajación
        self.chi = nn.Parameter(th.tensor(3e-2))    # Parámetro de atenuación
        
    def compute_propagation(self, W, c0=1, sigma=1):
        Nz, Nx = W.shape
        Z = np.zeros(W.shape)
        ZP_col = np.zeros((Nz, 2**(nextpow2(2*Nx) + 1) - 2*Nx))
        ZP_row = np.zeros((2**(nextpow2(2*Nz) + 1) - 2*Nz, 2**(nextpow2(2*Nx) + 1)))
        
        P = np.block([[Z, W, ZP_col], [Z, Z, ZP_col], [ZP_row]])
        Ly, Lx = P.shape
        
        kx = (np.arange(Lx) / Lx) * np.pi
        ky = (np.arange(Ly) / Ly) * np.pi
        
        f = np.zeros((Lx, Ly))
        for kxi in range(Lx):
            for kyi in range(Ly):
                f[kxi, kyi] = np.sqrt(kx[kxi]**2 + ky[kyi]**2)
        
        Pdet = np.zeros((P.shape[0], len(f)))
        P_hat = dct(dct(P.T, norm='ortho').T, norm='ortho')
        
        for t in range(P.shape[0]):
            cost = np.cos(c0 * f * t)
            Pcos = P_hat * cost.T
            Pt = idct(idct(Pcos.T, norm='ortho').T, norm='ortho') / 3
            Pdet[t, :] = Pt[0, :]
        
        noise = np.random.normal(0, sigma * np.std(Pdet), Pdet.shape)
        P_surf = Pdet + noise
        return P_surf
            
    def forward(self, x):
        batch_size = x.shape[0]
        output = []
        
        # Procesar cada imagen en el batch
        for i in range(batch_size):
            prop = self.compute_propagation(x[i, 0])  # Asume 1 canal
            output.append(prop.unsqueeze(0))
            
            
        return th.stack(output)

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
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.image_loss = nn.MSELoss()
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