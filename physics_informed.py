import torch as th
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

# 1. Bloque Squeeze-and-Excitation (SE)
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y  # Recalibración de canales

# 2. Bloque Self-Attention para el cuello de botella
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(th.zeros(1))  # Parámetro aprendible
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Proyecciones
        q = self.query(x).view(batch_size, -1, H*W).permute(0, 2, 1)  # (B, N, C//8)
        k = self.key(x).view(batch_size, -1, H*W)  # (B, C//8, N)
        v = self.value(x).view(batch_size, -1, H*W)  # (B, C, N)
        
        # Atención
        energy = th.bmm(q, k)  # (B, N, N)
        attention = th.softmax(energy, dim=-1)
        
        # Aplicar atención
        out = th.bmm(v, attention.permute(0, 2, 1))  # (B, C, N)
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x  # Conexión residual

class SA_UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Cuello de botella con Self-Attention
        self.bottleneck = DoubleConv(512, 1024)
        self.self_attn = SelfAttentionBlock(1024)
        
        # Decoder
        self.dec4 = DoubleConv(1024, 512)  # +512 por el skip connection
        self.dec3 = DoubleConv(512, 256)
        self.dec2 = DoubleConv(256, 128)
        self.dec1 = DoubleConv(128, 64)
        
        # Bloques SE en los skip connections
        self.se_enc1 = SEBlock(64)
        self.se_enc2 = SEBlock(128)
        self.se_enc3 = SEBlock(256)
        self.se_enc4 = SEBlock(512)
        
        # Upsampling y capa final
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc1_se = self.se_enc1(enc1)  # SE
        
        enc2 = self.enc2(self.pool(enc1))
        enc2_se = self.se_enc2(enc2)
        
        enc3 = self.enc3(self.pool(enc2))
        enc3_se = self.se_enc3(enc3)
        
        enc4 = self.enc4(self.pool(enc3))
        enc4_se = self.se_enc4(enc4)
        
        # Cuello de botella con atención
        bottleneck = self.bottleneck(self.pool(enc4))
        bottleneck = self.self_attn(bottleneck)  # Self-Attention
        
        # Decoder
        dec4 = self.up4(bottleneck)
        dec4 = th.cat([dec4, enc4_se], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = th.cat([dec3, enc3_se], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = th.cat([dec2, enc2_se], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = th.cat([dec1, enc1_se], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final_conv(dec1)

# Test
if __name__ == '__main__':
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model = SA_UNet().to(device)
    print("Parámetros totales:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    x = th.randn(1, 1, 128, 128).to(device)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")