import torch as th
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Attention Gate
        F_g: número de canales en las features del gating signal (decoder)
        F_l: número de canales en las features del input signal (encoder)
        F_int: número de canales en las features intermedias
        """
        super().__init__()
        
        self.Wg = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.Wx = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

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

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Encoder path
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Attention Gates
        self.attention4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.attention3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.attention2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.attention1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        
        # Decoder path
        self.dec4 = DoubleConv(1024, 512)
        self.dec3 = DoubleConv(512, 256)
        self.dec2 = DoubleConv(256, 128)
        self.dec1 = DoubleConv(128, 64)
        
        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2)
        
        # Upsampling
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    
    def forward(self, x):

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with attention
        dec4_up = self.up4(bottleneck)
        dec4_att = self.attention4(dec4_up, enc4)
        dec4 = self.dec4(th.cat([dec4_up, dec4_att], dim=1))
        dec3_up = self.up3(dec4)
        dec3_att = self.attention3(dec3_up, enc3)
        dec3 = self.dec3(th.cat([dec3_up, dec3_att], dim=1))
        dec2_up = self.up2(dec3)
        dec2_att = self.attention2(dec2_up, enc2)
        dec2 = self.dec2(th.cat([dec2_up, dec2_att], dim=1))
        dec1_up = self.up1(dec2)
        dec1_att = self.attention1(dec1_up, enc1)
        dec1 = self.dec1(th.cat([dec1_up, dec1_att], dim=1))
        # Final convolution
        out = self.final_conv(dec1)
        return out
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model = AttentionUNet().to(device)
    print("Model parameters: ", count_parameters(model))
    
    # test forward
    x = th.randn((1, 1, 128, 128)).to(device)
    print("Input shape: ", x.shape)
    y = model(x)
    print("Output shape: ", y.shape)