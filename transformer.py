import torch
import torch.nn as nn

class CNNTransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        # Convolutional projections for query, key, value
        self.query_conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.key_conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(channels, num_heads)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, kernel_size=1)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        # Save input for residual connection
        residual = x

        # Convolutional projections
        query = self.query_conv(x).flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        key = self.key_conv(x).flatten(2).permute(2, 0, 1)      # (H*W, B, C)
        value = self.value_conv(x).flatten(2).permute(2, 0, 1)  # (H*W, B, C)

        # Multi-head attention
        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = attn_output.permute(1, 2, 0).view_as(x)  # (B, C, H, W)

        # Add residual and normalize
        x = self.norm1((x + attn_output).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm2((x + ffn_output).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return x

class CNNTransformerReconstruction(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, num_heads):
        super().__init__()
        # Initial convolution (no downsampling)
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            CNNTransformerBlock(64, num_heads) for _ in range(num_layers)
        ])

        # Final convolution to produce output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final convolution
        x = self.final_conv(x)
        return x

# Example usage
model = CNNTransformerReconstruction(in_channels=1, out_channels=1, num_layers=4, num_heads=8)
x = torch.randn(1, 1, 128, 128)  # Input shape: (1, 1, 128, 128)
y = model(x)
print(y.shape)  # Output shape: (1, 1, 128, 128)