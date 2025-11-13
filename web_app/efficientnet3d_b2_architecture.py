"""
EfficientNet3D-B2 Architecture for Lung Cancer Detection
Adapted for 3D Medical Imaging with Compound Scaling
"""

import torch
import torch.nn as nn
import math


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation3D(nn.Module):
    """Squeeze-and-Excitation block for 3D tensors"""
    def __init__(self, in_channels, reduction_ratio=4):
        super(SqueezeExcitation3D, self).__init__()
        reduced_dim = max(1, in_channels // reduction_ratio)
        
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Conv3d(in_channels, reduced_dim, kernel_size=1, bias=True),
            Swish(),
            nn.Conv3d(reduced_dim, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale


class StochasticDepth(nn.Module):
    """Stochastic Depth: randomly drop entire blocks during training"""
    def __init__(self, survival_prob=0.8):
        super(StochasticDepth, self).__init__()
        self.survival_prob = survival_prob
    
    def forward(self, x):
        if not self.training or self.survival_prob == 1.0:
            return x
        
        batch_size = x.size(0)
        random_tensor = self.survival_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1, 1], dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        output = x / self.survival_prob * binary_tensor
        return output


class MBConv3D(nn.Module):
    """Mobile Inverted Bottleneck Convolution for 3D"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 expand_ratio=6, se_ratio=0.25, survival_prob=0.8):
        super(MBConv3D, self).__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        hidden_dim = int(in_channels * expand_ratio)
        padding = kernel_size // 2
        
        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv3d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                Swish()
            )
        else:
            self.expand_conv = nn.Identity()
            hidden_dim = in_channels
        
        # Depthwise convolution
        self.dwconv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            Swish()
        )
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            self.se = SqueezeExcitation3D(hidden_dim, reduction_ratio=int(1/se_ratio))
        else:
            self.se = nn.Identity()
        
        # Projection phase
        self.project = nn.Sequential(
            nn.Conv3d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        
        # Stochastic depth
        if self.use_residual:
            self.stochastic_depth = StochasticDepth(survival_prob)
        else:
            self.stochastic_depth = nn.Identity()
    
    def forward(self, x):
        identity = x
        
        out = self.expand_conv(x)
        out = self.dwconv(out)
        out = self.se(out)
        out = self.project(out)
        
        if self.use_residual:
            out = self.stochastic_depth(out)
            out = out + identity
        
        return out


class EfficientNet3D_B2(nn.Module):
    """EfficientNet-B2 adapted for 3D medical imaging
    
    Configuration:
    - Width multiplier: 1.1
    - Depth multiplier: 1.1
    - Resolution: 260 (scaled to 64 for 3D)
    - Dropout: 0.3
    """
    
    def __init__(self, in_channels=1, num_classes=2, width_mult=1.1, depth_mult=1.1, dropout_rate=0.3):
        super(EfficientNet3D_B2, self).__init__()
        
        # Block configuration: [expand_ratio, channels, num_blocks, stride, kernel_size]
        block_configs = [
            [1,  16,  2, 1, 3],  # Stage 1
            [6,  24,  3, 2, 3],  # Stage 2
            [6,  40,  3, 2, 5],  # Stage 3
            [6,  80,  4, 2, 3],  # Stage 4
            [6, 112,  4, 1, 5],  # Stage 5
            [6, 192,  5, 2, 5],  # Stage 6
            [6, 320,  2, 1, 3],  # Stage 7
        ]
        
        # Scaling functions
        def scale_width(channels):
            channels = int(channels * width_mult)
            return int((channels + 4) // 8 * 8)
        
        def scale_depth(num_blocks):
            return int(math.ceil(num_blocks * depth_mult))
        
        # Stem
        stem_channels = scale_width(32)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(stem_channels),
            Swish()
        )
        
        # Build MBConv blocks
        self.blocks = nn.ModuleList()
        in_ch = stem_channels
        total_blocks = sum([scale_depth(cfg[2]) for cfg in block_configs])
        block_idx = 0
        
        for expand_ratio, channels, num_blocks, stride, kernel_size in block_configs:
            out_ch = scale_width(channels)
            num_blocks = scale_depth(num_blocks)
            
            for i in range(num_blocks):
                survival_prob = 1.0 - (block_idx / total_blocks) * 0.2
                
                block = MBConv3D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                    se_ratio=0.25,
                    survival_prob=survival_prob
                )
                self.blocks.append(block)
                in_ch = out_ch
                block_idx += 1
        
        # Head
        head_channels = scale_width(1280)
        self.head = nn.Sequential(
            nn.Conv3d(in_ch, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(head_channels),
            Swish()
        )
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(head_channels, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
