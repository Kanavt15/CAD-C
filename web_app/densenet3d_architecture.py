"""
DenseNet3D with Multi-Head Attention Architecture
For Lung Cancer Detection Web Application
"""

import torch
import torch.nn as nn
import math


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class MultiHeadAttention3D(nn.Module):
    """Multi-Head Self-Attention for 3D features"""
    def __init__(self, channels, num_heads=4, dropout=0.1):
        super(MultiHeadAttention3D, self).__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"
        
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # B, 3*C, D, H, W
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, D * H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # 3, B, heads, D*H*W, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Combine
        out = (attn @ v).transpose(2, 3).reshape(B, C, D, H, W)
        out = self.proj(out)
        
        return out


class DenseLayer3D(nn.Module):
    """Dense layer with BN-ReLU-Conv pattern"""
    def __init__(self, in_channels, growth_rate, drop_path_rate=0.0):
        super(DenseLayer3D, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, growth_rate * 4, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm3d(growth_rate * 4)
        self.conv2 = nn.Conv3d(growth_rate * 4, growth_rate, kernel_size=3, 
                               padding=1, bias=False)
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.drop_path(out)
        return torch.cat([x, out], 1)  # Concatenate instead of add


class DenseBlock3D(nn.Module):
    """Dense block with multiple dense layers"""
    def __init__(self, num_layers, in_channels, growth_rate, drop_path_rate=0.0):
        super(DenseBlock3D, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = DenseLayer3D(
                in_channels + i * growth_rate,
                growth_rate,
                drop_path_rate=drop_path_rate * (i / num_layers)
            )
            self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer3D(nn.Module):
    """Transition layer to reduce spatial dimensions and compress features"""
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer3D, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
        
    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet3D_Attention(nn.Module):
    """3D DenseNet with Multi-Head Attention for Lung Cancer Detection"""
    
    def __init__(self, in_channels=1, num_classes=2, growth_rate=16, 
                 num_layers=[4, 4, 4], num_heads=4, drop_path_rate=0.2):
        super(DenseNet3D_Attention, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks with transitions
        num_features = 64
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        for i, num_layers_in_block in enumerate(num_layers):
            # Dense block
            block = DenseBlock3D(
                num_layers=num_layers_in_block,
                in_channels=num_features,
                growth_rate=growth_rate,
                drop_path_rate=drop_path_rate
            )
            self.dense_blocks.append(block)
            num_features += num_layers_in_block * growth_rate
            
            # Attention after each dense block
            attention = MultiHeadAttention3D(num_features, num_heads=num_heads)
            self.attentions.append(attention)
            
            # Transition layer (except for the last block)
            if i < len(num_layers) - 1:
                transition = TransitionLayer3D(num_features, num_features // 2)
                self.transitions.append(transition)
                num_features = num_features // 2
        
        # Final batch norm
        self.bn_final = nn.BatchNorm3d(num_features)
        self.relu_final = nn.ReLU(inplace=True)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
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
        # Initial conv
        x = self.conv1(x)
        
        # Dense blocks with attention and transitions
        for i, (dense_block, attention) in enumerate(zip(self.dense_blocks, self.attentions)):
            x = dense_block(x)
            x = x + attention(x)  # Add attention residual
            
            if i < len(self.transitions):
                x = self.transitions[i](x)
        
        # Final layers
        x = self.relu_final(self.bn_final(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
