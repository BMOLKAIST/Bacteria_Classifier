import torch
import torch.nn as nn


# Channel Squeeze
class SpatialExcitation(nn.Module):
    def __init__(self, in_c):
        super(SpatialExcitation, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(in_c, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        att = self.attention(x)
        return x * att


# Spatial Squeeze
class ChannelExcitation(nn.Module):
    def __init__(self, in_c):
        super(ChannelExcitation, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_c, in_c // 2, kernel_size=1),
            nn.Conv3d(in_c // 2, in_c, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        att = self.attention(x)
        return x * att
