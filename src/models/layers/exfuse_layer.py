import torch
import torch.nn as nn
import torch.nn.functional as F


# Conv-Norm-Activation
class CNA(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1,
                 norm=nn.BatchNorm3d, act=nn.ReLU):
        super(CNA, self).__init__()
        self.layer = nn.Sequential(nn.Conv3d(in_c, out_c, kernel_size, stride, padding),
                                   norm(out_c), act(True))

    def forward(self, x):
        return self.layer(x)


# UpConv-Norm-Activation
class UpCNA(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=4, stride=2, padding=1,
                 norm=nn.BatchNorm3d, act=nn.ReLU):
        super(UpCNA, self).__init__()
        self.layer = nn.Sequential(nn.ConvTranspose3d(in_c, out_c, kernel_size, stride, padding),
                                   norm(out_c), act(True))

    def forward(self, x):
        return self.layer(x)


# Semantic Embedding Branch, Fig 4
class SEB_dw(nn.Module):
    def __init__(self, low_feature, high_feature,
                 norm=nn.BatchNorm3d, up_scale=2):
        super(SEB_dw, self).__init__()
        self.conv = CNA(high_feature, low_feature, norm=norm)
        self.up = nn.Upsample(scale_factor=up_scale)

    def forward(self, low_feature, high_feature):
        high_feature = self.conv(high_feature)
        high_feature = self.up(high_feature)
        return low_feature * high_feature # element wise mul


# Orignal Paper Impl
class SEB(nn.Module):
    def __init__(self, low_feature, high_features,
                 norm=nn.BatchNorm3d, up_scale=2):
        super(SEB, self).__init__()
        self.sebs = []
        for c in range(len(high_features) - 1, 0, -1):
            self.sebs.append(nn.Sequential(CNA(high_features[c], high_features[c - 1], norm=norm),
                                           nn.Upsample(scale_factor=up_scale)))

    def forward(self, low_feature, *high_features):
        high_features = list(reversed(high_features))
        
        low_feature = self.seb[0](high_features[0]) * high_features[1]
        for c in range(1, len(high_features)):
            high_feature = self.sebs[c](high_features[c])
            low_feature *= high_feature
            
        return low_feature  # element wise mul


# Global Convolution Network
# https://github.com/ycszen/pytorch-segmentation
# https://arxiv.org/pdf/1703.02719.pdf
class GCN(nn.Module):
    def __init__(self, in_c, out_c, ks=7, norm=nn.BatchNorm3d):
        super(GCN, self).__init__()
        self.conv_l1 = CNA(in_c, out_c, kernel_size=(ks, 1, 1),
                           padding=(ks // 2, 0, 0), norm=norm)
        self.conv_l2 = CNA(out_c, out_c, kernel_size=(1, ks, 1),
                           padding=(0, ks // 2, 0), norm=norm)

        self.conv_r1 = CNA(in_c, out_c, kernel_size=(1, ks, 1),
                           padding=(0, ks // 2, 0), norm=norm)
        self.conv_r2 = CNA(out_c, out_c, kernel_size=(ks, 1, 1),
                           padding=(ks // 2, 0, 0), norm=norm)

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        return x_l + x_r


# Explicit Channel Resolution Embedding
class ECRE(nn.Module):
    def __init__(self, in_c, up_scale=2, norm=nn.BatchNorm3d):
        super(ECRE, self).__init__()
        self.ecre = nn.Sequential(CNA(in_c, in_c * up_scale * up_scale, norm=norm),
                                  nn.PixelShuffle(up_scale))

    def forward(self, input_):
        return self.ecre(input_)
