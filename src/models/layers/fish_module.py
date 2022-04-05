import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.fish_residual import _bn_relu_conv, ResBlock, DownRefinementBlock, TransferBlock, UpRefinementBlock
from models.layers.exfuse_layer import GCN

class FishTail(nn.Module):
    """
    Construct FishTail module.
    Each instances corresponds to each stages.
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks

    Forwarding Path:
        input image - (DRBlock) - output
    """
    def __init__(self, in_c, out_c, num_blk, norm=nn.BatchNorm3d, act=lambda : nn.ReLU(True)):
        super().__init__()
        self.layer = DownRefinementBlock(in_c, out_c, num_blk, norm=norm, act=act)

    def forward(self, x):
        return self.layer(x)

class Bridge(nn.Module):
    """
    Construct Bridge module.
    This module bridges the last FishTail stage and first FishBody stage.
    
    Args:
        ch : Number of channels in the input and output image
        num_blk : Number of Residual Blocks

    Forwarding Path:
                        r             (SEBlock)         ??
        input image - (stem) - (_ConvBlock)*num_blk - (mul & sum) - output
    """         
    def __init__(self, ch, num_blk, norm=nn.BatchNorm3d, act=lambda : nn.ReLU(True)):
        super().__init__()

        self.stem = nn.Sequential(
            norm(ch),
            act(),
            nn.Conv3d(ch, ch//2, kernel_size=1, bias=False),
            norm(ch//2),
            act(),
            nn.Conv3d(ch//2, ch*2, kernel_size=1, bias=True)
        )


        shortcut = _bn_relu_conv(ch*2, ch, norm=norm, act=act, kernel_size=1, bias=False)
        self.layers = nn.Sequential(
            ResBlock(ch*2, ch, shortcut=shortcut, norm=norm, act=act),
            *[ResBlock(ch, ch, norm=norm, act=act) for _ in range(1, num_blk)],
        )
        # https://github.com/kevin-ssy/FishNet/blob/master/models/fishnet.py#L45
        self.se_block = nn.Sequential(
            norm(ch*2),
            act(),
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(ch*2, ch//16, kernel_size=1),
            act(),
            nn.Conv3d(ch//16, ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.stem(x)
        att = self.se_block(x)
        out = self.layers(x)
        return (out * att) + att


  
class FishBody(nn.Module):
    r"""Construct FishBody module.
    Each instances corresponds to each stages.
    
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        trans_in_c : Number of channels in the transferred image
        num_trans : Number of Transfer Blocks
        dilation : Dilation rate of Conv in UpRefinementBlock
        
    Forwarding Path:
        input image - (URBlock)  ??        trans image - (transfer) --(concat)-- output
    """
    def __init__(self, in_c, out_c, num_blk,
                 trans_in_c, num_trans,
                 dilation=1, norm=nn.BatchNorm3d, act=lambda : nn.ReLU(True)):
        super().__init__()
        self.layer = UpRefinementBlock(in_c, out_c, num_blk, dilation=dilation, norm=norm, act=act)
        self.transfer = TransferBlock(trans_in_c, num_trans, norm=norm, act=act)

    def forward(self, x, trans_x):
        x = self.layer(x)
        trans_x = self.transfer(trans_x)
        pad_tuple = [0, 0, 0, 0, 0, 0]
        if x.shape[2] < trans_x.shape[2]:
            pad_tuple[2] = 1
            pad_tuple[5] = 1
        if x.shape[-1] < trans_x.shape[-1]:
            pad_tuple[0] = 1
        print(x.shape)
        print(trans_x.shape)
        x = F.pad(x, pad_tuple)
        return torch.cat([x, trans_x], dim=1)

class FishHead(nn.Module):
    r"""Construct FishHead module.
    Each instances corresponds to each stages.

    Different with Offical Code : we used shortcut layer in this Module. (shortcut layer is used according to the original paper)
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        trans_in_c : Number of channels in the transferred image
        num_trans : Number of Transfer Blocks
        
    Forwarding Path:
        input image - (DRBlock)  ??        trans image - (transfer) --(concat)-- output
    """
    def __init__(self, in_c, out_c, num_blk,
                 trans_in_c, num_trans,
                 norm=nn.BatchNorm3d, act=lambda : nn.ReLU(True)):
        super().__init__()

        self.layer = nn.Sequential(
            ResBlock(in_c, out_c, norm=norm, act=act),
            *[ResBlock(out_c, out_c, norm=norm, act=act) for _ in range(1, num_blk)],
            nn.MaxPool3d(2, stride=2)
        )
        self.transfer = TransferBlock(trans_in_c, num_trans, norm=norm, act=act)

    def forward(self, x, trans_x):
        x = self.layer(x)
        trans_x = self.transfer(trans_x)
        return torch.cat([x, trans_x], dim=1)


class FishGCNBody(FishBody):
     def __init__(self, in_c, out_c, num_blk, 
                 trans_in_c, num_trans,
                 dilation=1):
        super().__init__(in_c, out_c, num_blk, trans_in_c, num_trans, dilation)
        self.transfer = nn.Sequential(
            TransferBlock(trans_in_c, num_trans),
            GCN(trans_in_c, trans_in_c)
        )