import torch.nn as nn

def _bn_relu_conv(in_c, out_c, norm=nn.BatchNorm3d, act=lambda : nn.ReLU(True), **conv_kwargs):
    return nn.Sequential(
        norm(in_c),
        act(),
        nn.Conv3d(in_c, out_c, **conv_kwargs),
    )

class _ConvBlock(nn.Module):
    """
    Construct Basic Bottleneck Convolution Block module.
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        stride : Stride of middle conv layer
        dilation : Dilation rate of middle conv layer

    Forwarding Path:
        input image - (BN-ReLU-Conv) * 3 - output
    """
    def __init__(self, in_c, out_c, stride=1, dilation=1, norm=nn.BatchNorm3d, act=lambda : nn.ReLU(True)):
        super().__init__()

        mid_c = out_c // 4
        self.layers = nn.Sequential(
            _bn_relu_conv(in_c, mid_c,  norm=norm, act=act, kernel_size=1, bias=False),
            _bn_relu_conv(mid_c, mid_c, norm=norm, act=act, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            _bn_relu_conv(mid_c, out_c, norm=norm, act=act, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    """
    Construct Basic Bottleneck Convolution Block module.
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        stride : Stride of middle conv layer
        dilation : Dilation rate of middle conv layer

    Forwarding Path:
        input image - (BN-ReLU-Conv) * 3 - output
    """
    def __init__(self, in_c, out_c, shortcut=lambda x: x,
                 stride=1, dilation=1,
                 norm=nn.BatchNorm3d, act=lambda : nn.ReLU(True)):
        super(ResBlock, self).__init__()

        mid_c = out_c // 4
        self.layers = nn.Sequential(
            _bn_relu_conv(in_c, mid_c,  norm=norm, act=act, kernel_size=1, bias=False),
            _bn_relu_conv(mid_c, mid_c, norm=norm, act=act, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            _bn_relu_conv(mid_c, out_c, norm=norm, act=act, kernel_size=1, bias=False),
        )

        # shortcut = _bn_relu_conv / lambda x: x
        self.shortcut = shortcut

    def forward(self, x):
        return self.layers(x) + self.shortcut(x)



class TransferBlock(nn.Module):
    """
    Construct Transfer Block module.
    
    Args:
        ch : Number of channels in the input and output image
        num_blk : Number of Residual Blocks

    Forwarding Path:
        input image - (ConvBlock) * num_blk - output
    """
    def __init__(self, ch, num_blk, norm=nn.BatchNorm3d, act=lambda : nn.ReLU(True)):
        super().__init__()
        
        self.layers = nn.Sequential(
            *[ResBlock(ch, ch, norm=norm, act=act) for _ in range(0, num_blk)]
        )

    def forward(self, x):
        return self.layers(x)


class DownRefinementBlock(nn.Module):
    """
    Construct Down-RefinementBlock module. (DRBlock from the original paper)
    Consisted of one Residual Block and Conv Blocks.
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        stride : Stride of shortcut conv layer

    Forwarding Path:
                    ⎡      (BN-ReLU-Conv)     ⎤
        input image - (ConvBlock) * num_blk -(sum)- feature - (MaxPool) - output
    """
    def __init__(self, in_c, out_c, num_blk, stride=1, norm=nn.BatchNorm3d, act=lambda : nn.ReLU(True)):
        super().__init__()

        shortcut = _bn_relu_conv(in_c, out_c, norm=norm, act=act, kernel_size=1, stride=stride, bias=False)
        self.layer = nn.Sequential(
            ResBlock(in_c, out_c, shortcut=shortcut, norm=norm, act=act),
            *[ResBlock(out_c, out_c, norm=norm, act=act) for _ in range(1, num_blk)],
            nn.MaxPool3d(2, stride=2)
        )
        
    def forward(self, x):
        return self.layer(x)
    


class UpRefinementBlock(nn.Module):
    """
    Construct Up-RefinementBlock module. (URBlock from the original paper)
    Consisted of Residual Block and Conv Blocks.
    Not like DRBlock, this module reduces the number of channels of concatenated feature maps in the shortcut path.
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        stride : Stride of shortcut conv layer

    Forwarding Path:
                    ⎡   (Channel Reduction)    ⎤
        input image - (ConvBlock) * num_blk -(sum)- feature - (UpSample) - output
    """
    def __init__(self, in_c, out_c, num_blk, stride=1, dilation=1, norm=nn.BatchNorm3d, act=lambda : nn.ReLU(True)):
        super().__init__()
        self.k = in_c // out_c

        self.layer = nn.Sequential(
            ResBlock(in_c, out_c, shortcut=self.channel_reduction, dilation=dilation, norm=norm),
            *[ResBlock(out_c, out_c, norm=norm, act=act, dilation=dilation) for _ in range(1, num_blk)],
            nn.Upsample(scale_factor=2)            
        )

    def channel_reduction(self, x):
        n, c, *dim = x.shape
        return x.view(n, c // self.k, self.k, *dim).sum(2)
        
    def forward(self, x):
        return self.layer(x)