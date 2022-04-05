import torch.nn as nn
import torch.nn.functional as F

class Autofocus(nn.Module):
    def __init__(self, in_c, h_c, out_c,
                 padds=[0, 4, 8, 12],
                 dilations=[2, 6, 10, 14],
                 branches=4,
                 act=None):
        super(Autofocus, self).__init__()
        self.padds = padds
        self.dilations = dilations
        self.branches = branches

        self.conv1 = nn.Conv3d(in_c, h_c,
                               kernel_size=3, padding=2, dilation=self.dilations[0])

        self.c1 =nn.Conv3d(in_c, in_c//2, kernel_size=3, padding=1)
        self.r = act()
        self.c2 = nn.Conv3d(in_c//2, self.branches, kernel_size=1)
        self.att1 = nn.Sequential(
            nn.Conv3d(in_c, in_c // 2, kernel_size=3, padding=1),
            act(),
            nn.Conv3d(in_c // 2, self.branches, kernel_size=1),
        )

        self.bn_list1 = nn.ModuleList()
        for i in range(self.branches):
            self.bn_list1.append(nn.BatchNorm3d(h_c))


        self.conv2 = nn.Conv3d(h_c, out_c, kernel_size=3, padding=2,dilation=self.dilations[0])
        self.att2 = nn.Sequential(
            nn.Conv3d(h_c, h_c // 2, kernel_size=3, padding=1),
            act(),
            nn.Conv3d(h_c // 2, self.branches, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.bn_list2 = nn.ModuleList()
        for i in range(self.branches):
            self.bn_list2.append(nn.BatchNorm3d(out_c))
        
        self.act_f = act()

        if in_c == out_c:            
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv3d(in_c, out_c, kernel_size=1), nn.BatchNorm3d(out_c))
            
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x
        # compute attention weights in the first autofocus convolutional layer
        feature = x.detach()
        c1 = self.c1(feature)
        r = self.r(c1)
        c2 = self.c2(r)
        att = self.att1(feature)
        att = F.softmax(att, dim=1)

        # linear combination of different rates
        x1 = self.conv1(x)
        shape = x1.size()
        x1 = self.bn_list1[0](x1) * att[:, 0:1].expand(shape)

        for i in range(1, self.branches):
            x2 = F.conv3d(x, self.conv1.weight, padding=self.dilations[i], dilation=self.dilations[i])
            x2 = self.bn_list1[i](x2)
            x1 += x2 * att[:, i:i+1, :, : :].expand(shape)
        
        x = self.act_f(x1)
        # compute attention weights for the second autofocus layer
        feature2 = x.detach()
        att2 = self.att2(feature2)
        
        # linear combination of different rates
        x21 = self.conv2(x)
        shape = x21.size()
        x21 = self.bn_list2[0](x21)* att2[:, 0:1].expand(shape)
        
        for i in range(1, self.branches):
            x22 = F.conv3d(x, self.conv2.weight, padding =self.dilations[i], dilation=self.dilations[i])
            x22 = self.bn_list2[i](x22)
            x21 += x22 * att2[:, i:i+1].expand(shape)
                
        if self.downsample is not None:
            residual = self.downsample(residual)
     
        x = x21 + residual
        x = self.act_f(x)
        return x
