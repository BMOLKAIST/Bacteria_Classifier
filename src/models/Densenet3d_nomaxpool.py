import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math


__all__ = [
    'DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet264'
]


def d121_3d(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        last_features=1024,
        **kwargs)
    return model

def dwdense_3d(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(12, 24, 32, 24),
        last_features=1528,
        **kwargs)
    return model

def d169_3d(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        last_features=1664,
        **kwargs)
    return model


def d201_3d(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        last_features=1920,
        **kwargs)
    return model


def d264_3d(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 64, 48),
        last_features=2688,
        **kwargs)
    return model


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('denseblock{}'.format(i))
        ft_module_names.append('transition{}'.format(i))
    ft_module_names.append('norm5')
    ft_module_names.append('classifier')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, norm, act):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', norm(num_input_features))
        self.add_module('relu1', act())
        self.add_module('conv1',
                        nn.Conv3d(
                            num_input_features,
                            bn_size * growth_rate,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('norm2', norm(bn_size * growth_rate))
        self.add_module('relu2', act())
        self.add_module('conv2',
                        nn.Conv3d(
                            bn_size * growth_rate,
                            growth_rate,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False))
        self.drop_rate = drop_rate
        
    def forward(self, x):
        #print(x.size())
        
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=False)

        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate, norm, act):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate, norm, act)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, norm, act):
        super(_Transition, self).__init__()
        self.add_module('norm', norm(num_input_features))
        self.add_module('relu', act())
        self.add_module('conv',
                        nn.Conv3d(
                            num_input_features,
                            num_output_features,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 sample_size,
                 sample_duration,
                 last_features,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000,
                 norm="bn",
                 act="relu",
                 dim = "3d"):
        super(DenseNet, self).__init__()

        norm = {"bn":nn.BatchNorm3d, "in":nn.InstanceNorm3d}[norm]
        act = {
            "relu": lambda : nn.ReLU(inplace=True),
            "lrelu": lambda : nn.LeakyReLU(inplace=True),
            "prelu": lambda : nn.PReLU(),
        }[act]

        self.last_features = last_features
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes
        self.dim = dim
        
        if self.dim == "3d":
            chan_data = 1
        elif self.dim == "mv":
            chan_data = 98
        elif self.dim == "2d":
            chan_data = 2
        # First convolution
        self.features = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     chan_data,
                     num_init_features,
                     kernel_size=3,#original densenet: 7 kernel_size=7
                     stride=(2, 2, 2), 
                     padding=(1, 1, 1),#original densenet: 3 padding=(3, 3, 3),
                     bias=False)),
                ('norm0', norm(num_init_features)),
                ('relu0', act()),
                #('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)), #original densenet: yes
            ]))


        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                norm=norm, act=act)
            self.features.add_module('denseblock%d' % (i + 1), block)

            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    norm=norm, act=act)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
                
        # Final batch norm
        self.features.add_module('norm5', norm(num_features))
        self.features.add_module('act5', act())

        self.init_weight()

        # change avgpool to adaptive pooling by Gunho Choi
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=[1,1,1])

        # Linear layer(change by GunhoChoi)
        self.classifier = nn.Linear(self.last_features, num_classes)

        self.gram = nn.Linear(self.last_features, 1) #later from the learned features to subtask
        self.shape = nn.Linear(self.last_features, 1)
        self.mot = nn.Linear(self.last_features, 1)
        self.genus = nn.Linear(self.last_features, 14)

    def forward(self, x):
        features = self.features(x)

        pool = self.avgpool(features).view(x.size()[0],-1)

        out = self.classifier(pool)
        
        gram = self.gram(pool).view(-1)
        shape = self.shape(pool).view(-1)
        mot = self.mot(pool).view(-1)
        genus = self.genus(pool)
        return out, gram, shape, mot, genus
    
    def get_feature(self, x):
        features = self.features(x)
        return features
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_normal_(m.weight)
                m.bias = nn.init.normal_(m.bias)


if __name__ == "__main__":
    net = d169_3d(num_classes=19, sample_size=64, sample_duration=96, norm="bn", act="lrelu")
    from torchsummary import summary
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    torch_device = torch.device("cuda")
    
    summary(net, (1, 96, 96, 21), device="cpu")

