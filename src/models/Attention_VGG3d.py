import torch.nn as nn
import math
from .layers.attention_vgg3d_layer import *

class VGG_Each(nn.Module):

    def __init__(self, features, num_classes=7, init_weights=True,inter_channel=64,channel_list=[64,128,256,512,512]):
        super(VGG_Each, self).__init__()
        self.features = features
        self.num_classes = num_classes
        self.inter_channel = inter_channel
        self.channel_list = channel_list
        self.GAP_layer = nn.AdaptiveAvgPool3d(output_size=(1,1,1))
        
        self.feature_1 = nn.Sequential(*list(self.features.children())[:7])
        self.feature_2 = nn.Sequential(*list(self.features.children())[7:14])
        self.feature_3 = nn.Sequential(*list(self.features.children())[14:27])
        self.feature_4 = nn.Sequential(*list(self.features.children())[27:40])
        self.feature_5 = nn.Sequential(*list(self.features.children())[40:])

        self.attention_1 = Attention_Layer_3D(feature_size=(1,256, 12, 12, 5),
                                              global_size=(1,512, 3, 3, 1),
                                              inter_channel=self.inter_channel)
        self.attention_2 = Attention_Layer_3D(feature_size=(1,512, 6, 6, 2),
                                              global_size=(1,512, 3, 3, 1),
                                              inter_channel=self.inter_channel)

        self.classifier_1 = nn.Sequential(
            nn.Linear(self.channel_list[2],100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100,self.num_classes),
        )

        self.classifier_2 = nn.Sequential(
            nn.Linear(self.channel_list[3],100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100,self.num_classes),
        )

        self.classifier_3 = nn.Sequential(
            nn.Linear(self.channel_list[4],100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100,self.num_classes),
        )
    
        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        batch_size = x.size()[0]

        x = self.feature_1(x)
        x = self.feature_2(x)
        out_3 = self.feature_3(x)
        out_4 = self.feature_4(out_3)
        out_5 = self.feature_5(out_4)

        attention_1,weighted_feature_1 = self.attention_1(out_3,out_5)
        attention_2,weighted_feature_2 = self.attention_2(out_4,out_5)
        gap_out = self.GAP_layer(out_5)

        out_1 = self.classifier_1(weighted_feature_1.view(batch_size,-1))
        out_2 = self.classifier_2(weighted_feature_2.view(batch_size,-1))
        out   = self.classifier_3(gap_out.view(batch_size,-1))

        return out_1,out_2,out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG_Concat(nn.Module):

    def __init__(self, features, num_classes=7, init_weights=True,inter_channel=64):
        super(VGG_Concat, self).__init__()
        self.features = features
        self.num_classes = num_classes
        self.inter_channel = inter_channel
        self.GAP_layer = nn.AdaptiveAvgPool3d(output_size=(1,1,1))
        
        self.feature_1 = nn.Sequential(*list(self.features.children())[:7])
        self.feature_2 = nn.Sequential(*list(self.features.children())[7:14])
        self.feature_3 = nn.Sequential(*list(self.features.children())[14:27])
        self.feature_4 = nn.Sequential(*list(self.features.children())[27:40])
        self.feature_5 = nn.Sequential(*list(self.features.children())[40:])

        self.attention_1 = Attention_Layer_3D(feature_size=(1, 512, 24, 24, 5),
                                              global_size=(1, 512, 6, 6, 1),
                                              inter_channel=self.inter_channel)
        self.attention_2 = Attention_Layer_3D(feature_size=(1,512, 12, 12, 2),
                                              global_size=(1,512, 6, 6, 1),
                                              inter_channel=self.inter_channel)

        self.final_ch = 512 + 512 + 512 # out_3, out_4, out_5
        self.classifier_1 = nn.Sequential(
            nn.Linear(self.final_ch, 100),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(100),
            nn.Linear(100,self.num_classes),
        )

        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        batch_size = x.size()[0]
        x = self.feature_1(x)
        x = self.feature_2(x)
        out_3 = self.feature_3(x)
        out_4 = self.feature_4(out_3)
        out_5 = self.feature_5(out_4)

        attention_1, weighted_feature_1 = self.attention_1(out_3, out_5)
        attention_2, weighted_feature_2 = self.attention_2(out_4, out_5)
        gap_out = self.GAP_layer(out_5)

        concat_x = torch.cat((weighted_feature_1, weighted_feature_2, gap_out), dim=1)
        concat_x = concat_x.view(batch_size, self.final_ch)
        x = self.classifier_1(concat_x)
        return x, None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [28, 28, 'M', 56, 56, 'M', 112, 112, 112, 112, 'M', 224, 224, 224, 224, 'M', 224, 224, 224, 224, 'M'],
    'G':[32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'H':[16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'I':[8, 'M', 16, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
}


def attvgg11_bn(pretrained=False, agg=False, **kwargs):
    if agg:
        print("Training VGG with concatenated feature map from 3 Classifier")
        model = VGG_Concat(make_layers(cfg['A'], batch_norm=True), **kwargs)
    else:
        print("Training VGG Each with 3 Classifier")
        model = VGG_Each(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model

def attvgg13_bn(pretrained=False, agg=False, **kwargs):
    if agg:
        print("Training VGG with concatenated feature map from 3 Classifier")
        model = VGG_Concat(make_layers(cfg['B'], batch_norm=True), **kwargs)
    else:
        print("Training VGG Each with 3 Classifier")
        model = VGG_Each(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def attvgg16_bn(pretrained=False, agg=False, **kwargs):
    if agg:
        print("Training VGG with concatenated feature map from 3 Classifier")
        model = VGG_Concat(make_layers(cfg['D'], batch_norm=True), **kwargs)
    else:
        print("Training VGG Each with 3 Classifier")
        model = VGG_Each(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def attvgg19_bn(pretrained=False,agg=False, **kwargs):
    if agg:
        print("Training VGG with concatenated feature map from 3 Classifier")
        model = VGG_Concat(make_layers(cfg['E'], batch_norm=True), **kwargs)
    else:
        print("Training VGG Each with 3 Classifier")
        model = VGG_Each(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model


def attvgg11_gkim_bn(pretrained=False, **kwargs):
    if agg:
        print("Training VGG with concatenated feature map from 3 Classifier")
        model = VGG_Concat(make_layers(cfg['G'], batch_norm=True), **kwargs)
    else:
        print("Training VGG Each with 3 Classifier")
        model = VGG_Each(make_layers(cfg['G'], batch_norm=True), **kwargs)
    return model

def attvgg11_gkim2_bn(pretrained=False, **kwargs):
    if agg:
        print("Training VGG with concatenated feature map from 3 Classifier")
        model = VGG_Concat(make_layers(cfg['H'], batch_norm=True), **kwargs)
    else:
        print("Training VGG Each with 3 Classifier")
        model = VGG_Each(make_layers(cfg['H'], batch_norm=True), **kwargs)
    return model
        
def attvgg11_gkim3_bn(pretrained=False, **kwargs):
    if agg:
        print("Training VGG with concatenated feature map from 3 Classifier")
        model = VGG_Concat(make_layers(cfg['I'], batch_norm=True), **kwargs)
    else:
        print("Training VGG Each with 3 Classifier")
        model = VGG_Each(make_layers(cfg['I'], batch_norm=True), **kwargs)
    return model


def attvgg11_gkim(pretrained=False, **kwargs):
    if agg:
        print("Training VGG with concatenated feature map from 3 Classifier")
        model = VGG_Concat(make_layers(cfg['G'], batch_norm=False), **kwargs)
    else:
        print("Training VGG Each with 3 Classifier")
        model = VGG_Each(make_layers(cfg['G'], batch_norm=False), **kwargs)
    return model

def attvgg11_gkim2(pretrained=False, **kwargs):
    if agg:
        print("Training VGG with concatenated feature map from 3 Classifier")
        model = VGG_Concat(make_layers(cfg['H'], batch_norm=False), **kwargs)
    else:
        print("Training VGG Each with 3 Classifier")
        model = VGG_Each(make_layers(cfg['H'], batch_norm=False), **kwargs)
    return model
        
def attvgg11_gkim3(pretrained=False, **kwargs):
    if agg:
        print("Training VGG with concatenated feature map from 3 Classifier")
        model = VGG_Concat(make_layers(cfg['I'], batch_norm=False), **kwargs)
    else:
        print("Training VGG Each with 3 Classifier")
        model = VGG_Each(make_layers(cfg['I'], batch_norm=False), **kwargs)
    return model
