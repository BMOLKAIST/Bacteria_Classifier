from multiprocessing import Process
import os
import json
import argparse
import utils

import torch
import torch.nn as nn
#torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark = False

from datas.BacLoader import bacLoader

from datas.preprocess3d import TRAIN_AUGS_3D, TEST_AUGS_3D

from models.VGG3d import vgg16_bn, vgg19_bn
from models.Densenet3d_nomaxpool import d169_3d, d121_3d, d201_3d, dwdense_3d, d264_3d
from models.Attention_VGG3d import attvgg11_bn, attvgg13_bn, attvgg16_bn, attvgg19_bn

from runners.BacRunner import BacRunner

"""parsing and configuration"""
def arg_parse():
    # projects description
    desc = "Bacteria Classifier"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpus', type=str, default="0",
                        help="Specify GPU device index (multiple devices listed with comma)")
    parser.add_argument('--cpus', type=int, default="16",
                        help="Specify CPU Number workers")
    parser.add_argument('--data', type=str, default="",
                        help="Specify the directory containing bacteria")

    parser.add_argument('--aug', type=float, default=1, help='The number of Augmentation Rate')

    parser.add_argument('--norm', type=str, default='bn',
                        choices=["bn", "in"])
    parser.add_argument('--act', type=str, default='lrelu',
                        choices=["relu", "lrelu", "prelu"])
    parser.add_argument('--task', type=str, default='bac',
                        choices=["bac",  "gram", "aero"])

    parser.add_argument('--model', type=str, default='dense169',
                        choices=["attvgg11", "attvgg13", "attvgg16", "attvgg19",
                                 "dense169", "dense121", "dense201","dwdense",
                                 "dense_aniso", "d264"],
                        help='The architecture of the classifier network')

    parser.add_argument('--agg', action="store_true", help='Attention Aggregator')

    parser.add_argument('--model_dir', type=str, default='',
                        help='Directory name to save the model')
    parser.add_argument('--data_dir', type=str, default='',
                        help='Directory name to collect the data')
                        
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--load_fname',    type=str, default=None, help  ="")
    parser.add_argument('--drop_rate', type=float, default = 0)
    return parser.parse_args()


if __name__ == "__main__":
    arg = arg_parse()

    if os.path.exists(arg.model_dir) is False:
        os.mkdir(arg.model_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cuda")
    data_path = arg.data_dir
    print("Data Path : ",data_path)


    test_loader = bacLoader(data_path, arg.batch_size, task=arg.task,
                            transform=TEST_AUGS_3D, aug_rate=0,
                            num_workers=arg.cpus, shuffle=False, drop_last=False)


    classes = {"bac":19, "gram":2, "aero":2}[arg.task]

    if arg.model == "attvgg19":
        net = attvgg19_bn(pretrained=False, num_classes=classes, agg=arg.agg)
    elif arg.model == "attvgg11":
        net = attvgg11_bn(pretrained=False, num_classes=classes, agg=arg.agg)
    elif arg.model == "attvgg13":
        net = attvgg13_bn(pretrained=False, num_classes=classes, agg=arg.agg)
    elif arg.model == "attvgg16":
        net = attvgg16_bn(pretrained=False, num_classes=classes, agg=arg.agg)
    elif arg.model == "vgg16":
        net = vgg16_bn(pretrained=False, num_classes=classes)
    elif arg.model == "vgg19":
        net = vgg19_bn(pretrained=False, num_classes=classes)
    elif arg.model == "dense169":
        net = d169_3d(num_classes=classes, sample_size=64, sample_duration=96, norm=arg.norm, act=arg.act, drop_rate = arg.drop_rate, dim = "3d")
    elif arg.model == "dense121":
        net = d121_3d(num_classes=classes, sample_size=64, sample_duration=96, norm=arg.norm, act=arg.act, drop_rate = arg.drop_rate, dim = "3d")
    elif arg.model == "dense201":
        net = d201_3d(num_classes=classes, sample_size=64, sample_duration=96, norm=arg.norm, act=arg.act, drop_rate = arg.drop_rate, dim = "3d")
    
    net = nn.DataParallel(net).to(torch_device)
    loss = nn.CrossEntropyLoss()

    if arg.load_fname != None and (len(arg.load_fname) > 0):
        idx_comma = [index for index, value in enumerate(arg.load_fname) if value == ',']
        idx_front = [-1];
        idx_front = idx_front+idx_comma
        idx_comma.append(len(arg.load_fname))
        print(idx_comma)
        print(idx_front)
        for order_model in range(len(idx_comma)):
            fname_temp = arg.load_fname[idx_front[order_model]+1:idx_comma[order_model]]  
            print(fname_temp)
            model = BacRunner(arg, net, torch_device, load_fname = fname_temp)
    
            model.test(test_loader = test_loader)
    else:
        model = BacRunner(arg, net, torch_device, load_fname = fname_temp)

        model.test(test_loader = test_loader)