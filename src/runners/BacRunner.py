import os
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from .BaseRunner import BaseRunner
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import time
from utils import get_confusion
import scipy.io as io
import time

class BacRunner(BaseRunner):
    def __init__(self, arg, net, torch_device,  load_fname = None):
        super().__init__(arg, torch_device)

        self.fname = load_fname
        if(self.fname == None):
            self.fname = "save_temp"#"epoch[%05d]"%(self.epoch)
        self.net = net

        self.load(load_fname) 
            
  

    def load(self, filename=None):
        """ Model load. same with save"""
        if filename is None:
            # load last epoch model
            filenames = sorted(glob(self.model_dir + "/*.pth.tar"))
            if len(filenames) == 0:
                print("Not Load")
                return
            else:
                filename = os.path.basename(filenames[-1])

        file_path = self.model_dir + "/" + filename
        if os.path.exists(file_path) is True:
            print("Load %s to %s File"%(self.model_dir, filename))
            ckpoint = torch.load(file_path)
            if ckpoint["model_type"] != self.model_type:
                raise ValueError("Ckpoint Model Type is %s"%(ckpoint["model_type"]))
            
            self.net.load_state_dict(ckpoint['network'])
            print("Load Model Type : %s, filename: %s"%(ckpoint["model_type"], self.fname))
        else:
            print("Load Failed, not exists file")

    def _get_acc_test(self, loader, confusion=False): #outputs the scores(confidence) too, will only work for batch size = 1 (GK, 200320)
        correct = 0
        preds, labels = [], []
        matdict = {}
        targets = np.zeros(0, dtype=np.int8)
        scores = np.zeros(0, dtype=np.float32)
        paths = np.zeros(0, dtype=np.object)
        for input_, target_, path_ in loader:
            input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
            output_, *_ = self.net(input_)

            
            targets = np.append(targets, target_.cpu().numpy())
            scores = np.append(scores, output_.cpu().numpy())
            paths = np.append(paths, path_)
            
            _, idx = output_.max(dim=1)

            correct += torch.sum(target_ == idx).float().cpu().item()

            if confusion:
                preds += idx.view(-1).tolist()
                labels += target_.view(-1).tolist()
                
        matdict['targets'] = targets
        matdict['scores'] = scores
        matdict['paths'] = paths
        if confusion:
            confusion = get_confusion(preds, labels)
        return correct / len(loader.dataset), confusion, matdict

    def test(self, test_loader):
        print("\n Start Test")
        # self.load()
        self.net.eval()
        with torch.no_grad():
        
            test_acc, test_confusion, matdict_test  = self._get_acc_test(test_loader, confusion=True)
            
            os.mkdir(self.model_dir + "/" + self.fname[0:46])
            np.save(self.model_dir + "/" + self.fname[0:46]+"/test_confusion.npy", test_confusion)
            io.savemat(self.model_dir + "/" + self.fname[0:46]+"/result_test.mat", matdict_test)
            print(test_confusion)
        return test_acc
        
