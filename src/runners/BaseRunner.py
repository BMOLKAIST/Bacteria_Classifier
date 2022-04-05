import utils

class BaseRunner:
    def __init__(self, arg, torch_device):
        self.arg = arg
        self.torch_device = torch_device 
        
        self.model_type = arg.model

        self.batch_size = arg.batch_size
        
        self.model_dir = arg.model_dir
    
    def save(self):
        raise NotImplementedError("notimplemented save method")

    def load(self):
        raise NotImplementedError("notimplemented save method")

    def train(self):
        raise NotImplementedError("notimplemented save method")

    def valid(self):
        raise NotImplementedError("notimplemented valid method")

    def test(self):
        raise NotImplementedError("notimplemented test method")

    def inference(self):
        raise NotImplementedError("notimplemented interence method")
        
        
