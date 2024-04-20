#from src.models.resunet.layers import ResUnetEncoder, ResUnetDecoder, ResUnetClassifier, BNIdentity
from src.models.resunet.layers import SwinEncoder, SwinDecoder, SwinClassifier, BNIdentity
from torch import nn
import torch
from abc import abstractmethod
from einops import rearrange
#from ..utils import ModelModule, ModelModuleMultiTask

class GenericModel(nn.Module):
    def __init__(
        self, 
        in_dims, 
        base_dim ,
        window_size ,
        shift_size ,
        patch_size ,
        n_heads ,
        n_blocks,
        n_classes, 
        *args, **kargs) -> None:
        
        super().__init__(*args, **kargs)
        
        self.n_classes = n_classes
        self.in_dims = in_dims
        self.base_dim = base_dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.in_dims = in_dims
        self.in_dims = in_dims
        self.in_dims = in_dims

    def get_opt(self, x):
        return rearrange(x['opt'], 'b i c h w -> b (i c) h w')
    
    def get_sar(self, x):
        return rearrange(x['sar'], 'b i c h w -> b (i c) h w')
    


class GenericSwin(GenericModel):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.encoder = SwinEncoder(self.in_dims, self.depths)
        self.bn = BNIdentity(self.depths)
        self.decoder = SwinDecoder(self.bn.out_depths)
        self.classifier = SwinClassifier(self.depths, self.n_classes)

    @abstractmethod
    def prepare_input(self, x):
        pass
    
    def forward(self, x):
        #x = self.prepare_input(x)
        x = self.encoder(x)
        x = self.bn(x)
        x = self.decoder(x)
        x = self.classifier(x)
        return x

class SwinOpt(GenericSwin):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.exp.n_opt_imgs * cfg.general.n_opt_bands + 1 
    
    def prepare(self, x):
        x_img = self.get_opt(x)
        x = torch.cat((x_img, x['previous']), dim=1)
        return x
    
class SwinSAR(GenericSwin):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.exp.n_sar_imgs * cfg.general.n_sar_bands + 1 
    
    def prepare(self, x):
        x_img = self.get_sar(x)
        x = torch.cat((x_img, x['previous']), dim=1)
        return x

class SwinOptNoPrevMap(GenericSwin):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.exp.n_opt_imgs * cfg.general.n_opt_bands 
    
    def prepare(self, x):
        x_img = self.get_opt(x)
        return x_img
    
class SwinSARNoPrevMap(GenericSwin):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.exp.n_sar_imgs * cfg.general.n_sar_bands 
    
    def prepare(self, x):
        x_img = self.get_sar(x)
        return x_img