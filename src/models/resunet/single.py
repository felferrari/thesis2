from src.models.resunet.layers import ResUnetEncoder, ResUnetDecoder, ResUnetClassifier, BNIdentity
from torch import nn
import torch
from abc import abstractmethod
from einops import rearrange
#from ..utils import ModelModule, ModelModuleMultiTask

# class GenericModel(nn.Module):
#     def __init__(self, depths, n_classes, in_dims, *args, **kargs) -> None:
#         super().__init__(*args, **kargs)
#         self.n_classes = n_classes
#         self.depths = depths
#         self.in_dims = in_dims
#         self.construct_model()

#     def get_opt(self, x):
#         return rearrange(x['opt'], 'b i c h w -> b (i c) h w')
    
#     def get_sar(self, x):
#         return rearrange(x['sar'], 'b i c h w -> b (i c) h w')
    
#     def construct_model(self):
#         self.encoder = ResUnetEncoder(self.in_dims, self.depths)
#         self.fusion = IdentityFusion(self.depths)
#         self.decoder = ResUnetDecoder(self.fusion.out_depths)
#         self.classifier = ResUnetClassifier(self.depths, self.n_classes)

class GenericResunet(nn.Module):
    def __init__(self, depths, n_classes, in_dims, *args, **kargs) -> None:
        super().__init__(*args, **kargs)
        self.n_classes = n_classes
        self.depths = depths
        self.in_dims = in_dims
        self.construct_model()

    def get_opt(self, x):
        return rearrange(x['opt'], 'b i c h w -> b (i c) h w')
    
    def get_sar(self, x):
        return rearrange(x['sar'], 'b i c h w -> b (i c) h w')
    
    def construct_model(self):
        self.encoder = ResUnetEncoder(self.in_dims, self.depths)
        self.bn = BNIdentity(self.depths)
        self.decoder = ResUnetDecoder(self.depths)
        self.classifier = ResUnetClassifier(self.depths, self.n_classes)
        
    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.encoder(x)
        x = self.bn(x)
        x = self.decoder(x)
        x = self.classifier(x)
        return x

class ResUnetOpt(GenericResunet):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.exp.n_opt_imgs * cfg.general.n_opt_bands + 1 
    
    def prepare(self, x):
        x_img = self.get_opt(x)
        #x = torch.cat((x_img, x['previous']), dim=1)
        return (x_img, x['previous'])
    
class ResUnetSAR(GenericResunet):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.exp.n_sar_imgs * cfg.general.n_sar_bands + 1 
    
    def prepare(self, x):
        x_img = self.get_sar(x)
        #x = torch.cat((x_img, x['previous']), dim=1)
        return (x_img, x['previous'])

class ResUnetOptNoPrevMap(GenericResunet):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.exp.n_opt_imgs * cfg.general.n_opt_bands 
    
    def prepare(self, x):
        x_img = self.get_opt(x)
        return (x_img,)
    
class ResUnetSARNoPrevMap(GenericResunet):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.exp.n_sar_imgs * cfg.general.n_sar_bands 
    
    def prepare(self, x):
        x_img = self.get_sar(x)
        return (x_img,)