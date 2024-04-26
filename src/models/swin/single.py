#from src.models.resunet.layers import ResUnetEncoder, ResUnetDecoder, ResUnetClassifier, BNIdentity
from src.models.swin.layers import SwinEncoder, SwinDecoder, SwinClassifier, BNIdentity
from torch import nn
import torch
from abc import abstractmethod
from einops import rearrange
#from ..utils import ModelModule, ModelModuleMultiTask

# class GenericModel(nn.Module):
#     def __init__(
#         self, 
#         in_dims, 
#         img_size,
#         base_dim,
#         window_size,
#         shift_size,
#         patch_size,
#         n_heads,
#         n_blocks,
#         n_classes, 
#         *args, **kargs) -> None:
        
#         super().__init__(*args, **kargs)
        
#         self.n_classes = n_classes
#         self.in_dims = in_dims
#         self.base_dim = base_dim
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.patch_size = patch_size
#         self.n_heads = n_heads
#         self.n_blocks = n_blocks
#         self.img_size = img_size

#     def get_opt(self, x):
#         return rearrange(x['opt'], 'b i c h w -> b (i c) h w')
    
#     def get_sar(self, x):
#         return rearrange(x['sar'], 'b i c h w -> b (i c) h w')
    
#     def construct_model(self):
#         self.encoder = SwinEncoder(
#             input_depth = self.in_dims, 
#             base_dim = self.base_dim, 
#             window_size = self.window_size,
#             shift_size = self.shift_size,
#             img_size = self.img_size,
#             patch_size = self.patch_size,
#             n_heads = self.n_heads,
#             n_blocks = self.n_blocks
#             )
#         self.bn = BNIdentity()
#         self.decoder = SwinDecoder(
#             base_dim=self.base_dim,
#             n_heads=self.n_heads,
#             n_blocks = self.n_blocks,
#             window_size = self.window_size,
#             shift_size = self.shift_size
#             )
        
#         self.classifier = SwinClassifier(
#             self.base_dim, 
#             n_heads=self.n_heads,
#             n_blocks = self.n_blocks,
#             window_size = self.window_size,
#             shift_size = self.shift_size,
#             n_classes = self.n_classes
#             )



class GenericSwin(nn.Module):
    def __init__(
        self, 
        in_dims, 
        img_size,
        base_dim,
        window_size,
        shift_size,
        patch_size,
        n_heads,
        n_blocks,
        n_classes, 
        *args, **kargs) -> None:
        
        super().__init__(*args, **kargs)
        
        self.n_classes = n_classes
        self.in_dims = in_dims
        self.base_dim = base_dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.patch_size = patch_size
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.img_size = img_size

    def get_opt(self, x):
        return rearrange(x['opt'], 'b i c h w -> b (i c) h w')
    
    def get_sar(self, x):
        return rearrange(x['sar'], 'b i c h w -> b (i c) h w')
    
    def construct_model(self):
        self.encoder = SwinEncoder(
            input_depth = self.in_dims, 
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads = self.n_heads,
            n_blocks = self.n_blocks
            )
        self.bn = BNIdentity()
        self.decoder = SwinDecoder(
            base_dim=self.base_dim,
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size
            )
        
        self.classifier = SwinClassifier(
            self.base_dim, 
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size,
            n_classes = self.n_classes
            )

    def forward(self, x):
        #x = self.prepare_input(x)
        x = torch.cat(x, dim=1)
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
        #x = torch.cat((x_img, x['previous']), dim=1)
        return (x_img, x['previous'])
    
class SwinSAR(GenericSwin):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.exp.n_sar_imgs * cfg.general.n_sar_bands + 1 
    
    def prepare(self, x):
        x_img = self.get_sar(x)
        #x = torch.cat((x_img, x['previous']), dim=1)
        return (x_img, x['previous'])

class SwinOptNoPrevMap(GenericSwin):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.exp.n_opt_imgs * cfg.general.n_opt_bands 
    
    def prepare(self, x):
        x_img = self.get_opt(x)
        return (x_img,)
    
class SwinSARNoPrevMap(GenericSwin):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.exp.n_sar_imgs * cfg.general.n_sar_bands 
    
    def prepare(self, x):
        x_img = self.get_sar(x)
        return (x_img,)