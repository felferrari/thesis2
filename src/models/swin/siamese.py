from pydoc import locate
from src.models.swin.layers import SwinEncoder, SwinDecoder, SwinClassifier, SwinPoolings
from torch import nn
import torch
from abc import abstractmethod
from einops import rearrange


class GenericSwinSiamese(nn.Module):
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
        temp_fusion,
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
        self.temp_fusion = locate(temp_fusion)
        self.construct_model()

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
        self.fusion = self.temp_fusion(self.base_dim, self.n_blocks)
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
        x_0 = self.encoder(x[0])
        x_1 = self.encoder(x[1])
        x = self.fusion([x_0, x_1])
        x = self.decoder(x)
        x = self.classifier(x)
        return x

class SiameseOpt(GenericSwinSiamese):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.general.n_opt_bands
    
    def prepare(self, x):
        return (x['opt'][:, 0], x['opt'][:, 1])
    
class SiameseSAR(GenericSwinSiamese):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.general.n_sar_bands
    
    def prepare(self, x):
        return (x['sar'][:, 0], x['sar'][:, 1])
    
class GenericSwinSiamesePrevDef(GenericSwinSiamese):
    def construct_model(self):
        super().construct_model()
        self.prev_def_poolings = SwinPoolings(self.n_blocks)
        
    def forward(self, x):
        x_0 = self.encoder(x[0])
        x_1 = self.encoder(x[1])
        x_2 = self.prev_def_poolings(x[2])
        x = self.fusion([x_0, x_1, x_2])
        x = self.decoder(x)
        x = self.classifier(x)
        return x
    
class SiameseOptPrevMap(GenericSwinSiamesePrevDef):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.general.n_opt_bands
    
    def prepare(self, x):
        return (x['opt'][:, 0], x['opt'][:, 1], x['previous'])
    
class SiameseSARPrevMap(GenericSwinSiamesePrevDef):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.general.n_sar_bands
    
    def prepare(self, x):
        return (x['sar'][:, 0], x['sar'][:, 1], x['previous'])