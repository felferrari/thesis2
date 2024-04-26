from pydoc import locate
from src.models.resunet.layers import ResUnetEncoder, ResUnetDecoder, ResUnetClassifier, IdentityFusion
from torch import nn
from abc import abstractmethod

class GenericResunetSiamese(nn.Module):
    def __init__(self, temp_fusion, depths, n_classes, in_dims, *args, **kargs) -> None:
        super().__init__(*args, **kargs)
        self.n_classes = n_classes
        self.depths = depths
        self.in_dims = in_dims
        self.temp_fusion = locate(temp_fusion)
        self.construct_model()
        
    def construct_model(self):
        self.encoder = ResUnetEncoder(self.in_dims, self.depths)
        self.fusion = self.temp_fusion(self.depths)
        self.decoder = ResUnetDecoder(self.depths)
        self.classifier = ResUnetClassifier(self.depths, self.n_classes)
        
    def forward(self, x):
        x_0 = self.encoder(x[0])
        x_1 = self.encoder(x[1])
        x = self.fusion([x_0, x_1])
        x = self.decoder(x)
        x = self.classifier(x)
        
        return x
        
class SiameseOpt(GenericResunetSiamese):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.general.n_opt_bands
    
    def prepare(self, x):
        return (x['opt'][:, 0], x['opt'][:, 1])
    
class SiameseSAR(GenericResunetSiamese):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.general.n_sar_bands
    
    def prepare(self, x):
        return (x['sar'][:, 0], x['sar'][:, 1])