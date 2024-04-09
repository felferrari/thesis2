from src.models.layers import ResUnetEncoder, ResUnetDecoder, ResUnetClassifier, ResUnetDecoderJF, ResUnetDecoderJFNoSkip, ResUnetRegressionClassifier
from torch import nn
import torch
from abc import abstractmethod
from einops import rearrange
#from ..utils import ModelModule, ModelModuleMultiTask

class GenericModel(nn.Module):
    def __init__(self, depths, n_classes, *args, **kargs) -> None:
        super().__init__(*args, **kargs)
        self.n_classes = n_classes
        self.depths = depths

    def get_opt(self, x):
        return rearrange(x['opt'], 'b i c h w -> b (i c) h w')
    
    def get_sar(self, x):
        return rearrange(x['sar'], 'b i c h w -> b (i c) h w')

class GenericResunet(GenericModel):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.encoder = ResUnetEncoder(self.depths)
        self.decoder = ResUnetDecoder(self.depths)
        self.classifier = ResUnetClassifier(self.depths[0], self.n_classes)

    @abstractmethod
    def prepare_input(self, x):
        pass
    
    def forward(self, x):
        #x = self.prepare_input(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)
        return x

class ResUnetOpt(GenericResunet):
    def prepare_input(self, x):
        x_img = self.get_opt(x)
        x = torch.cat((x_img, x['previous']), dim=1)
        return x
    
    def prepare(self, x):
        x_img = self.get_opt(x)
        x = torch.cat((x_img, x['previous']), dim=1)
        return x
    
class ResUnetSAR(GenericResunet):
    def prepare_input(self, x):
        x_img = self.get_sar(x)
        x = torch.cat((x_img, x[2]), dim=1)
        return x
    