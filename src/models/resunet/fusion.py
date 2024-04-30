from abc import abstractmethod
from pydoc import locate
from src.models.resunet.single import GenericResunet
from src.models.resunet.layers import ResUnetEncoder, ResUnetDecoder, ResUnetClassifier, BNIdentity, ModalConcat
import torch

class PixelLevel(GenericResunet):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.exp.n_opt_imgs * cfg.general.n_opt_bands + cfg.exp.n_sar_imgs * cfg.general.n_sar_bands + 1 
    
    def prepare(self, x):
        x_opt = self.get_opt(x)
        x_sar = self.get_sar(x)
        #x = torch.cat((x_img, x['previous']), dim=1)
        return (x_opt, x_sar, x['previous'])
    
class MiddleFeatureLevel(GenericResunet):
    def __init__(self, modal_fusion, *args, **kargs) -> None:
        self.modal_fusion = locate(modal_fusion)
        super().__init__(*args, **kargs)
        
    @abstractmethod
    def get_input_dims(cfg):
        return (cfg.exp.n_opt_imgs * cfg.general.n_opt_bands + 1,  + cfg.exp.n_sar_imgs * cfg.general.n_sar_bands + 1)
    
    def prepare(self, x):
        x_opt = self.get_opt(x)
        x_sar = self.get_sar(x)
        #x = torch.cat((x_img, x['previous']), dim=1)
        return (x_opt, x_sar, x['previous'])
    
    def construct_model(self):
        self.encoder_opt = ResUnetEncoder(self.in_dims[0], self.depths)
        self.encoder_sar = ResUnetEncoder(self.in_dims[1], self.depths)
        self.fusion = self.modal_fusion(self.depths)
        self.decoder = ResUnetDecoder(self.depths)
        self.classifier = ResUnetClassifier(self.depths, self.n_classes)    

    def forward(self, x):
        x_opt = torch.cat([x[0], x[2]], dim=1)
        x_sar = torch.cat([x[1], x[2]], dim=1)
        x_opt = self.encoder_opt(x_opt)
        x_sar = self.encoder_sar(x_sar)
        x = self.fusion([x_opt, x_sar])
        x = self.decoder(x)
        x = self.classifier(x)
        return x
    
class LateFeatureLevel(MiddleFeatureLevel):
    
    def construct_model(self):
        self.encoder_opt = ResUnetEncoder(self.in_dims[0], self.depths)
        self.encoder_sar = ResUnetEncoder(self.in_dims[1], self.depths)
        self.bn_opt = BNIdentity(self.depths)
        self.bn_sar = BNIdentity(self.depths)
        self.decoder_opt = ResUnetDecoder(self.depths)
        self.decoder_sar = ResUnetDecoder(self.depths)
        self.fusion = self.modal_fusion(self.depths)
        self.classifier = ResUnetClassifier(self.depths, self.n_classes)    

    def forward(self, x):
        x_opt = torch.cat([x[0], x[2]], dim=1)
        x_sar = torch.cat([x[1], x[2]], dim=1)
        x_opt = self.encoder_opt(x_opt)
        x_sar = self.encoder_sar(x_sar)
        x_opt = self.bn_opt(x_opt)
        x_sar = self.bn_sar(x_sar)
        x_opt = self.decoder_opt(x_opt)
        x_sar = self.decoder_sar(x_sar)
        x = self.fusion([x_opt, x_sar])
        x = self.classifier(x)
        return x
    