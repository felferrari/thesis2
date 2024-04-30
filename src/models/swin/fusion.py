from src.models.swin.single import GenericSwin
from src.models.swin.layers import SwinEncoder, SwinDecoder, SwinClassifier, BNIdentity
from abc import abstractmethod
from pydoc import locate
import torch
class PixelLevel(GenericSwin):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.exp.n_opt_imgs * cfg.general.n_opt_bands + cfg.exp.n_sar_imgs * cfg.general.n_sar_bands + 1 
    
    def prepare(self, x):
        x_opt = self.get_opt(x)
        x_sar = self.get_sar(x)
        #x = torch.cat((x_img, x['previous']), dim=1)
        return (x_opt, x_sar, x['previous'])
    
class MiddleFeatureLevel(GenericSwin):
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
        self.encoder_opt = SwinEncoder(
            input_depth = self.in_dims[0], 
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads = self.n_heads,
            n_blocks = self.n_blocks
            )
        self.encoder_sar = SwinEncoder(
            input_depth = self.in_dims[1], 
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads = self.n_heads,
            n_blocks = self.n_blocks
            )
        self.fusion = self.modal_fusion(self.base_dim, self.n_blocks)
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
        self.encoder_opt = SwinEncoder(
            input_depth = self.in_dims[0], 
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads = self.n_heads,
            n_blocks = self.n_blocks
            )
        self.encoder_sar = SwinEncoder(
            input_depth = self.in_dims[1], 
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads = self.n_heads,
            n_blocks = self.n_blocks
            )
        self.bn_opt = BNIdentity()
        self.bn_sar = BNIdentity()
        self.decoder_opt = SwinDecoder(
            base_dim=self.base_dim,
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size
            )
        self.decoder_sar = SwinDecoder(
            base_dim=self.base_dim,
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size
            )
        self.fusion = self.modal_fusion(self.base_dim)
        self.classifier = SwinClassifier(
            self.base_dim, 
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size,
            n_classes = self.n_classes
            )

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
    