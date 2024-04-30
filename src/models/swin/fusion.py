from .single import GenericSwin
from abc import abstractmethod

class PixelLevel(GenericSwin):
    @abstractmethod
    def get_input_dims(cfg):
        return cfg.exp.n_opt_imgs * cfg.general.n_opt_bands + cfg.exp.n_sar_imgs * cfg.general.n_sar_bands + 1 
    
    def prepare(self, x):
        x_opt = self.get_opt(x)
        x_sar = self.get_sar(x)
        #x = torch.cat((x_img, x['previous']), dim=1)
        return (x_opt, x_sar, x['previous'])