from typing import Any, Dict
import pandas as pd
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform
from einops import rearrange

class NoTransform(Transform):
    def __init__(self) -> None:
        super().__init__()
        
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt

class Augmentation():
    def __init__(self, cfg) -> None:
        self.transforms = v2.RandomChoice([
            v2.RandomVerticalFlip(p=1),
            v2.RandomHorizontalFlip(p=1),
        ])
        self.cfg = cfg
    def __call__(self, x, label):
        # if  self.cfg.exp.n_opt_imgs > 0:
        #     x['opt'] = rearrange(x['opt'], 'b n c h w -> b (n c) h w')
        # if  self.cfg.exp.n_sar_imgs > 0:
        #     x['sar'] = rearrange(x['sar'], 'b n c h w -> b (n c) h w')
        x, label = self.transforms(x, label)
        # if  self.cfg.exp.n_opt_imgs > 0:
        #     x['opt'] = rearrange(x['opt'], 'b (n c) h w -> b n c h w', n = self.cfg.exp.n_opt_imgs)
        # if  self.cfg.exp.n_sar_imgs > 0:
        #     x['sar'] = rearrange(x['sar'], 'b (n c) h w -> b n c h w', n = self.cfg.exp.n_sar_imgs)
        return x, label
        

class Normalization():
    def __init__(self, cfg) -> None:
        opt_df = pd.read_csv(cfg.path.prepared.opt_statistics)
        sar_df = pd.read_csv(cfg.path.prepared.sar_statistics)
        
        opt_df = opt_df[opt_df['cond'] == cfg.exp.opt_condition]
        sar_df = sar_df[sar_df['cond'] == cfg.exp.sar_condition]
        
        if len(opt_df) > 0:
            # opt_means =  list(opt_df.sort_values(by='band')['mean'])
            # opt_stds =  list(opt_df.sort_values(by='band')['std'])
            opt_means =  list(opt_df.sort_values(by='band')['min'])
            opt_stds =  list(opt_df.sort_values(by='band')['delta'])
            self.opt_transforms = v2.Compose([
                v2.Normalize(mean=opt_means, std=opt_stds),
            ])
        else:
            self.opt_transforms = None
            
        if len(sar_df) > 0:
            # sar_means =  list(sar_df.sort_values(by='band')['mean'])
            # sar_stds =  list(sar_df.sort_values(by='band')['std'])
            sar_means =  list(sar_df.sort_values(by='band')['min'])
            sar_stds =  list(sar_df.sort_values(by='band')['delta'])
            self.sar_transforms = v2.Compose([
                v2.Normalize(mean=sar_means, std=sar_stds),
            ])
        else:
            self.sar_transforms = None

    def __call__(self, x):
        if self.opt_transforms: 
            x['opt'] = self.opt_transforms(x['opt'])
        if self.sar_transforms:
            x['sar'] = self.sar_transforms(x['sar'])
        return x
    