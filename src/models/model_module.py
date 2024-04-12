from typing import Any
import lightning as L
from pydoc import locate
from torch.nn import Softmax
import pandas as pd
from torchvision.transforms import v2


class Augmentation():
    def __init__(self, cfg) -> None:
        self.transforms = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])
    def __call__(self, x):
        x = self.transforms(x)
        return x
        

class Normalization():
    def __init__(self, cfg) -> None:
        opt_df = pd.read_csv(cfg.path.prepared.opt_statistics)
        sar_df = pd.read_csv(cfg.path.prepared.sar_statistics)
        
        opt_df = opt_df[opt_df['cond'] == cfg.exp.opt_condition]
        sar_df = sar_df[sar_df['cond'] == cfg.exp.sar_condition]
        
        if len(opt_df) > 0:
            opt_means =  list(opt_df.sort_values(by='band')['mean'])
            opt_stds =  list(opt_df.sort_values(by='band')['std'])
            self.opt_transforms = v2.Compose([
                v2.Normalize(mean=opt_means, std=opt_stds),
            ])
        else:
            self.opt_transforms = None
            
        if len(sar_df) > 0:
            sar_means =  list(sar_df.sort_values(by='band')['mean'])
            sar_stds =  list(sar_df.sort_values(by='band')['std'])
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
class ModelModule(L.LightningModule):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        model_params = dict(cfg.exp.model.params)
        model_params['n_classes'] = cfg.general.n_classes
        self.model = locate(cfg.exp.model.target)(**model_params)
        
        train_params = dict(cfg.exp.train_params)
        self.optimizer = locate(train_params['optimizer']['target'])
        self.optimizer_params = train_params ['optimizer']['params']
        
        self.criterion = locate(train_params['criterion']['target'])(**train_params['criterion']['params'])
        
        self.pred_softmax = Softmax(dim=1)
        
        self.augmentation = Augmentation(cfg)
        self.normalization = Normalization(cfg)
        
    def training_step(self, batch, batch_idx):
        x, label = batch
        x = self.augmentation(x)
        x = self.normalization(x)
        x = self.model.prepare(x)
        y_hat = self.model(x)
        loss = self.criterion(y_hat, label)
        self.log('train_loss', loss.detach().cpu().item(), prog_bar=True, on_epoch=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, label = batch
        x = self.normalization(x)
        x = self.model.prepare(x)
        y_hat = self.model(x)
        loss = self.criterion(y_hat, label)
        self.log('val_loss', loss.detach().cpu().item(), prog_bar=True, on_epoch=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, idx = batch
        x = self.normalization(x)
        x = self.model.prepare(x)
        y_hat = self.model(x)
        return self.pred_softmax(y_hat)
        
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), **self.optimizer_params)