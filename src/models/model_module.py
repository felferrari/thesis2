from typing import Any
import lightning as L
from pydoc import locate
from torch.nn import Softmax
import pandas as pd
from torchvision.transforms import v2
from torchmetrics.classification import MulticlassF1Score

class Augmentation():
    def __init__(self, cfg) -> None:
        self.transforms = v2.RandomChoice([
            #v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])
    def __call__(self, x, label):
        x, label = self.transforms(x, label)
        return x, label
        

class Normalization():
    def __init__(self, cfg) -> None:
        opt_df = pd.read_csv(cfg.path.prepared.opt_statistics)
        sar_df = pd.read_csv(cfg.path.prepared.sar_statistics)
        
        opt_df = opt_df[opt_df['cond'] == cfg.exp.opt_condition]
        sar_df = sar_df[sar_df['cond'] == cfg.exp.sar_condition]
        
        if len(opt_df) > 0:
            opt_means =  list(opt_df.sort_values(by='band')['mean'])
            opt_stds =  list(opt_df.sort_values(by='band')['std'])
            # opt_means =  list(opt_df.sort_values(by='band')['min'])
            # opt_stds =  list(opt_df.sort_values(by='band')['delta'])
            self.opt_transforms = v2.Compose([
                v2.Normalize(mean=opt_means, std=opt_stds),
            ])
        else:
            self.opt_transforms = None
            
        if len(sar_df) > 0:
            sar_means =  list(sar_df.sort_values(by='band')['mean'])
            sar_stds =  list(sar_df.sort_values(by='band')['std'])
            # sar_means =  list(sar_df.sort_values(by='band')['min'])
            # sar_stds =  list(sar_df.sort_values(by='band')['delta'])
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
        
        self.train_metric = MulticlassF1Score(num_classes = cfg.general.n_classes, average= 'none')
        self.val_metric = MulticlassF1Score(num_classes = cfg.general.n_classes, average= 'none')
        
    def training_step(self, batch, batch_idx):
        x, label = batch
        x, label = self.augmentation(x, label)
        x = self.normalization(x)
        x = self.model.prepare(x)
        y_hat = self.model(x)
        loss = self.criterion(y_hat, label)
        self.train_metric.update(y_hat, label)
        self.log('train_loss', loss.detach().cpu().item(), prog_bar=True, on_epoch=True, on_step = False)
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        train_metric = self.train_metric.compute()
        self.log('train_f1_score_0', train_metric.detach().cpu().numpy()[0])
        self.log('train_f1_score_1', train_metric.detach().cpu().numpy()[1])
        self.log('train_f1_score_2', train_metric.detach().cpu().numpy()[2])
        self.log('train_f1_score_3', train_metric.detach().cpu().numpy()[3])
        return super().on_train_epoch_end()
        
    def validation_step(self, batch, batch_idx):
        x, label = batch
        x = self.normalization(x)
        x = self.model.prepare(x)
        y_hat = self.model(x)
        loss = self.criterion(y_hat, label)
        self.val_metric.update(y_hat, label)
        self.log('val_loss', loss.detach().cpu().item(), prog_bar=True, on_epoch=True, on_step = False)
        return loss
    
    def on_validation_epoch_end(self) -> None:
        val_metric = self.val_metric.compute()
        self.log('val_f1_score_0', val_metric.detach().cpu().numpy()[0])
        self.log('val_f1_score_1', val_metric.detach().cpu().numpy()[1])
        self.log('val_f1_score_2', val_metric.detach().cpu().numpy()[2])
        self.log('val_f1_score_3', val_metric.detach().cpu().numpy()[3])
        return super().on_validation_epoch_end()
    
    def predict_step(self, batch, batch_idx):
        x, idx = batch
        x = self.normalization(x)
        x = self.model.prepare(x)
        y_hat = self.model(x)
        y_hat = self.pred_softmax(y_hat)
        return y_hat
        
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), **self.optimizer_params)