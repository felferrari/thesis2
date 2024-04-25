from typing import Any
import lightning as L
from pydoc import locate
from torch.nn import Softmax
from torchmetrics.classification import MulticlassF1Score
from torch import nn
import torch
from einops import rearrange
from src.attributes import IntegratedGradients
    
class ModelModule(L.LightningModule):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        model_params = dict(cfg.exp.model.params)
        model_params['n_classes'] = cfg.general.n_classes
        model_class = locate(cfg.exp.model.target) 
        model_params['in_dims'] = model_class.get_input_dims(cfg)
        self.model = model_class(**model_params)
        
        train_params = dict(cfg.exp.train_params)
        self.optimizer_class = locate(train_params['optimizer']['target'])
        self.optimizer_params = train_params ['optimizer']['params']
        
        self.scheduler_class = locate(train_params['scheduler']['target'])
        self.scheduler_params = train_params ['scheduler']['params']
        
        self.criterion = locate(train_params['criterion']['target'])(**train_params['criterion']['params'])
        
        self.train_metric = MulticlassF1Score(num_classes = cfg.general.n_classes, average= 'none')
        self.val_metric = MulticlassF1Score(num_classes = cfg.general.n_classes, average= 'none')
        
        self.prefix = ''
        
    def training_step(self, batch, batch_idx):
        x, label = batch
        x = self.model.prepare(x)
        y_hat = self.model(x)
        loss = self.criterion(y_hat, label)
        self.train_metric.update(y_hat, label)
        self.log(f'{self.prefix}train_loss', loss.detach().cpu().item(), prog_bar=True, on_epoch=True, on_step = False)
        
        return loss
    
    def on_train_epoch_start(self) -> None:
        self.log(f'{self.prefix}lr', self.lr_schedulers().get_last_lr()[0])
        return super().on_train_epoch_start()
    
    def on_train_epoch_end(self) -> None:
        train_metric = self.train_metric.compute()
        self.log(f'{self.prefix}train_f1_score_0', train_metric.detach().cpu().numpy()[0])
        self.log(f'{self.prefix}train_f1_score_1', train_metric.detach().cpu().numpy()[1])
        self.log(f'{self.prefix}train_f1_score_2', train_metric.detach().cpu().numpy()[2])
        self.log(f'{self.prefix}train_f1_score_3', train_metric.detach().cpu().numpy()[3])
        return super().on_train_epoch_end()
        
    def validation_step(self, batch, batch_idx):
        x, label = batch
        x = self.model.prepare(x)
        y_hat = self.model(x)
        loss = self.criterion(y_hat, label)
        self.val_metric.update(y_hat, label)
        self.log(f'{self.prefix}val_loss', loss.detach().cpu().item(), prog_bar=True, on_epoch=True, on_step = False)
        return loss
    
    def on_validation_epoch_end(self) -> None:
        val_metric = self.val_metric.compute()
        self.log(f'{self.prefix}val_f1_score_0', val_metric.detach().cpu().numpy()[0])
        self.log(f'{self.prefix}val_f1_score_1', val_metric.detach().cpu().numpy()[1])
        self.log(f'{self.prefix}val_f1_score_2', val_metric.detach().cpu().numpy()[2])
        self.log(f'{self.prefix}val_f1_score_3', val_metric.detach().cpu().numpy()[3])
        return super().on_validation_epoch_end()
    
    def predict_step(self, batch, batch_idx):
        x, _, _ = batch
        x = self.model.prepare(x)
        y_hat = self.model(x)
        return y_hat
        
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_params)
        scheduler = self.scheduler_class(optimizer, **self.scheduler_params)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1_score_1",
            },
        }
        
        
def create_wrapper(model):
    def fn_wrapper(input):
        x = rearrange(input, 'b c (h w) -> b c h w', h=224, w=224)
        y = model(x)
        y = torch.argmax(y, 1)
        y = rearrange(y, 'b h w-> b (h w)')
        return y
    return fn_wrapper
        
    
    
        
class AnalizeModelModule(L.LightningModule):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.ig = IntegratedGradients(model)
                
    def predict_step(self, batch, batch_idx):
        x, _, y = batch
        x = self.model.prepare(x)
        self.ig.attributes(x)