from typing import Any
import lightning as L
from pydoc import locate
from torch.nn import Softmax
from torchmetrics.classification import MulticlassF1Score
from src.models.augmentation import Augmentation, Normalization
    
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
        # x, label = self.augmentation(x, label)
        # x = self.normalization(x)
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
        # x = self.normalization(x)
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
        # x = self.normalization(x)
        x = self.model.prepare(x)
        y_hat = self.model(x)
        #y_hat = self.pred_softmax(y_hat)
        return y_hat
        
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), **self.optimizer_params)