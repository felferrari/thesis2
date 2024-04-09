from typing import Any
import lightning as L
from pydoc import locate

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
        
    def training_step(self, batch, batch_idx):
        x, label = batch
        x = self.model.prepare(x)
        
        
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), **self.optimizer_params)