import hydra
from src.dataset.data_module import DataModule, TrainDataset
from src.models.model_module import ModelModule
from src.utils.mlflow import update_pretrained_weights
from lightning.pytorch.trainer.trainer import Trainer
from tempfile import TemporaryDirectory
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import mlflow
from mlflow.pytorch import autolog, log_model
from time import time
import torch 
from datetime import datetime
from pydoc import locate
from tqdm import trange
import numpy as np

        
@hydra.main(version_base=None, config_path='conf', config_name='config.yaml')
def train(cfg):
    torch.set_float32_matmul_precision('high')
    
    mlflow.set_experiment(experiment_name = cfg.site.name)
    

    parent_runs = mlflow.search_runs(
        filter_string = f'run_name = "{cfg.exp.name}"'
        )['run_id']
    assert len(parent_runs) == 1, 'Must have 1 run to retrain.'
    parent_run_id = parent_runs[0]
    parent_run_name = None

    
    with mlflow.start_run(run_name=parent_run_name, run_id=parent_run_id) as parent_run:
        # return()
        print(f'Exp {cfg.exp.name}')
        data_module = DataModule(cfg)
        
        train_dl = data_module.train_dataloader()
        val_dl = data_module.val_dataloader()
        
        n_train_dl = len(train_dl)
        n_val_dl = len(val_dl)
        
        train_dl = iter(train_dl)
        val_dl = iter(val_dl)
        
        train_means, train_sum = 0, 0
        for _ in trange(n_train_dl):
            batch = next(train_dl)
            #train_means += (batch[0]['cloud'].mean(axis=(2,3)).max(axis=1)[0] > 0.2).sum().item()
            train_means += ((batch[0]['cloud'].max(axis=1)[0] > 0.5).float().mean(axis=(1,2)) > 0.5).sum().item()
            train_sum += batch[0]['cloud'].shape[0]
            
        val_means, val_sum = 0, 0
        for _ in trange(n_val_dl):
            batch = next(val_dl)
            val_means += ((batch[0]['cloud'].max(axis=1)[0] > 0.5).float().mean(axis=(1,2)) > 0.5).sum().item()
            val_sum += batch[0]['cloud'].shape[0]
            
        mlflow.log_table(
            {
                'training': 100*(train_means / train_sum),
                'total training': train_sum,
                'validation': 100*(val_means / val_sum),
                'total validation': val_sum,
            }, 'train/cloud.json'
        )
        
        
if __name__ == "__main__":
    train()