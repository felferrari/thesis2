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
    run_models = mlflow.search_runs(
        filter_string = f'run_name = "model_{cfg.retrain_model}" AND params.parent_run_id = "{parent_run_id}"'
        )['run_id']
        
    
    with mlflow.start_run(run_name=parent_run_name, run_id=parent_run_id) as parent_run:
        # return()
        print(f'Exp {cfg.exp.name}')
        data_module = DataModule(cfg)
        
        model = ModelModule(cfg).model
        
        model = model.to('cuda:0')
        
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_metric('n_params', count_parameters(model), 0)
        
        # input, label = next(iter(data_module.train_dataloader()))
        # input_l = model.prepare(input)
        # label = label.to('cuda:0')
        
        # input_l = [inp_i.to('cuda:0') for inp_i in input_l]
        # input_l = tuple(input_l)
        
        # train_params = dict(cfg.exp.train_params)
        # criterion = locate(train_params['criterion']['target'])(**train_params['criterion']['params'])
        # criterion = criterion.to('cuda:0')
        
        # optimizer_class = locate(train_params['optimizer']['target'])
        # optimizer_params = train_params ['optimizer']['params']
        
        # optimizer = optimizer_class(model.parameters(), **optimizer_params)
        
        # model.train()
        # for i in range(50):
        #     t0 = time()
        #     for j in range(10):
        #         y_hat = model(input_l)
        #         loss = criterion(y_hat, label)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #     t = time()-t0
        #     mlflow.log_metric('train_10epochs', t, i)
            
        # print('------')
            
        # model.eval()
        # for i in range(50):
        #     t0 = time()
        #     for j in range(10):
        #         model(input_l)
        #     t = time()-t0
        #     mlflow.log_metric('eval_10epochs', t, i)
        
if __name__ == "__main__":
    train()