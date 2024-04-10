import hydra
from src.data.data_module import DataModule, PredDataset
from src.models.model_module import ModelModule
from lightning.pytorch.trainer.trainer import Trainer
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader
from src.callbacks import PredictionCallback
import mlflow
from mlflow.pytorch import autolog, log_model
from time import time
import threading
import queue
        
@hydra.main(version_base=None, config_path='conf', config_name='config.yaml')
def predict(cfg):
    
    runs = mlflow.search_runs(
        filter_string = f'run_name = "{cfg.exp.name}"'
        )
    parent_run_id = runs['run_id'][0]
    nested_runs = mlflow.search_runs(
        filter_string = f'tag.parent_run_id = "{parent_run_id}"'
        )
    
    with mlflow.start_run(run_id=parent_run_id) as parent_run:
        #data_module = DataModule(cfg)
        
        with TemporaryDirectory() as tempdir:
            
            for model_i in range (cfg.general.n_models):
                
                run_model_id = mlflow.search_runs(
                    filter_string = f'run_name = "model_{model_i}" AND tag.parent_run_id = "{parent_run_id}"'
                    )['run_id'][0]
                
                with mlflow.start_run(run_id=run_model_id, nested=True):
                    
                    model_id = f'runs:/{run_model_id}/model'
                    model_module = mlflow.pytorch.load_model(model_id)
                    
                    autolog(
                        log_models=False,
                        checkpoint=False,
                    )
                    
                    imgs_combinations = PredDataset.test_combinations(cfg)
                    for img_combination in imgs_combinations:
                        predict_dataset = PredDataset(cfg, img_combination)
                        predict_dataloader = DataLoader(predict_dataset, batch_size=32)
                        
                        trainer = Trainer(
                            accelerator='gpu',
                            logger = False,
                            enable_progress_bar=True,
                        )
                        
                        
                        t0 = time()
                        trainer.predict(
                            model=model_module,
                            dataloaders=predict_dataloader,
                            return_predictions=False
                        )
                        elapsed_time = time() - t0
                        
                        mlflow.log_metric('pred_time', elapsed_time)



if __name__ == "__main__":
    predict()