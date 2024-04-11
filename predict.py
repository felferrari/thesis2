import hydra
from src.data.data_module import DataModule, PredDataset
from src.models.model_module import ModelModule
from src.callbacks import PredictionCallback
from src.utils.ops import save_geotiff
from lightning.pytorch.trainer.trainer import Trainer
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader
import mlflow
from time import time
import numpy as np
from pathlib import Path
        
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
            
            imgs_combinations = PredDataset.test_combinations(cfg)
            for img_combination in imgs_combinations:
                predict_dataset = PredDataset(cfg, img_combination)
            
                for model_i in range (cfg.general.n_models):
                    
                    pred_callback = PredictionCallback(cfg)
                    
                    callbacks = [pred_callback]
                    
                    run_model_id = mlflow.search_runs(
                        filter_string = f'run_name = "model_{model_i}" AND tag.parent_run_id = "{parent_run_id}"'
                        )['run_id'][0]
                    
                    with mlflow.start_run(run_id=run_model_id, nested=True):
                        
                        model_id = f'runs:/{run_model_id}/model'
                        model_module = mlflow.pytorch.load_model(model_id)
                        
                        model_module_b = ModelModule(cfg)
                        model_module_b.model = model_module.model
                        model_module = model_module_b
                        
                        
                        
                        trainer = Trainer(
                            accelerator='gpu',
                            logger = False,
                            callbacks=callbacks,
                            enable_progress_bar=True,
                        )
                        
                        
                        t0 = time()
                        #for overlap in cfg.exp.pred_params.overlaps:
                        for overlap in cfg.exp.pred_params.strides:
                            predict_dataset.set_overlap(overlap)
                            predict_dataloader = DataLoader(predict_dataset, batch_size=cfg.exp.pred_params.batch_size)
                            
                            trainer.predict(
                                model=model_module,
                                dataloaders=predict_dataloader,
                                return_predictions=False
                            )
                        pred = pred_callback.get_final_image()
                        pred_int = (pred*255.0).astype(np.uint8)
                        base_image = Path(cfg.path.opt) / cfg.site.original_data.opt.train.imgs[0]
                        save_geotiff(base_image, 'test2.tif', pred[:,:,1:2], 'float')
                        save_geotiff(base_image, 'testint.tif', pred_int[:,:,1], 'byte')
                        
                        
                        pred_callback.reset_image()
                        
                        
                        elapsed_time = time() - t0
                        
                        mlflow.log_metric('pred_time', elapsed_time)



if __name__ == "__main__":
    predict()