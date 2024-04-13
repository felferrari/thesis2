import hydra
from src.dataset.data_module import DataModule
from src.models.model_module import ModelModule
from lightning.pytorch.trainer.trainer import Trainer
from tempfile import TemporaryDirectory
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import mlflow
from mlflow.pytorch import autolog, log_model
from time import time
        
@hydra.main(version_base=None, config_path='conf', config_name='config.yaml')
def train(cfg):
    
    runs = mlflow.search_runs(
        filter_string = f'run_name = "{cfg.exp.name}"'
        )
    for run_id in runs['run_id']:
        nested_runs = mlflow.search_runs(
            filter_string = f'params.parent_run_id = "{run_id}"'
            )
        for nested_run_id in nested_runs['run_id']:
            mlflow.delete_run(run_id=nested_run_id)
        mlflow.delete_run(run_id=run_id)
    
    with mlflow.start_run(run_name=cfg.exp.name) as parent_run:
        mlflow.set_tags({
            'Optical Condition': cfg.exp.opt_condition,
            'SAR Condition': cfg.exp.sar_condition
        })
        data_module = DataModule(cfg)
        
        with TemporaryDirectory() as tempdir:
            
            for model_i in range (cfg.general.n_models):
                
                with mlflow.start_run(run_name=f'model_{model_i}', nested=True, log_system_metrics=True):
                    
                    mlflow.log_params({
                        'parent_run_id': parent_run.info.run_id
                    })
                    
                    autolog(
                        log_models=False,
                        checkpoint=False,
                    )
                    
                    model_module = ModelModule(cfg)
                    
                    checkpoint_callback = ModelCheckpoint(
                        dirpath=tempdir,
                        filename=f'model_{model_i}',
                        monitor='val_f1_score_1',
                        mode = 'max',
                        verbose = True
                    )
                    
                    earlystop_callback = EarlyStopping(
                        monitor= 'val_f1_score_1',
                        mode='max',
                        verbose = True,
                        **dict(cfg.exp.train_params.early_stop_params)
                    )
                    
                    callbacks = [
                        checkpoint_callback,
                        earlystop_callback
                    ]
                    
                    trainer = Trainer(
                        accelerator=cfg.general.accelerator.name,
                        devices=cfg.general.accelerator.devices,
                        logger = False,
                        callbacks=callbacks,
                        enable_progress_bar=False,
                        max_epochs = cfg.exp.train_params.max_epochs
                    )
                    t0 = time()
                    trainer.fit(
                        model=model_module,
                        datamodule=data_module,
                    )
                    elapsed_time = (time() - t0) / 60.0
                    
                    mlflow.log_metric('train_time', elapsed_time)
                    
                    best_model = ModelModule.load_from_checkpoint(checkpoint_callback.best_model_path)
                    log_model(best_model, 'model')

if __name__ == "__main__":
    train()