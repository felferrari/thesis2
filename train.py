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
    
    mlflow.set_experiment(experiment_name = cfg.site.name)
    
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
            'opt_cond': cfg.exp.opt_condition,
            'sar_cond': cfg.exp.sar_condition,
            'site': cfg.site.name
        })
        total_t0 = time()
        data_module = DataModule(cfg)
        
        with TemporaryDirectory() as tempdir:
            
            for model_i in range(cfg.general.n_models):
                
                for model_attempt in range(cfg.exp.train_params.max_retrain_models + 1):
                
                    with mlflow.start_run(run_name=f'model_{model_i}', nested=True, log_system_metrics=True) as model_run:
                        
                        mlflow.log_params({
                            'parent_run_id': parent_run.info.run_id
                        })
                        mlflow.set_tags({
                            'Training': 'executing',
                            'Retrain attempt': model_attempt
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
                            limit_train_batches=cfg.exp.train_params.limit_train_batches,
                            limit_val_batches=cfg.exp.train_params.limit_val_batches,
                            max_epochs = 3 #cfg.exp.train_params.max_epochs,
                            #min_epochs = cfg.exp.train_params.min_epochs
                        )
                        t0 = time()
                        trainer.fit(
                            model=model_module,
                            datamodule=data_module,
                        )
                        mlflow.log_metric('train_time', (time() - t0) / 60.0)
                        
                        best_model = ModelModule.load_from_checkpoint(checkpoint_callback.best_model_path)
                        log_model(best_model, 'model')
                        
                        last_run_id = model_run.info.run_id
                        if checkpoint_callback.best_model_score >= cfg.exp.train_params.min_val_f1:
                            mlflow.set_tag('Training', 'success')
                        else:
                            mlflow.set_tag('Training', 'failed')
                    if checkpoint_callback.best_model_score >= cfg.exp.train_params.min_val_f1:
                        break
                    else:
                        mlflow.delete_run(run_id=last_run_id)
                    

        mlflow.log_metric('total_train_time', (time() - total_t0) / 60.)

if __name__ == "__main__":
    train()