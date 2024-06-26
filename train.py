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

        
@hydra.main(version_base=None, config_path='conf', config_name='config.yaml')
def train(cfg):
    torch.set_float32_matmul_precision('high')
    
    mlflow.set_experiment(experiment_name = cfg.site.name)
    
    if cfg.retrain_model is None:
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
        parent_run_name = parent_run_name=cfg.exp.name
        parent_run_id = None
    else:
        parent_runs = mlflow.search_runs(
            filter_string = f'run_name = "{cfg.exp.name}"'
            )['run_id']
        assert len(parent_runs) == 1, 'Must have 1 run to retrain.'
        parent_run_id = parent_runs[0]
        parent_run_name = None
        run_models = mlflow.search_runs(
            filter_string = f'run_name = "model_{cfg.retrain_model}" AND params.parent_run_id = "{parent_run_id}"'
            )['run_id']
        for nested_run_id in run_models:
                mlflow.delete_run(run_id=nested_run_id)
        
    
    with mlflow.start_run(run_name=parent_run_name, run_id=parent_run_id) as parent_run:
        mlflow.set_tags({
            'opt_cond': cfg.exp.opt_condition,
            'sar_cond': cfg.exp.sar_condition,
            'site': cfg.site.name,
            'exp_code': cfg.exp.code
        })
        total_t0 = time()
        data_module = DataModule(cfg)
        
        mlflow.log_params({
            'train_patches': TrainDataset(cfg = cfg, mode = 'train').n_patches,
            'val_patches': TrainDataset(cfg = cfg, mode = 'validation').n_patches,
        })
        
        
        if cfg.retrain_model is None:
            models_n = range(cfg.general.n_models)
        else:
            models_n = [cfg.retrain_model]
        for model_i in models_n:
        
            for model_attempt in range(cfg.exp.train_params.max_retrain_models + 1):
                model_module = ModelModule(cfg)
                
                model_module = update_pretrained_weights(cfg, model_module, model_i)
                
                with TemporaryDirectory() as tempdir:
                        
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
                        earlystop_callback,
                    ]

                    with mlflow.start_run(run_name=f'model_{model_i}', nested=True) as model_run:
                        
                        mlflow.log_params({
                            'parent_run_id': parent_run.info.run_id,
                            'Training Params': dict(cfg.exp.train_params),
                            'Model Params': dict(cfg.exp.model),
                        })
                        mlflow.set_tags({
                            'Training': 'executing',
                            'Retrain attempt': model_attempt
                            })
                        
                        autolog(
                            log_models=False,
                            checkpoint=False,
                            silent=True
                        )
                        t0 = time()
                        if cfg.exp.train_params.warmup_epochs is not None:
                            model_module.prefix = 'warmup_'
                            trainer = Trainer(
                                accelerator=cfg.general.accelerator.name,
                                devices=cfg.general.accelerator.devices,
                                logger = False,
                                enable_progress_bar=True,
                                enable_checkpointing = False,
                                limit_train_batches=cfg.exp.train_params.limit_train_batches,
                                limit_val_batches=cfg.exp.train_params.limit_val_batches,
                                max_epochs = cfg.exp.train_params.warmup_epochs,
                            )
                            trainer.fit(
                                model=model_module,
                                datamodule=data_module,
                            )
                            model_module.prefix = ''
                        
                        trainer = Trainer(
                            accelerator=cfg.general.accelerator.name,
                            devices=cfg.general.accelerator.devices,
                            logger = False,
                            callbacks=callbacks,
                            enable_progress_bar=True,
                            limit_train_batches=cfg.exp.train_params.limit_train_batches,
                            limit_val_batches=cfg.exp.train_params.limit_val_batches,
                            max_epochs = cfg.exp.train_params.max_epochs,
                        )
                        trainer.fit(
                            model=model_module,
                            datamodule=data_module,
                        )
                        mlflow.log_metric('train_time', (time() - t0) / 60.0)
                        
                        best_model = ModelModule.load_from_checkpoint(checkpoint_callback.best_model_path)
                        log_model(best_model, 'model')
                        
                        last_run_id = model_run.info.run_id
                        #if checkpoint_callback.best_model_score >= cfg.exp.train_params.min_val_f1:
                        #if trainer.current_epoch >= cfg.exp.train_params.min_epochs and checkpoint_callback.best_model_score >= cfg.exp.train_params.min_val_f1:
                        if checkpoint_callback.best_model_score >= cfg.exp.train_params.min_val_f1:
                            mlflow.set_tag('Training', 'success')
                            mlflow.set_tag('Train_time', datetime.now())
                        else:
                            mlflow.set_tag('Training', 'failed')
                    #if checkpoint_callback.best_model_score >= cfg.exp.train_params.min_val_f1:
                    #if trainer.current_epoch >= cfg.exp.train_params.min_epochs and checkpoint_callback.best_model_score >= cfg.exp.train_params.min_val_f1:
                    if checkpoint_callback.best_model_score >= cfg.exp.train_params.min_val_f1:
                        break
                    else:
                        mlflow.delete_run(run_id=last_run_id)
                    

        mlflow.log_metric('total_train_time', (time() - total_t0) / 60.)

if __name__ == "__main__":
    train()