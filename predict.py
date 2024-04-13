import hydra
from src.dataset.data_module import DataModule, PredDataset
from src.models.model_module import ModelModule
from src.callbacks import PredictionCallback
from src.utils.ops import save_geotiff, possible_combinations
from lightning.pytorch.trainer.trainer import Trainer
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader
import mlflow
from time import time
import numpy as np
from pathlib import Path
from skimage.transform import rescale
        
@hydra.main(version_base=None, config_path='conf', config_name='config.yaml')
def predict(cfg):
    
    runs = mlflow.search_runs(
        filter_string = f'run_name = "{cfg.exp.name}"'
        )
    parent_run_id = runs['run_id'][0]

    with mlflow.start_run(run_id=parent_run_id) as parent_run:
        #data_module = DataModule(cfg)
        
        with TemporaryDirectory() as tempdir:
            
            imgs_combinations = PredDataset.test_combinations(cfg)
            for img_combination in imgs_combinations:

                predict_dataset = PredDataset(cfg, img_combination)
                gen_opt_imgs, gen_sar_imgs = predict_dataset.generate_sample_images(img_combination)
                for opt_i, sample_opt in enumerate(gen_opt_imgs):
                    mlflow.log_image(sample_opt, f'combination_{img_combination}/opt_{img_combination[0][opt_i]}.jpg')
                for sar_i, sample_sar in enumerate(gen_sar_imgs):
                    mlflow.log_image(sample_sar[:,:,0], f'combination_{img_combination}/sar_0_{img_combination[1][sar_i]}.jpg')
                    mlflow.log_image(sample_sar[:,:,1], f'combination_{img_combination}/sar_1_{img_combination[1][sar_i]}.jpg')

                for model_i in range (cfg.general.n_models):
                    
                    run_model_id = mlflow.search_runs(
                        filter_string = f'run_name = "model_{model_i}" AND params.parent_run_id = "{parent_run_id}"'
                        )['run_id'][0]
                    
                    with mlflow.start_run(run_id=run_model_id, nested=True):
                        #model_id = f'runs:/{run_model_id}/model'
                        #model_module = mlflow.pytorch.load_model(model_id)
                        model_module = ModelModule(cfg)
                        # model_module_b = ModelModule(cfg)
                        # model_module_b.model = model_module.model
                        # model_module = model_module_b
                        
                        pred_callback = PredictionCallback(cfg)
                        callbacks = [pred_callback]
                        
                        trainer = Trainer(
                            accelerator=cfg.general.accelerator.name,
                            devices=cfg.general.accelerator.devices,
                            logger = False,
                            callbacks=callbacks,
                            enable_progress_bar=False,
                        )
                        
                        t0 = time()
                        #for overlap in cfg.exp.pred_params.overlaps:
                        #for discard_border in cfg.exp.pred_params.discard_borders:
                        #    predict_dataset.set_overlap(discard_border)
                        predict_dataloader = DataLoader(predict_dataset, batch_size=cfg.exp.pred_params.batch_size, num_workers=1)
                        
                        trainer.predict(
                            model=model_module,
                            dataloaders=predict_dataloader,
                            return_predictions=False
                        )
                        pred = pred_callback.get_final_image()
                        #pred_int = (pred*255.0).astype(np.uint8)
                        base_image = Path(cfg.path.opt) / cfg.site.original_data.opt.train.imgs[0]
                        
                        elapsed_time = (time() - t0) / 60.0
                        
                        pred_file_path= Path(tempdir) / f'{cfg.site.name}:{cfg.exp.name}:{img_combination}:{model_i}.tif'
                        save_geotiff(base_image, pred_file_path, pred, 'float')
                        # pred_0_file_path = Path(tempdir) / f'{cfg.site.name}:{cfg.exp.name}:{img_combination}:{model_i}_0.tif'
                        # save_geotiff(base_image, pred_0_file_path, np.round(255.0*pred[:,:,0]).astype(np.uint8), 'byte')
                        # pred_1_file_path = Path(tempdir) / f'{cfg.site.name}:{cfg.exp.name}:{img_combination}:{model_i}_1.tif'
                        # save_geotiff(base_image, pred_1_file_path, np.round(255.0*pred[:,:,1]).astype(np.uint8), 'byte')
                        
                        #pred_callback.reset_image()
                        
                        preview = rescale(pred[:,:,0:3], 0.1, channel_axis=2, preserve_range=True)
                        preview = 255 * preview 
                        preview = np.clip(preview.astype(np.int32), 0, 255)
                        
                        mlflow.log_metric('prediction_time', elapsed_time)
                        mlflow.log_artifact(pred_file_path, 'predictions')
                        #mlflow.log_artifact(pred_1_file_path, 'predictions')
                        mlflow.log_image(preview, f'predictions/preview_{cfg.site.name}:{cfg.exp.name}:{img_combination}:{model_i}.jpg')

if __name__ == "__main__":
    predict()