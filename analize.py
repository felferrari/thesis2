import hydra
from src.dataset.data_module import PredDataset
from src.models.model_module import AnalizeModelModule
from src.callbacks import PredictionCallback
from src.utils.ops import save_geotiff
from lightning.pytorch.trainer.trainer import Trainer
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader
import mlflow
from time import time
import numpy as np
from pathlib import Path
from skimage.transform import rescale
import torch
from multiprocessing import Process

from torchvision import models
from torchvision import transforms

from captum.attr import visualization as viz
from captum.attr import LayerGradCam, FeatureAblation, LayerActivation, LayerAttribution


def agg_segmentation_wrapper(model):
    def agg(inp, out_max):
        model_out = model(inp)
        # Creates binary matrix with 1 for original argmax class for each pixel
        # and 0 otherwise. Note that this may change when the input is ablated
        # so we use the original argmax predicted above, out_max.
        selected_inds = torch.zeros_like(model_out[0:1]).scatter_(1, out_max, 1)
        return (model_out * selected_inds).sum(dim=(2,3))
    return agg

@hydra.main(version_base=None, config_path='conf', config_name='config.yaml')
def analize(cfg):
    torch.set_float32_matmul_precision('high')
    mlflow.set_experiment(experiment_name = cfg.site.name)
    
    runs = mlflow.search_runs(
        filter_string = f'run_name = "{cfg.exp.name}"'
        )
    parent_run_id = runs['run_id'][0]

    with mlflow.start_run(run_id=parent_run_id) as parent_run:
        total_t0 = time()
    
        imgs_combinations = PredDataset.test_combinations(cfg)
        for img_comb_i, img_combination in enumerate(imgs_combinations):
            p = Process(target=predict_models, args=(cfg, img_comb_i, img_combination, parent_run.info.run_id))
            p.start()
            p.join()
            
        mlflow.log_metric('total_pred_time', (time() - total_t0) / 60.)



def predict_models(cfg, img_comb_i, img_combination, parent_run_id):
    print(f'Predicting Combination {img_comb_i}')
    predict_dataset = PredDataset(cfg, img_combination)
    
    pred_sum = None
    t0 = time()
    with TemporaryDirectory() as tempdir:
        for model_i in range (cfg.general.n_models):
            run_model_id = mlflow.search_runs(
                filter_string = f'run_name = "model_{model_i}" AND params.parent_run_id = "{parent_run_id}"'
                )['run_id'][0]
            
            with mlflow.start_run(run_id=run_model_id, nested=True) as model_run:
                model_id = f'runs:/{run_model_id}/model'
                #model_module = mlflow.pytorch.load_model(model_id)
                
                analize_model_module = mlflow.pytorch.load_model(model_id).model
                analize_model_module.eval()
                analize_model_module.to('cuda:0')
                
                predict_dataloader = DataLoader(predict_dataset, 
                                                batch_size=cfg.exp.pred_params.analize_batch_size, 
                                                num_workers=1,
                                                )
                
                for x, idx, label in predict_dataloader:
                    for k in x.keys():
                        x[k] = x[k].to('cuda:0')
                    x = analize_model_module.prepare(x)
                    
                    out = analize_model_module(x)
                    out_max = torch.argmax(out, dim=1, keepdim=True)
                    
                    fa = FeatureAblation(agg_segmentation_wrapper(analize_model_module))
                    fa_attr = fa.attribute(x, feature_mask=out_max, perturbations_per_eval=2, target=1)
                
                

        
if __name__ == "__main__":
    analize()