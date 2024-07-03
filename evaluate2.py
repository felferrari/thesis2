import hydra
from src.dataset.data_module import PredDataset
from src.utils.ops import save_geotiff, load_ml_image, load_sb_image
from src.utils.generate import read_imgs
from src.utils.roi import gen_roi_figures
from tempfile import TemporaryDirectory
import mlflow
from time import time
import numpy as np
from pathlib import Path
from skimage.morphology import area_opening
import torch
from operator import itemgetter
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

eval_version = 3

@hydra.main(version_base=None, config_path='conf', config_name='config.yaml')
def eval(cfg):
    torch.set_float32_matmul_precision('high')
    mlflow.set_experiment(experiment_name = cfg.site.name)
    
    runs = mlflow.search_runs(
        filter_string = f'run_name = "{cfg.exp.name}"'
        )
    parent_run_id = runs['run_id'][0]
    
    #with mlflow.start_run(run_id=parent_run_id) as parent_run:
    total_t0 = time()

    imgs_combinations = PredDataset.test_combinations(cfg)
    args_list = []
    for img_comb_i, img_combination in enumerate(imgs_combinations):
        args_list.append((cfg, img_comb_i, img_combination, parent_run_id))
        
        
    with Pool(cfg.exp.eval_params.n_rois_processes) as pool:
        _ = pool.starmap(gen_roi_figures, args_list)
        
    #metrics_ = np.stack(metrics, axis= 0).sum(axis=0)
    
 
        
if __name__ == "__main__":
    eval()