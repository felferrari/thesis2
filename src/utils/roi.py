import hydra
from src.dataset.data_module import PredDataset
from src.utils.ops import save_geotiff, load_ml_image, load_sb_image, load_opt_image, load_SAR_image
from src.utils.generate import read_imgs
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
from matplotlib import pyplot as plt
import matplotlib
from tqdm import tqdm

def save_def_class_fig(fp, data):
    fig, ax = plt.subplots(1,1, figsize = (8,8))
    plt.imshow(data[:,:,1], vmin=0, vmax=1, cmap='viridis')
    plt.axis('off')
    plt.savefig(fp)
    plt.close(fig)

def save_entropy_fig(fp, data):
    fig, ax = plt.subplots(1,1, figsize = (8,8))
    plt.imshow(data, vmin=0, vmax=0.3466, cmap='viridis')
    plt.axis('off')
    plt.savefig(fp)
    plt.close(fig)

def save_label_fig(fp, data):
    fig, ax = plt.subplots(1,1, figsize = (8,8))
    colors = ['green', 'red', 'blue', 'gray']
    plt.imshow(data, cmap=matplotlib.colors.ListedColormap(colors))
    plt.axis('off')
    plt.savefig(fp)
    plt.close(fig)
    
def save_opt_fig(fp, data):
    fig, ax = plt.subplots(1,1, figsize = (8,8))
    factor = np.array([4, 5, 6])
    img = np.clip(data[:,:,[2,1,0]]*factor, 0, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(fp)
    plt.close(fig)
    
def save_cloud_fig(fp, data):
    fig, ax = plt.subplots(1,1, figsize = (8,8))
    plt.imshow(data, vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig(fp)
    plt.close(fig)
    
def save_sar_b0_fig(fp, data):
    factor = 2
    img = np.clip(data*factor, 0, 1)
    fig, ax = plt.subplots(1,1, figsize = (8,8))
    plt.imshow(img, vmin=0, vmax=1, cmap = 'gray')
    plt.axis('off')
    plt.savefig(fp)
    plt.close(fig)
    
def save_sar_b1_fig(fp, data):
    factor = 8
    img = np.clip(data*factor, 0, 1)
    fig, ax = plt.subplots(1,1, figsize = (8,8))
    plt.imshow(img, vmin=0, vmax=1, cmap = 'gray')
    plt.axis('off')
    plt.savefig(fp)
    plt.close(fig)
                
def gen_roi_figures(cfg, img_comb_i, img_combination, parent_run_id):
    print(f'Generating ROI Figures {img_comb_i}')
    #base_image = Path(cfg.path.opt) / cfg.site.original_data.opt.train.imgs[0]
    opt_comb, sar_comb = img_combination
    with TemporaryDirectory() as tempdir:
        temp_dir = Path(tempdir)
        true_data = np.squeeze(load_ml_image(cfg.path.label.test), axis=-1)
        
        predict_path = mlflow.artifacts.download_artifacts(run_id=parent_run_id, dst_path = tempdir, artifact_path= f'predictions/{cfg.site.name}-{cfg.exp.name}-{img_comb_i}.tif')
        predict_data = load_ml_image(predict_path)
        
        entropy_path = mlflow.artifacts.download_artifacts(run_id=parent_run_id, dst_path = tempdir, artifact_path= f'entropy/entropy_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}.tif')
        entropy_data = load_ml_image(entropy_path)
        
        for roi_i, roi in enumerate(cfg.site.rois):
            x, y, dx, dy = roi
            x_0 = x - dx
            x_1 = x + dx
            y_0 = y - dy
            y_1 = y + dy
            
            true_roi = true_data[y_0:y_1, x_0: x_1]
            predict_roi = predict_data[y_0:y_1, x_0: x_1]
            entropy_roi = entropy_data[y_0:y_1, x_0: x_1]
            
            true_labels_path = temp_dir / f'true_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}_roi_{roi_i}.png'
            save_label_fig(true_labels_path, true_roi)
            mlflow.log_artifact(true_labels_path, f'rois/{roi_i}/comb_{img_comb_i}', run_id=parent_run_id)
            
            def_class_path = temp_dir / f'def_class_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}_roi_{roi_i}.png'
            save_def_class_fig(def_class_path, predict_roi)
            mlflow.log_artifact(def_class_path, f'rois/{roi_i}/comb_{img_comb_i}', run_id=parent_run_id)

            entropy_path = temp_dir / f'entropy_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}_roi_{roi_i}.png'
            save_entropy_fig(entropy_path, entropy_roi)
            mlflow.log_artifact(entropy_path, f'rois/{roi_i}/comb_{img_comb_i}', run_id=parent_run_id)

        del true_data, predict_data, entropy_data
        if len(opt_comb) > 0:
            opt_i = itemgetter(*list(opt_comb))
            opt_imgs = read_imgs(
                folder=cfg.path.opt, 
                imgs=opt_i(cfg.site.original_data.opt.test.imgs), 
                read_fn=load_opt_image, 
                dtype=np.float16, 
                significance=cfg.general.outlier_significance, 
                factor=1.0,
                flatten=False
                )
            cloud_imgs = read_imgs(
                folder=cfg.path.opt, 
                imgs=opt_i(cfg.site.original_data.opt.test.imgs), 
                read_fn=load_sb_image, 
                dtype=np.float16, 
                significance=cfg.general.outlier_significance, 
                factor=1.0/100,
                prefix_name='cloud_',
                flatten=False
                )
            for roi_i, roi in enumerate(cfg.site.rois):
                x, y, dx, dy = roi
                x_0 = x - dx
                x_1 = x + dx
                y_0 = y - dy
                y_1 = y + dy
                for opt_img_i, opt_img in enumerate(opt_imgs):
                    opt_roi = opt_img[y_0:y_1, x_0: x_1]
                    opt_path = temp_dir / f'opt_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}_roi_{roi_i}_opt_{opt_img_i}.png'
                    save_opt_fig(opt_path, opt_roi)
                    mlflow.log_artifact(opt_path, f'rois/{roi_i}/comb_{img_comb_i}', run_id=parent_run_id)
                    
                for cloud_img_i, cloud_img in enumerate(cloud_imgs):
                    cloud_roi = cloud_img[y_0:y_1, x_0: x_1]
                    cloud_path = temp_dir / f'cloud_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}_roi_{roi_i}_opt_{cloud_img_i}.png'
                    save_cloud_fig(cloud_path, cloud_roi)
                    mlflow.log_artifact(cloud_path, f'rois/{roi_i}/comb_{img_comb_i}', run_id=parent_run_id)
                
            del opt_imgs, cloud_imgs
        
        if len(sar_comb) > 0:
            sar_i = itemgetter(*list(sar_comb))
            sar_imgs = read_imgs(
                folder=cfg.path.sar, 
                imgs=sar_i(cfg.site.original_data.sar.test.imgs), 
                read_fn=load_SAR_image, 
                dtype=np.float16, 
                significance=cfg.general.outlier_significance, 
                factor=1.0,
                flatten=False
                )
            
            for roi_i, roi in enumerate(cfg.site.rois):
                x, y, dx, dy = roi
                x_0 = x - dx
                x_1 = x + dx
                y_0 = y - dy
                y_1 = y + dy
                for sar_img_i, sar_img in enumerate(sar_imgs):
                    sar_roi = sar_img[y_0:y_1, x_0: x_1]
                    sar_path_0 = temp_dir / f'sar_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}_roi_{roi_i}_sar_{sar_img_i}_b_0.png'
                    sar_path_1 = temp_dir / f'sar_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}_roi_{roi_i}_sar_{sar_img_i}_b_1.png'
                    save_sar_b0_fig(sar_path_0, sar_roi[:,:,0])
                    save_sar_b1_fig(sar_path_1, sar_roi[:,:,1])
                    mlflow.log_artifact(sar_path_0, f'rois/{roi_i}/comb_{img_comb_i}', run_id=parent_run_id)
                    mlflow.log_artifact(sar_path_1, f'rois/{roi_i}/comb_{img_comb_i}', run_id=parent_run_id)
                
            del sar_imgs