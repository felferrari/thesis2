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
from multiprocessing import Process
from operator import itemgetter
import pandas as pd
from multiprocessing import Pool
from matplotlib import pyplot as plt
import matplotlib

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
        
    with Pool(cfg.exp.eval_params.n_metric_processes) as pool:
        metrics = pool.starmap(evaluate_models, args_list)
        
    with Pool(cfg.exp.eval_params.n_rois_processes) as pool:
        _ = pool.starmap(gen_roi_figures, args_list)
    metrics_ = np.stack(metrics, axis= 0).sum(axis=0)
    
    (
        tn_cloud_0,
        tp_cloud_0,
        fn_cloud_0,
        fp_cloud_0,
        tn_cloud_1,
        tp_cloud_1,
        fn_cloud_1,
        fp_cloud_1,
        tn_entropy_0,
        tp_entropy_0,
        fn_entropy_0,
        fp_entropy_0,
        tn_entropy_1,
        tp_entropy_1,
        fn_entropy_1,
        fp_entropy_1,
        cloud_0_entropy_0,
        cloud_1_entropy_1,
        cloud_1_entropy_0,
        cloud_0_entropy_1
    ) = metrics_
    
    tp_global = tp_cloud_0 + tp_cloud_1
    tn_global = tn_cloud_0 + tn_cloud_1
    fn_global = fn_cloud_0 + fn_cloud_1
    fp_global = fp_cloud_0 + fp_cloud_1
    
    precision_global = tp_global / (tp_global + fp_global)
    recall_global = tp_global / (tp_global + fn_global)
    f1_global = 2 * precision_global * recall_global / (precision_global + recall_global)
    
    precision_cloud_0 = tp_cloud_0 / (tp_cloud_0 + fp_cloud_0)
    recall_cloud_0 = tp_cloud_0 / (tp_cloud_0 + fn_cloud_0)
    f1_cloud_0 = 2 * precision_cloud_0 * recall_cloud_0 / (precision_cloud_0 + recall_cloud_0)
    
    precision_cloud_1 = tp_cloud_1 / (tp_cloud_1 + fp_cloud_1)
    recall_cloud_1 = tp_cloud_1 / (tp_cloud_1 + fn_cloud_1)
    f1_cloud_1 = 2 * precision_cloud_1 * recall_cloud_1 / (precision_cloud_1 + recall_cloud_1)
    
    precision_entropy_0 = tp_entropy_0 / (tp_entropy_0 + fp_entropy_0)
    recall_entropy_0 = tp_entropy_0 / (tp_entropy_0 + fn_entropy_0)
    f1_entropy_0 = 2 * precision_entropy_0 * recall_entropy_0 / (precision_entropy_0 + recall_entropy_0)
    
    precision_entropy_1 = tp_entropy_1 / (tp_entropy_1 + fp_entropy_1)
    recall_entropy_1 = tp_entropy_1 / (tp_entropy_1 + fn_entropy_1)
    f1_entropy_1 = 2 * precision_entropy_1 * recall_entropy_1 / (precision_entropy_1 + recall_entropy_1)
    
    entropy_cloud_0 = cloud_0_entropy_1 / (cloud_0_entropy_0 + cloud_0_entropy_1)
    entropy_cloud_1 = cloud_1_entropy_1 / (cloud_1_entropy_0 + cloud_1_entropy_1)
    
    results_data = {
        'metric':[
            'f1score',
            'precision',
            'recall',
            'f1score',
            'precision',
            'recall',
            'f1score',
            'precision',
            'recall',
            'f1score',
            'precision',
            'recall',
            'f1score',
            'precision',
            'recall',
            'high_entropy_prop',
            'high_entropy_prop',
        ],
        'cond':[
            'global',
            'global',
            'global',
            'cloud_0',
            'cloud_0',
            'cloud_0',
            'cloud_1',
            'cloud_1',
            'cloud_1',
            'entropy_0',
            'entropy_0',
            'entropy_0',
            'entropy_1',
            'entropy_1',
            'entropy_1',
            'cloud_0',
            'cloud_1'
        ],
        'value':[
            f1_global,
            precision_global,
            recall_global,
            f1_cloud_0,
            precision_cloud_0,
            recall_cloud_0,
            f1_cloud_1,
            precision_cloud_1,
            recall_cloud_1,
            f1_entropy_0,
            precision_entropy_0,
            recall_entropy_0,
            f1_entropy_1,
            precision_entropy_1,
            recall_entropy_1,
            entropy_cloud_0,
            entropy_cloud_1
        ]
    }
    
    metrics_results = pd.DataFrame(data = results_data, columns=['metric', 'cond', 'value'])
    
    with TemporaryDirectory() as tempdir:
        metrics_results_file = Path(tempdir) / f'metrics_results_{cfg.site.name}-{cfg.exp.name}.csv'
        metrics_results.to_csv(metrics_results_file)
        mlflow.log_artifact(metrics_results_file, 'results', run_id=parent_run_id)
        
    mlflow.log_metric('total_eval_time', (time() - total_t0) / 60., run_id=parent_run_id)


def evaluate_models(cfg, img_comb_i, img_combination, parent_run_id):
    print(f'Evaluating Combination {img_comb_i}')
    base_image = Path(cfg.path.opt) / cfg.site.original_data.opt.train.imgs[0]
    with TemporaryDirectory() as tempdir:
        true_data = np.squeeze(load_ml_image(cfg.path.label.test), axis=-1)
        true_labels = np.zeros_like(true_data)
        true_labels[true_data == 0] = 0 #negative
        true_labels[true_data == 1] = 1 #positive
        true_labels[true_data == 2] = 2 #discard
        true_labels[true_data == 3] = 2 # discard
        
        predict_path = mlflow.artifacts.download_artifacts(run_id=parent_run_id, dst_path = tempdir, artifact_path= f'predictions/{cfg.site.name}-{cfg.exp.name}-{img_comb_i}.tif')
        predict_data = load_ml_image(predict_path)
        
        #calculate entropy
        pred_prob = predict_data[:,:,1]
        ep = 1e-7
        cliped_pred_prob = np.clip(pred_prob.astype(np.float32), ep, 1-ep)
        entropy = (-1/2) * (cliped_pred_prob * np.log(cliped_pred_prob) + (1-cliped_pred_prob) * np.log(1-cliped_pred_prob))
        entropy_tif_file = Path(tempdir) / f'entropy_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}.tif'
        save_geotiff(base_image, entropy_tif_file, entropy, 'float')
        mlflow.log_artifact(entropy_tif_file, 'entropy', run_id=parent_run_id)
        
        #clean predictions based on previous knowledge
        predict_data[true_data == 3] = np.array([0,0,0,1])
        predict_data[true_data == 2] = np.array([0,0,1,0])
        predict_data[np.logical_and(true_data != 2, true_data != 3)] = np.array([1,1,0,0]) * predict_data[np.logical_and(true_data != 2, true_data != 3)]
        predict_labels = np.argmax(predict_data, axis=-1)
        predict_labels[predict_labels==3] = 2
        
        del true_data, predict_data, pred_prob, cliped_pred_prob
        
        
        #remove small areas
        positive_data = np.zeros_like(predict_labels)
        positive_data[predict_labels == 1] = 1
        remove_data = positive_data - area_opening(positive_data, cfg.general.area_min)
        predict_labels[remove_data == 1] = 2
        
        del remove_data, positive_data
        
        error_matrix = np.zeros_like(predict_labels)
        
        error_matrix[np.logical_and(true_labels == 0, predict_labels == 0)] = 0 #tn
        error_matrix[np.logical_and(true_labels == 1, predict_labels == 1)] = 1 #tp
        error_matrix[np.logical_and(true_labels == 1, predict_labels == 0)] = 2 #fn
        error_matrix[np.logical_and(true_labels == 0, predict_labels == 1)] = 3 #fp
        error_tif_file = Path(tempdir) / f'error_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}.tif'
        save_geotiff(base_image, error_tif_file, error_matrix, 'byte')
        mlflow.log_artifact(error_tif_file, 'error', run_id=parent_run_id)
        
        
        #load cloud data
        opt_comb = img_combination[0]
        cloudy_pixels = np.zeros_like(predict_labels)
        if len(opt_comb) > 0:
            opt_i = itemgetter(*list(opt_comb))
            cloud_imgs = read_imgs(
                folder=cfg.path.opt, 
                imgs=opt_i(cfg.site.original_data.opt.train.imgs), 
                read_fn=load_sb_image, 
                dtype=np.float16, 
                significance=cfg.general.outlier_significance, 
                factor=1.0/100,
                prefix_name='cloud_'
                )
            cloud_data = np.stack(cloud_imgs, axis=0).max(0).reshape(cloudy_pixels.shape)
            cloudy_pixels[cloud_data > cfg.exp.eval_params.min_cloudy] = 1
            
            del cloud_data, cloud_imgs
            
        #high entropy data
        entropy_pixels = np.zeros_like(predict_labels)
        entropy_pixels[entropy > cfg.exp.eval_params.min_entropy] = 1
        
        del entropy
        
        #cloud
        error_0 = error_matrix.flatten()[cloudy_pixels.flatten() == 0]
        tn_cloud_0 = (error_0 == 0).sum()
        tp_cloud_0 = (error_0 == 1).sum()
        fn_cloud_0 = (error_0 == 2).sum()
        fp_cloud_0 = (error_0 == 3).sum()
        error_1 = error_matrix.flatten()[cloudy_pixels.flatten() == 1]
        tn_cloud_1 = (error_1 == 0).sum()
        tp_cloud_1 = (error_1 == 1).sum()
        fn_cloud_1 = (error_1 == 2).sum()
        fp_cloud_1 = (error_1 == 3).sum()
        
        #entropy
        error_0 = error_matrix.flatten()[entropy_pixels.flatten() == 0]
        tn_entropy_0 = (error_0 == 0).sum()
        tp_entropy_0 = (error_0 == 1).sum()
        fn_entropy_0 = (error_0 == 2).sum()
        fp_entropy_0 = (error_0 == 3).sum()
        error_1 = error_matrix.flatten()[entropy_pixels.flatten() == 1]
        tn_entropy_1 = (error_1 == 0).sum()
        tp_entropy_1 = (error_1 == 1).sum()
        fn_entropy_1 = (error_1 == 2).sum()
        fp_entropy_1 = (error_1 == 3).sum()
        
        
        cloud_0_entropy_0 = np.logical_and(cloudy_pixels == 0, entropy_pixels == 0).sum()
        cloud_1_entropy_1 = np.logical_and(cloudy_pixels == 1, entropy_pixels == 1).sum()
        cloud_1_entropy_0 = np.logical_and(cloudy_pixels == 1, entropy_pixels == 0).sum()
        cloud_0_entropy_1 = np.logical_and(cloudy_pixels == 0, entropy_pixels == 1).sum()
        
        return (
            tn_cloud_0,
            tp_cloud_0,
            fn_cloud_0,
            fp_cloud_0,
            tn_cloud_1,
            tp_cloud_1,
            fn_cloud_1,
            fp_cloud_1,
            tn_entropy_0,
            tp_entropy_0,
            fn_entropy_0,
            fp_entropy_0,
            tn_entropy_1,
            tp_entropy_1,
            fn_entropy_1,
            fp_entropy_1,
            cloud_0_entropy_0,
            cloud_1_entropy_1,
            cloud_1_entropy_0,
            cloud_0_entropy_1
        )
    
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
        
if __name__ == "__main__":
    eval()