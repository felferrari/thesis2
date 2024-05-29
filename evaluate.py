import hydra
from src.dataset.data_module import PredDataset
from src.utils.ops import save_geotiff, load_ml_image, load_sb_image, load_opt_image, load_SAR_image
from src.utils.roi import gen_roi_figures
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

eval_version = 2

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
        
    # with Pool(cfg.exp.eval_params.n_rois_processes) as pool:
    #     _ = pool.starmap(gen_roi_figures, args_list)
        
    #metrics_ = np.stack(metrics, axis= 0).sum(axis=0)
    
    results = [m[0] for m in metrics]
    proportions = [m[1] for m in metrics]
    entropy = [m[2] for m in metrics]
    
    results = pd.concat(results)
    proportions = pd.concat(proportions)
    entropy = pd.concat(entropy)
    
    global_results = results.groupby(['cond']).sum().drop(['comb_i'], axis=1)
    global_results = global_results.reset_index()
    global_results.loc[:, 'precision'] = global_results['tps'] / (global_results['tps'] + global_results['fps'])
    global_results.loc[:, 'recall'] = global_results['tps'] / (global_results['tps'] + global_results['fns'])
    global_results.loc[:, 'f1score'] = (2 * global_results.loc[:, 'precision'] * global_results.loc[:, 'recall']) / (global_results.loc[:, 'precision'] + global_results.loc[:, 'recall'])
    global_results = global_results.reindex([2, 0, 1, 4, 3])
    global_results = global_results[['cond', 'f1score', 'precision', 'recall', 'tns', 'tps', 'fns', 'fps']]

    all_results = results.groupby(['cond', 'comb_i']).sum()
    all_results = all_results.reset_index()
    all_results.loc[:, 'precision'] = all_results['tps'] / (all_results['tps'] + all_results['fps'])
    all_results.loc[:, 'recall'] = all_results['tps'] / (all_results['tps'] + all_results['fns'])
    all_results.loc[:, 'f1score'] = (2 * all_results.loc[:, 'precision'] * all_results.loc[:, 'recall']) / (all_results.loc[:, 'precision'] + all_results.loc[:, 'recall'])
    all_results = all_results[['cond', 'comb_i', 'f1score', 'precision', 'recall', 'tns', 'tps', 'fns', 'fps']]
    
    entropy_analysis = entropy.groupby(['percentile']).sum().drop(['comb_i'], axis=1)
    entropy_analysis = entropy_analysis.reset_index()
    entropy_analysis.loc[:, 'precision'] = entropy_analysis['tps'] / (entropy_analysis['tps'] + entropy_analysis['fps'])
    entropy_analysis.loc[:, 'recall'] = entropy_analysis['tps'] / (entropy_analysis['tps'] + entropy_analysis['fns'])
    entropy_analysis.loc[:, 'f1score'] = (2 * entropy_analysis.loc[:, 'precision'] * entropy_analysis.loc[:, 'recall']) / (entropy_analysis.loc[:, 'precision'] + entropy_analysis.loc[:, 'recall'])
    
    entropy_analysis.loc[:, 'precision_low'] = entropy_analysis['tps_low'] / (entropy_analysis['tps_low'] + entropy_analysis['fps_low'])
    entropy_analysis.loc[:, 'recall_low'] = entropy_analysis['tps_low'] / (entropy_analysis['tps_low'] + entropy_analysis['fns_low'])
    entropy_analysis.loc[:, 'f1score_low'] = (2 * entropy_analysis.loc[:, 'precision_low'] * entropy_analysis.loc[:, 'recall_low']) / (entropy_analysis.loc[:, 'precision_low'] + entropy_analysis.loc[:, 'recall_low'])
    
    entropy_analysis.loc[:, 'precision_high'] = entropy_analysis['tps_high'] / (entropy_analysis['tps_high'] + entropy_analysis['fps_high'])
    entropy_analysis.loc[:, 'recall_high'] = entropy_analysis['tps_high'] / (entropy_analysis['tps_high'] + entropy_analysis['fns_high'])
    entropy_analysis.loc[:, 'f1score_high'] = (2 * entropy_analysis.loc[:, 'precision_high'] * entropy_analysis.loc[:, 'recall_high']) / (entropy_analysis.loc[:, 'precision_high'] + entropy_analysis.loc[:, 'recall_high'])
    
    
    with TemporaryDirectory() as tempdir:
        metrics_results_file = Path(tempdir) / f'metrics_results_{cfg.site.name}-{cfg.exp.name}-global.csv'
        global_results.to_csv(metrics_results_file)
        mlflow.log_artifact(metrics_results_file, 'results', run_id=parent_run_id)
        
        all_results_file = Path(tempdir) / f'metrics_results_{cfg.site.name}-{cfg.exp.name}-all.csv'
        all_results.to_csv(all_results_file)
        mlflow.log_artifact(all_results_file, 'results', run_id=parent_run_id)
        
        entropy_props_file = Path(tempdir) / f'metrics_results_{cfg.site.name}-{cfg.exp.name}-entropy-proportions.csv'
        proportions.to_csv(entropy_props_file)
        mlflow.log_artifact(entropy_props_file, 'results', run_id=parent_run_id)
        
        entropy_analysis_file = Path(tempdir) / f'metrics_results_{cfg.site.name}-{cfg.exp.name}-entropy-analysis.csv'
        entropy_analysis.to_csv(entropy_analysis_file)
        mlflow.log_artifact(entropy_analysis_file, 'results', run_id=parent_run_id)
    
    if cfg.clean_predictions:
        with mlflow.start_run(run_id=parent_run_id) as parent_run:
            mlflow.set_tag('eval_version', eval_version)
            for img_comb_i, img_combination in enumerate(imgs_combinations):
                predict_paths = mlflow.artifacts.list_artifacts(run_id=parent_run_id, artifact_path= f'predictions')
                for  pred_path in predict_paths:
                    if pred_path.path.endswith('.tif'):
                        file_path = Path(parent_run.info.artifact_uri[7:]) / pred_path.path
                        file_path.unlink()
                entropy_paths = mlflow.artifacts.list_artifacts(run_id=parent_run_id, artifact_path= f'entropy')
                for entropy_path in entropy_paths:
                    if entropy_path.path.endswith('.tif'):
                        file_path = Path(parent_run.info.artifact_uri[7:]) / entropy_path.path
                        file_path.unlink()
                    
                    
def evaluate_models(cfg, img_comb_i, img_combination, parent_run_id):
    print(f'Evaluating Combination {img_comb_i}')
    base_image = Path(cfg.path.opt) / cfg.site.original_data.opt.train.imgs[0]
    with TemporaryDirectory() as tempdir:
        ref_data = np.squeeze(load_ml_image(cfg.path.label.test), axis=-1)
        ref_labels = np.zeros_like(ref_data)
        ref_labels[ref_data == 0] = 0 #negative
        ref_labels[ref_data == 1] = 1 #positive
        ref_labels[ref_data == 2] = 2 #discard
        ref_labels[ref_data == 3] = 2 # discard
        
        predict_path = mlflow.artifacts.download_artifacts(run_id=parent_run_id, dst_path = tempdir, artifact_path= f'predictions/{cfg.site.name}-{cfg.exp.name}-{img_comb_i}.tif')
        predict_data = load_ml_image(predict_path)
        
        # #calculate entropy
        # pred_prob = predict_data[:,:,1]
        # ep = 1e-7
        # cliped_pred_prob = np.clip(pred_prob.astype(np.float32), ep, 1-ep)
        # entropy = (-1/2) * (cliped_pred_prob * np.log(cliped_pred_prob) + (1-cliped_pred_prob) * np.log(1-cliped_pred_prob))
        # entropy_tif_file = Path(tempdir) / f'entropy_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}.tif'
        # save_geotiff(base_image, entropy_tif_file, entropy, 'float')
        # mlflow.log_artifact(entropy_tif_file, 'entropy', run_id=parent_run_id)
        
        entropy_path = mlflow.artifacts.download_artifacts(run_id=parent_run_id, dst_path = tempdir, artifact_path= f'entropy/entropy_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}.tif')
        entropy = load_sb_image(entropy_path)
        
        #clean predictions based on previous knowledge
        predict_data[ref_data == 3] = np.array([0,0,0,1])
        predict_data[ref_data == 2] = np.array([0,0,1,0])
        predict_data[np.logical_and(ref_data != 2, ref_data != 3)] = np.array([1,1,0,0]) * predict_data[np.logical_and(ref_data != 2, ref_data != 3)]
        predict_labels = np.argmax(predict_data, axis=-1)
        predict_labels[predict_labels==3] = 2
        
        del ref_data, predict_data #, pred_prob, cliped_pred_prob
        
        
        #remove small areas
        positive_data = np.zeros_like(predict_labels)
        positive_data[predict_labels == 1] = 1
        remove_data = positive_data - area_opening(positive_data, cfg.general.area_min)
        predict_labels[remove_data == 1] = 2
        
        del remove_data, positive_data
        
        error_matrix = 4*np.ones_like(predict_labels)
        
        error_matrix[np.logical_and(ref_labels == 0, predict_labels == 0)] = 0 #tn
        error_matrix[np.logical_and(ref_labels == 1, predict_labels == 1)] = 1 #tp
        error_matrix[np.logical_and(ref_labels == 1, predict_labels == 0)] = 2 #fn
        error_matrix[np.logical_and(ref_labels == 0, predict_labels == 1)] = 3 #fp
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
        
        #del entropy
        conds, tns, tps, fns, fps = [], [], [], [], []
        
        #global
        conds.append('global')
        tns.append((error_matrix.flatten() == 0).sum())
        tps.append((error_matrix.flatten() == 1).sum())
        fns.append((error_matrix.flatten() == 2).sum())
        fps.append((error_matrix.flatten() == 3).sum())
        
        #Cloud
        
        #Cloud-free
        error_0 = error_matrix.flatten()[cloudy_pixels.flatten() == 0]
        conds.append('cloud-free')
        tns.append((error_0 == 0).sum())
        tps.append((error_0 == 1).sum())
        fns.append((error_0 == 2).sum())
        fps.append((error_0 == 3).sum())
        
        #Cloudy
        error_1 = error_matrix.flatten()[cloudy_pixels.flatten() == 1]
        conds.append('cloudy')
        tns.append((error_1 == 0).sum())
        tps.append((error_1 == 1).sum())
        fns.append((error_1 == 2).sum())
        fps.append((error_1 == 3).sum())
        
        #Entropy
        
        #Low-entropy
        error_0 = error_matrix.flatten()[entropy_pixels.flatten() == 0]
        conds.append('low-entropy')
        tns.append((error_0 == 0).sum())
        tps.append((error_0 == 1).sum())
        fns.append((error_0 == 2).sum())
        fps.append((error_0 == 3).sum())
        
        #High-entropy
        error_1 = error_matrix.flatten()[entropy_pixels.flatten() == 1]
        conds.append('high-entropy')
        tns.append((error_1 == 0).sum())
        tps.append((error_1 == 1).sum())
        fns.append((error_1 == 2).sum())
        fps.append((error_1 == 3).sum())
        
        #Entropy-cloud proportions
        
        data = {
            'comb_i': img_comb_i,
            'cond': conds,
            'tns': tns,
            'tps': tps,
            'fns': fns,
            'fps': fps,
        }
        
        results = pd.DataFrame(data=data)
        
        conds, pixels = [], []
        
        conds.append('cloud_0_entropy_0')
        pixels.append(np.logical_and(cloudy_pixels == 0, entropy_pixels == 0).sum())
        conds.append('cloud_1_entropy_1')
        pixels.append(np.logical_and(cloudy_pixels == 1, entropy_pixels == 1).sum())
        conds.append('cloud_1_entropy_0')
        pixels.append(np.logical_and(cloudy_pixels == 1, entropy_pixels == 0).sum())
        conds.append('cloud_0_entropy_1')
        pixels.append(np.logical_and(cloudy_pixels == 0, entropy_pixels == 1).sum())
        
        data = {
            'comb_i': img_comb_i,
            'cond': conds,
            'pixels': pixels,
        }
        
        proportions = pd.DataFrame(data=data)
        
        #entropy_percentiles = np.concatenate([np.linspace(0.0, .95, 20), np.linspace(1, 5, 11), np.linspace(5.5, 10, 10)])
        entropy_percentiles = np.concatenate([np.linspace(0.0, .98, 50), np.linspace(1, 5, 41), np.linspace(5.5, 10, 10)])
        percs, tns, tps, fns, fps, entropys = [], [], [], [], [], []
        tns_high, tps_high, fns_high, fps_high = [], [], [], []
        tns_low, tps_low, fns_low, fps_low = [], [], [], []
        for percentile in tqdm(entropy_percentiles):
            min_entropy = np.percentile(entropy.flatten(), (100-percentile))
            entropys.append(min_entropy)
            percs.append(percentile)
            
            audit_pixels = np.zeros_like(predict_labels)
            audit_pixels[entropy > min_entropy] = 1
            
            #audition
            error_p = error_matrix.copy()
            error_p[np.logical_and(audit_pixels == 1, ref_labels == 0)] = 0
            error_p[np.logical_and(audit_pixels == 1, ref_labels == 1)] = 1
            
            tns.append((error_p == 0).sum())
            tps.append((error_p == 1).sum())
            fns.append((error_p == 2).sum())
            fps.append((error_p == 3).sum())
            
            #high entropy
            tns_high.append((error_matrix[entropy > min_entropy] == 0).sum())
            tps_high.append((error_matrix[entropy > min_entropy] == 1).sum())
            fns_high.append((error_matrix[entropy > min_entropy] == 2).sum())
            fps_high.append((error_matrix[entropy > min_entropy] == 3).sum())
            
            #low entropy
            tns_low.append((error_matrix[entropy <= min_entropy] == 0).sum())
            tps_low.append((error_matrix[entropy <= min_entropy] == 1).sum())
            fns_low.append((error_matrix[entropy <= min_entropy] == 2).sum())
            fps_low.append((error_matrix[entropy <= min_entropy] == 3).sum())
            
            
            
        data = {
            'comb_i': img_comb_i,
            'percentile': percs,
            'entropy': entropys,
            'tns': tns,
            'tps': tps,
            'fns': fns,
            'fps': fps,
            'tns_high': tns_high,
            'tps_high': tps_high,
            'fns_high': fns_high,
            'fps_high': fps_high,
            'tns_low': tns_low,
            'tps_low': tps_low,
            'fns_low': fns_low,
            'fps_low': fps_low,
        }
        
        entropy = pd.DataFrame(data=data)
        
        return (results, proportions, entropy)
    
# def save_def_class_fig(fp, data):
#     fig, ax = plt.subplots(1,1, figsize = (8,8))
#     plt.imshow(data[:,:,1], vmin=0, vmax=1, cmap='viridis')
#     plt.axis('off')
#     plt.savefig(fp)
#     plt.close(fig)

# def save_entropy_fig(fp, data):
#     fig, ax = plt.subplots(1,1, figsize = (8,8))
#     plt.imshow(data, vmin=0, vmax=0.3466, cmap='viridis')
#     plt.axis('off')
#     plt.savefig(fp)
#     plt.close(fig)

# def save_label_fig(fp, data):
#     fig, ax = plt.subplots(1,1, figsize = (8,8))
#     colors = ['green', 'red', 'blue', 'gray']
#     plt.imshow(data, cmap=matplotlib.colors.ListedColormap(colors))
#     plt.axis('off')
#     plt.savefig(fp)
#     plt.close(fig)
    
# def save_opt_fig(fp, data):
#     fig, ax = plt.subplots(1,1, figsize = (8,8))
#     factor = np.array([4, 5, 6])
#     img = np.clip(data[:,:,[2,1,0]]*factor, 0, 1)
#     plt.imshow(img)
#     plt.axis('off')
#     plt.savefig(fp)
#     plt.close(fig)
    
# def save_cloud_fig(fp, data):
#     fig, ax = plt.subplots(1,1, figsize = (8,8))
#     plt.imshow(data, vmin=0, vmax=1)
#     plt.axis('off')
#     plt.savefig(fp)
#     plt.close(fig)
    
# def save_sar_b0_fig(fp, data):
#     factor = 2
#     img = np.clip(data*factor, 0, 1)
#     fig, ax = plt.subplots(1,1, figsize = (8,8))
#     plt.imshow(img, vmin=0, vmax=1, cmap = 'gray')
#     plt.axis('off')
#     plt.savefig(fp)
#     plt.close(fig)
    
# def save_sar_b1_fig(fp, data):
#     factor = 8
#     img = np.clip(data*factor, 0, 1)
#     fig, ax = plt.subplots(1,1, figsize = (8,8))
#     plt.imshow(img, vmin=0, vmax=1, cmap = 'gray')
#     plt.axis('off')
#     plt.savefig(fp)
#     plt.close(fig)
                
# def gen_roi_figures(cfg, img_comb_i, img_combination, parent_run_id):
#     print(f'Generating ROI Figures {img_comb_i}')
#     #base_image = Path(cfg.path.opt) / cfg.site.original_data.opt.train.imgs[0]
#     opt_comb, sar_comb = img_combination
#     with TemporaryDirectory() as tempdir:
#         temp_dir = Path(tempdir)
#         true_data = np.squeeze(load_ml_image(cfg.path.label.test), axis=-1)
        
#         predict_path = mlflow.artifacts.download_artifacts(run_id=parent_run_id, dst_path = tempdir, artifact_path= f'predictions/{cfg.site.name}-{cfg.exp.name}-{img_comb_i}.tif')
#         predict_data = load_ml_image(predict_path)
        
#         entropy_path = mlflow.artifacts.download_artifacts(run_id=parent_run_id, dst_path = tempdir, artifact_path= f'entropy/entropy_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}.tif')
#         entropy_data = load_ml_image(entropy_path)
        
#         for roi_i, roi in enumerate(cfg.site.rois):
#             x, y, dx, dy = roi
#             x_0 = x - dx
#             x_1 = x + dx
#             y_0 = y - dy
#             y_1 = y + dy
            
#             true_roi = true_data[y_0:y_1, x_0: x_1]
#             predict_roi = predict_data[y_0:y_1, x_0: x_1]
#             entropy_roi = entropy_data[y_0:y_1, x_0: x_1]
            
#             true_labels_path = temp_dir / f'true_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}_roi_{roi_i}.png'
#             save_label_fig(true_labels_path, true_roi)
#             mlflow.log_artifact(true_labels_path, f'rois/{roi_i}/comb_{img_comb_i}', run_id=parent_run_id)
            
#             def_class_path = temp_dir / f'def_class_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}_roi_{roi_i}.png'
#             save_def_class_fig(def_class_path, predict_roi)
#             mlflow.log_artifact(def_class_path, f'rois/{roi_i}/comb_{img_comb_i}', run_id=parent_run_id)

#             entropy_path = temp_dir / f'entropy_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}_roi_{roi_i}.png'
#             save_entropy_fig(entropy_path, entropy_roi)
#             mlflow.log_artifact(entropy_path, f'rois/{roi_i}/comb_{img_comb_i}', run_id=parent_run_id)

#         del true_data, predict_data, entropy_data
#         if len(opt_comb) > 0:
#             opt_i = itemgetter(*list(opt_comb))
#             opt_imgs = read_imgs(
#                 folder=cfg.path.opt, 
#                 imgs=opt_i(cfg.site.original_data.opt.test.imgs), 
#                 read_fn=load_opt_image, 
#                 dtype=np.float16, 
#                 significance=cfg.general.outlier_significance, 
#                 factor=1.0,
#                 flatten=False
#                 )
#             cloud_imgs = read_imgs(
#                 folder=cfg.path.opt, 
#                 imgs=opt_i(cfg.site.original_data.opt.test.imgs), 
#                 read_fn=load_sb_image, 
#                 dtype=np.float16, 
#                 significance=cfg.general.outlier_significance, 
#                 factor=1.0/100,
#                 prefix_name='cloud_',
#                 flatten=False
#                 )
#             for roi_i, roi in enumerate(cfg.site.rois):
#                 x, y, dx, dy = roi
#                 x_0 = x - dx
#                 x_1 = x + dx
#                 y_0 = y - dy
#                 y_1 = y + dy
#                 for opt_img_i, opt_img in enumerate(opt_imgs):
#                     opt_roi = opt_img[y_0:y_1, x_0: x_1]
#                     opt_path = temp_dir / f'opt_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}_roi_{roi_i}_opt_{opt_img_i}.png'
#                     save_opt_fig(opt_path, opt_roi)
#                     mlflow.log_artifact(opt_path, f'rois/{roi_i}/comb_{img_comb_i}', run_id=parent_run_id)
                    
#                 for cloud_img_i, cloud_img in enumerate(cloud_imgs):
#                     cloud_roi = cloud_img[y_0:y_1, x_0: x_1]
#                     cloud_path = temp_dir / f'cloud_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}_roi_{roi_i}_opt_{cloud_img_i}.png'
#                     save_cloud_fig(cloud_path, cloud_roi)
#                     mlflow.log_artifact(cloud_path, f'rois/{roi_i}/comb_{img_comb_i}', run_id=parent_run_id)
                
#             del opt_imgs, cloud_imgs
        
#         if len(sar_comb) > 0:
#             sar_i = itemgetter(*list(sar_comb))
#             sar_imgs = read_imgs(
#                 folder=cfg.path.sar, 
#                 imgs=sar_i(cfg.site.original_data.sar.test.imgs), 
#                 read_fn=load_SAR_image, 
#                 dtype=np.float16, 
#                 significance=cfg.general.outlier_significance, 
#                 factor=1.0,
#                 flatten=False
#                 )
            
#             for roi_i, roi in enumerate(cfg.site.rois):
#                 x, y, dx, dy = roi
#                 x_0 = x - dx
#                 x_1 = x + dx
#                 y_0 = y - dy
#                 y_1 = y + dy
#                 for sar_img_i, sar_img in enumerate(sar_imgs):
#                     sar_roi = sar_img[y_0:y_1, x_0: x_1]
#                     sar_path_0 = temp_dir / f'sar_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}_roi_{roi_i}_sar_{sar_img_i}_b_0.png'
#                     sar_path_1 = temp_dir / f'sar_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}_roi_{roi_i}_sar_{sar_img_i}_b_1.png'
#                     save_sar_b0_fig(sar_path_0, sar_roi[:,:,0])
#                     save_sar_b1_fig(sar_path_1, sar_roi[:,:,1])
#                     mlflow.log_artifact(sar_path_0, f'rois/{roi_i}/comb_{img_comb_i}', run_id=parent_run_id)
#                     mlflow.log_artifact(sar_path_1, f'rois/{roi_i}/comb_{img_comb_i}', run_id=parent_run_id)
                
#             del sar_imgs
        
if __name__ == "__main__":
    eval()