import hydra
from src.dataset.data_module import PredDataset
from src.utils.ops import save_geotiff, load_ml_image, load_sb_image
from tempfile import TemporaryDirectory
import mlflow
from time import time
import numpy as np
from pathlib import Path
from skimage.morphology import area_opening
import torch
from multiprocessing import Process
from src.utils.generate import read_imgs
from operator import itemgetter
import pandas as pd
from multiprocessing import Pool

@hydra.main(version_base=None, config_path='conf', config_name='config.yaml')
def predict(cfg):
    torch.set_float32_matmul_precision('high')
    mlflow.set_experiment(experiment_name = cfg.site.name)
    
    runs = mlflow.search_runs(
        filter_string = f'run_name = "{cfg.exp.name}"'
        )
    parent_run_id = runs['run_id'][0]

    with mlflow.start_run(run_id=parent_run_id) as parent_run:
        total_t0 = time()
    
        imgs_combinations = PredDataset.test_combinations(cfg)
        args_list = []
        for img_comb_i, img_combination in enumerate(imgs_combinations):
            args_list.append((cfg, img_comb_i, img_combination, parent_run.info.run_id))
            
        with Pool(cfg.exp.eval_params.n_processes) as pool:
            metrics = pool.starmap(evaluate_models, args_list)
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
                'high_entropy_prop',
                'high_entropy_prop',
            ],
            'cond':[
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
            mlflow.log_artifact(metrics_results_file, 'results')
            
        '''cloud_results = zip(
            ['f1score', 'precision', 'recall', 'f1score', 'precision', 'recall'],
            ['cloud_0', 'cloud_0', 'cloud_0', 'cloud_1', 'cloud_1', 'cloud_1'],
            cloud_metrics,
            )
        
        metrics_results = pd.DataFrame(data = cloud_results, columns=['metric', 'cond', 'value'])
        
        entropy_results = zip(
            ['f1score', 'precision', 'recall', 'f1score', 'precision', 'recall'],
            ['entropy_0', 'entropy_0', 'entropy_0', 'entropy_1', 'entropy_1', 'entropy_1'],
            entropy_metrics,
            )
        
        metrics_results = pd.concat([metrics_results, pd.DataFrame(data = entropy_results, columns=['metric', 'cond', 'value'])])
        metrics_results_file = Path(tempdir) / f'metrics_results_{cfg.site.name}-{cfg.exp.name}-{img_comb_i}.csv'
        metrics_results.to_csv()
        mlflow.log_artifact()
        
        cloud_entropy_correlation = zip(
            ['cloud_0_entropy_0','cloud_1_entropy_1','cloud_0_entropy_1','cloud_1_entropy_0'],
            [np.logical_and(cloudy_pixels == 0, entropy_pixels == 0).sum(),
            np.logical_and(cloudy_pixels == 1, entropy_pixels == 1).sum(),
            np.logical_and(cloudy_pixels == 0, entropy_pixels == 1).sum(),
            np.logical_and(cloudy_pixels == 1, entropy_pixels == 0).sum()]
            )
        
        cloud_entropy_correlation_results = pd.DataFrame(data = cloud_entropy_correlation, columns=['cond', 'count'])'''
            
        mlflow.log_metric('total_eval_time', (time() - total_t0) / 60.)


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
        mlflow.log_artifact(entropy_tif_file, 'entropy')
        
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
        mlflow.log_artifact(error_tif_file, 'error')
        
        
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
    
        
                
# def eval_metrics(error_matrix, mask_matrix):
#     #mask == 0
#     error_0 = error_matrix.flatten()[mask_matrix.flatten() == 0]
#     if len(error_0) == 0:
#         f1score_0, precision_0, recall_0 = np.nan, np.nan, np.nan
#     else:
#         recall_1 = (error_1 == 1).sum() / ((error_1 == 1).sum() + (error_1 == 2).sum())
        
#         f1score_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1)
        
#     return f1score_0, precision_0, recall_0, f1score_1, precision_1, recall_1
        
        
        
        
if __name__ == "__main__":
    predict()