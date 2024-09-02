import pandas as pd
import mlflow
from tempfile import TemporaryDirectory
import streamlit as st
from PIL import Image
from pathlib import Path

def get_exp_metric(site_name, exp_name, metric, include_warmup = True):
    mlflow_client = mlflow.client.MlflowClient()
    
    mlflow_experiment_l = mlflow_client.search_experiments(filter_string=f"name='{site_name}'")
    if len(mlflow_experiment_l) == 0:
        return
    mlflow_experiment = mlflow_experiment_l[0]
    
    mlflow_parent_run_l = mlflow_client.search_runs(experiment_ids=[mlflow_experiment.experiment_id], filter_string=f"run_name='{exp_name}'")
    if len(mlflow_parent_run_l) == 0:
        return
    mlflow_parent_run = mlflow_parent_run_l[0]
    
    mlflow_runs = mlflow_client.search_runs(experiment_ids=[mlflow_experiment.experiment_id], filter_string=f"params.parent_run_id = '{mlflow_parent_run.info.run_id}'")
    if len(mlflow_runs) != 5:
        return
    
    
    data_return = []
    for run in mlflow_runs:
        data = []
        historic = mlflow_client.get_metric_history(run.info.run_id, f'{metric}')
        for hist_i in historic:
            data.append([hist_i.step, hist_i.value, run.info.run_name])
        data_df = pd.DataFrame(data, columns = ['step', 'value', 'model'])
        if include_warmup:
            if f'warmup_{metric}' in run.data.metrics.keys():
                historic = mlflow_client.get_metric_history(run.info.run_id, f'warmup_{metric}')
                warmup_data = []
                for hist_i in historic:
                    warmup_data.append([hist_i.step, hist_i.value, run.info.run_name])
                warmup_data_df = pd.DataFrame(warmup_data, columns = ['step', 'value', 'model'])
                
                data_df['step'] = len(warmup_data_df) + data_df['step']
                
                data_df = pd.concat([warmup_data_df, data_df])
        data_return.append(data_df)
        
    data_return = pd.concat(data_return)        
    return data_return

def get_exps_metric(site_name, exp_codes, metric, experiments, include_warmup = True):
    metrics = None
    my_bar = st.progress(0, text='Loading Data')
    n_exps = len(exp_codes)
    for i, exp_code in enumerate(exp_codes):
        my_bar.progress((i/n_exps))
        exp_name = experiments[exp_code]['name'] 
        exp_full_name = experiments[exp_code]['full_name'] 
        if metrics is None:
            metrics = get_exp_metric(site_name, exp_name, metric, include_warmup)
            metrics = include_names(metrics, experiments, site_name, exp_name, exp_code)
            # metrics['exp_name'] = exp_name
            # metrics['exp_full_name'] = exp_full_name
            # metrics['site_name'] = site_name
            if metrics is None:
                return
        else:
            metrics_i = get_exp_metric(site_name, exp_name, metric, include_warmup)
            metrics_i = include_names(metrics_i, experiments, site_name, exp_name, exp_code)
            # metrics_i['exp_name'] = exp_name
            # metrics_i['exp_full_name'] = exp_full_name
            # metrics_i['site_name'] = site_name
            metrics =  pd.concat([metrics, metrics_i])  
    my_bar.empty()
    return metrics

def get_site_results(site_name, experiments, exp_codes = None):
    mlflow_client = mlflow.client.MlflowClient()
    
    mlflow_experiment_l = mlflow_client.search_experiments(filter_string=f"name='{site_name}'")
    if len(mlflow_experiment_l) == 0:
        return
    mlflow_experiment = mlflow_experiment_l[0]
    
    results = []
    my_bar = st.progress(0, text='Loading Data')
    with TemporaryDirectory() as temp_dir:
        if exp_codes is None:
            exp_codes = list(experiments.keys)
        n_exps = len(exp_codes)
        for i, exp_code in enumerate(exp_codes):
            my_bar.progress((i/n_exps))
            exp_name = experiments[exp_code]['name']
            mlflow_parent_run_l = mlflow_client.search_runs(experiment_ids=[mlflow_experiment.experiment_id], filter_string=f"run_name='{exp_name}'")
            if len(mlflow_parent_run_l) == 0:
                break
            mlflow_parent_run = mlflow_parent_run_l[0]
            result_file_name = f'metrics_results_{site_name}-{exp_name}-global.csv'
            result_file = mlflow_client.download_artifacts(mlflow_parent_run.info.run_id, f'results/{result_file_name}', temp_dir)
            results_df = pd.read_csv(result_file)
            results_df = include_names(results_df, experiments, site_name, exp_name, exp_code)
            
            results.append(results_df)
    results = pd.concat(results)
    results = results.reset_index(drop=True)
    
    my_bar.empty()
    return results

def get_site_time_results(site_name, experiments, exp_codes = None):
    mlflow_client = mlflow.client.MlflowClient()
    
    mlflow_experiment_l = mlflow_client.search_experiments(filter_string=f"name='{site_name}'")
    if len(mlflow_experiment_l) == 0:
        return
    mlflow_experiment = mlflow_experiment_l[0]
    
    results = []
    my_bar = st.progress(0, text='Loading Data')
    with TemporaryDirectory() as temp_dir:
        if exp_codes is None:
            exp_codes = list(experiments.keys)
        n_exps = len(exp_codes)
        for i, exp_code in enumerate(exp_codes):
            my_bar.progress((i/n_exps))
            
            
            exp_name = experiments[exp_code]['name']
            mlflow_parent_run_l = mlflow_client.search_runs(experiment_ids=[mlflow_experiment.experiment_id], filter_string=f"run_name='{exp_name}'")
            if len(mlflow_parent_run_l) == 0:
                break
            mlflow_parent_run = mlflow_parent_run_l[0]
            exp_full_name = experiments[exp_code]['full_name'] 
            
            historic = mlflow_client.get_metric_history(mlflow_parent_run.info.run_id, f'eval_10epochs')
            eval_time_data = []
            for hist_i in historic:
                eval_time_data.append(['Prediction',hist_i.step, hist_i.value])
            eval_time_df = pd.DataFrame(eval_time_data, columns=['stage', 'step', 'value'])
            eval_time_df = include_names(eval_time_df, experiments, site_name, exp_name, exp_code)

            historic = mlflow_client.get_metric_history(mlflow_parent_run.info.run_id, f'train_10epochs')
            train_time_data = []
            for hist_i in historic:
                train_time_data.append(['Training', hist_i.step, hist_i.value])
            train_time_data = pd.DataFrame(train_time_data, columns=['stage', 'step', 'value'])
            train_time_data = include_names(train_time_data, experiments, site_name, exp_name, exp_code)
            
            
            results.append(pd.concat([train_time_data, eval_time_df]))
            
    results = pd.concat(results)
    results = results.reset_index(drop=True)
    
    my_bar.empty()
    return results


def get_site_pred_time(site_name, experiments, exp_codes = None):
    mlflow_client = mlflow.client.MlflowClient()
    
    mlflow_experiment_l = mlflow_client.search_experiments(filter_string=f"name='{site_name}'")
    if len(mlflow_experiment_l) == 0:
        return
    mlflow_experiment = mlflow_experiment_l[0]
    
    results = []
    my_bar = st.progress(0, text='Loading Data')
    with TemporaryDirectory() as temp_dir:
        if exp_codes is None:
            exp_codes = list(experiments.keys)
        n_exps = len(exp_codes)
        for i, exp_code in enumerate(exp_codes):
            my_bar.progress((i/n_exps))
            
            
            exp_name = experiments[exp_code]['name']
            mlflow_parent_run_l = mlflow_client.search_runs(experiment_ids=[mlflow_experiment.experiment_id], filter_string=f"run_name='{exp_name}'")
            if len(mlflow_parent_run_l) == 0:
                break
            mlflow_parent_run = mlflow_parent_run_l[0]
            exp_full_name = experiments[exp_code]['full_name'] 
            
            historic = mlflow_client.get_metric_history(mlflow_parent_run.info.run_id, f'comb_pred_time_0')
            eval_time_data = []
            for hist_i in historic[-1:]:
                eval_time_data.append(['Prediction',hist_i.step, hist_i.value])
            eval_time_df = pd.DataFrame(eval_time_data, columns=['stage', 'step', 'value'])
            eval_time_df = include_names(eval_time_df, experiments, site_name, exp_name, exp_code)

            results.append(eval_time_df)
            
    results = pd.concat(results)
    results = results.reset_index(drop=True)
    
    my_bar.empty()
    return results

def get_site_size_results(site_name, experiments, exp_codes = None):
    mlflow_client = mlflow.client.MlflowClient()
    
    mlflow_experiment_l = mlflow_client.search_experiments(filter_string=f"name='{site_name}'")
    if len(mlflow_experiment_l) == 0:
        return
    mlflow_experiment = mlflow_experiment_l[0]
    
    results = []
    my_bar = st.progress(0, text='Loading Data')
    with TemporaryDirectory() as temp_dir:
        if exp_codes is None:
            exp_codes = list(experiments.keys)
        n_exps = len(exp_codes)
        for i, exp_code in enumerate(exp_codes):
            my_bar.progress((i/n_exps))
            
            
            exp_name = experiments[exp_code]['name']
            mlflow_parent_run_l = mlflow_client.search_runs(experiment_ids=[mlflow_experiment.experiment_id], filter_string=f"run_name='{exp_name}'")
            if len(mlflow_parent_run_l) == 0:
                break
            mlflow_parent_run = mlflow_parent_run_l[0]
            
            historic = mlflow_client.get_metric_history(mlflow_parent_run.info.run_id, f'n_params')
            size_data = []
            for hist_i in historic:
                size_data.append(['Trainable Paramters',hist_i.step, hist_i.value])
            size_df = pd.DataFrame(size_data, columns=['Data', 'step', 'value'])
            size_df = include_names(size_df, experiments, site_name, exp_name, exp_code)
            
            results.append(size_df)
            
    results = pd.concat(results)
    results = results.reset_index(drop=True)
    
    my_bar.empty()
    return results



def get_uncertainty_data(site_name, experiments, exp_codes = None):
    mlflow_client = mlflow.client.MlflowClient()
    
    mlflow_experiment_l = mlflow_client.search_experiments(filter_string=f"name='{site_name}'")
    if len(mlflow_experiment_l) == 0:
        return
    mlflow_experiment = mlflow_experiment_l[0]
    
    results = []
    my_bar = st.progress(0, text='Loading Data')
    with TemporaryDirectory() as temp_dir:
        if exp_codes is None:
            exp_codes = list(experiments.keys)
        n_exps = len(exp_codes)
        for i, exp_code in enumerate(exp_codes):
            my_bar.progress((i/n_exps))
            exp_name = experiments[exp_code]['name']
            mlflow_parent_run_l = mlflow_client.search_runs(experiment_ids=[mlflow_experiment.experiment_id], filter_string=f"run_name='{exp_name}'")
            if len(mlflow_parent_run_l) == 0:
                break
            mlflow_parent_run = mlflow_parent_run_l[0]
            result_file_name = f'metrics_results_{site_name}-{exp_name}-entropy-analysis.csv'
            result_file = mlflow_client.download_artifacts(mlflow_parent_run.info.run_id, f'results/{result_file_name}', temp_dir)
            results_df = pd.read_csv(result_file)
            results_df = include_names(results_df, experiments, site_name, exp_name, exp_code)
            # results_df['site'] = site_
            
            results.append(results_df)
    results = pd.concat(results)
    results = results.reset_index(drop=True)
    
    my_bar.empty()
    return results

def get_uncertainty_proportions_data(site_name, experiments, exp_codes = None):
    mlflow_client = mlflow.client.MlflowClient()
    
    mlflow_experiment_l = mlflow_client.search_experiments(filter_string=f"name='{site_name}'")
    if len(mlflow_experiment_l) == 0:
        return
    mlflow_experiment = mlflow_experiment_l[0]
    
    results = []
    my_bar = st.progress(0, text='Loading Data')
    with TemporaryDirectory() as temp_dir:
        if exp_codes is None:
            exp_codes = list(experiments.keys)
        n_exps = len(exp_codes)
        for i, exp_code in enumerate(exp_codes):
            my_bar.progress((i/n_exps))
            exp_name = experiments[exp_code]['name']
            mlflow_parent_run_l = mlflow_client.search_runs(experiment_ids=[mlflow_experiment.experiment_id], filter_string=f"run_name='{exp_name}'")
            if len(mlflow_parent_run_l) == 0:
                break
            mlflow_parent_run = mlflow_parent_run_l[0]
            result_file_name = f'metrics_results_{site_name}-{exp_name}-entropy-proportions.csv'
            result_file = mlflow_client.download_artifacts(mlflow_parent_run.info.run_id, f'results/{result_file_name}', temp_dir)
            results_df = pd.read_csv(result_file)
            results_df = include_names(results_df, experiments, site_name, exp_name, exp_code)
            # results_df['site'] = site_name
            
            
            results.append(results_df)
    results = pd.concat(results)
    results = results.reset_index(drop=True)
    
    my_bar.empty()
    return results

def include_names(results_df, experiments, site_name, exp_name, exp_code):
    results_df['site'] = site_name
    results_df['exp_name'] = exp_name
    results_df['exp_code'] = exp_code
    results_df['base_architecture'] = experiments[exp_code]['base_architecture']
    results_df['opt_condition'] = experiments[exp_code]['opt_condition']
    results_df['sar_condition'] = experiments[exp_code]['sar_condition']
    results_df['full_name'] = experiments[exp_code]['full_name']
    if 'no_prevmap' in exp_name:
        results_df['prev_map'] = False
    else:
        results_df['prev_map'] = True
        
    if 'opt' in exp_name:
        results_df['name'] = 'Optical'
    elif 'sar' in exp_name:
        results_df['name'] = 'SAR'
    elif 'pixel_level' in exp_name:
        results_df['name'] = 'Pixel'
    elif 'feature_middle' in exp_name:
        results_df['name'] = 'Feat-Mid'
    elif 'feature_late' in exp_name:
        results_df['name'] = 'Feat-Late'
    elif 'cross_fusion' in exp_name:
        results_df['name'] = 'Cross-Fusion'
    else:
        results_df['name'] = ''
        
    if 'siamese' in exp_name:
        results_df['siamese'] = True
    else:
        results_df['siamese'] = False
        
    if 'pretrained' in exp_name:
        results_df['pretrained'] = True
    else:
        results_df['pretrained'] = False
        
    return results_df

def load_model(site_name, exp_code, model_i, device):
    mlflow_client = mlflow.MlflowClient()
    
    experiment = mlflow_client.search_experiments(filter_string=f"name = '{site_name}'")[0]
    
    parent_run = mlflow_client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.exp_code = '{exp_code}'"
    )[0]
    
    run =  mlflow_client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"parameters.parent_run_id = '{parent_run.info.run_id}' AND attributes.run_name = 'model_{model_i}'"
    )[0]
    
    model_id = f'runs:/{run.info.run_id}/model'
    return mlflow.pytorch.load_model(model_id, map_location = device)
    
            
        
def update_pretrained_weights(cfg, model_module, model_i):
    if cfg.exp.train_params.pretrain_encoder is not None:
        site_name = cfg.site.name
        exp_opt_code, exp_sar_code = cfg.exp.train_params.pretrain_encoder
        
        opt_model_module = load_model(site_name, exp_opt_code, model_i, model_module.device)
        sar_model_module = load_model(site_name, exp_sar_code, model_i, model_module.device)
        
        model_module.model.encoder_opt.load_state_dict(opt_model_module.model.encoder.state_dict())
        model_module.model.encoder_sar.load_state_dict(sar_model_module.model.encoder.state_dict())
            
    if cfg.exp.train_params.pretrain_encoder_decoder is not None:
        site_name = cfg.site.name
        exp_opt_code, exp_sar_code = cfg.exp.train_params.pretrain_encoder_decoder
        
        opt_model_module = load_model(site_name, exp_opt_code, model_i, model_module.device)
        sar_model_module = load_model(site_name, exp_sar_code, model_i, model_module.device)
        
        model_module.model.encoder_opt.load_state_dict(opt_model_module.model.encoder.state_dict())
        model_module.model.bn_opt.load_state_dict(opt_model_module.model.bn.state_dict())
        model_module.model.decoder_opt.load_state_dict(opt_model_module.model.decoder.state_dict())
        
        model_module.model.encoder_sar.load_state_dict(sar_model_module.model.encoder.state_dict())
        model_module.model.bn_sar.load_state_dict(sar_model_module.model.bn.state_dict())
        model_module.model.decoder_sar.load_state_dict(sar_model_module.model.decoder.state_dict())
    
    return model_module

def get_rois_images(site, experiments, exp_codes, roi_codes = None):
    mlflow_client = mlflow.client.MlflowClient()
    
    mlflow_experiment_l = mlflow_client.search_experiments(filter_string=f"name='{site['name']}'")
    if len(mlflow_experiment_l) == 0:
        return
    mlflow_experiment = mlflow_experiment_l[0]
    
    results = {}
    my_bar = st.progress(0, text='Loading Data')
    for exp_i, exp_code in enumerate(exp_codes):
        my_bar.progress((exp_i/len(exp_codes)))
        results[exp_code] = {}
    
        exp_name = experiments[exp_code]['name']
        mlflow_parent_run_l = mlflow_client.search_runs(experiment_ids=[mlflow_experiment.experiment_id], filter_string=f"run_name='{exp_name}'")
        if len(mlflow_parent_run_l) == 0:
            my_bar.empty()
            st.text(f'Erro exp {exp_code}')
            return
        mlflow_parent_run = mlflow_parent_run_l[0]
        
        with TemporaryDirectory() as temp_dir:
            
            for roi_i in range(len(site['rois'])):
                if (roi_codes is not None) and  not (f'roi_{roi_i}' in roi_codes):
                    continue
                results[exp_code][f'roi_{roi_i}'] = {}
                
                comb_list = mlflow.artifacts.list_artifacts(run_id = mlflow_parent_run.info.run_id, artifact_path = f'rois/{roi_i}')
                
                for comb in comb_list:
                    
                    comb_i = int(comb.path.split('_')[1])
                    results[exp_code][f'roi_{roi_i}'][f'comb_{comb_i}'] = {}
                    # images = mlflow.artifacts.list_artifacts(run_id = mlflow_parent_run.info.run_id, artifact_path = f'{comb.path}')
                    images = mlflow.artifacts.download_artifacts(run_id = mlflow_parent_run.info.run_id, artifact_path = f'{comb.path}', dst_path = temp_dir)
                    images = list(Path(images).glob('*.*'))
                    images.sort()
                    
                    for image in images:
                        results[exp_code][f'roi_{roi_i}'][f'comb_{comb_i}'][image.name.split('.')[0]] = Image.open(image)
                
    my_bar.empty()
    return results
                