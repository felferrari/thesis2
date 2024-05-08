import pandas as pd
import mlflow
from tempfile import TemporaryDirectory
import streamlit as st

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
    for exp_code in exp_codes:
        exp_name = experiments[exp_code]['name'] 
        if metrics is None:
            metrics = get_exp_metric(site_name, exp_name, metric, include_warmup)
            metrics['exp_name'] = exp_name
            metrics['site_name'] = site_name
            if metrics is None:
                return
        else:
            metrics_i = get_exp_metric(site_name, exp_name, metric, include_warmup)
            metrics_i['exp_name'] = exp_name
            metrics_i['site_name'] = site_name
            metrics =  pd.concat([metrics, metrics_i])  
    return metrics

def get_site_results(site_name, experiments):
    mlflow_client = mlflow.client.MlflowClient()
    
    mlflow_experiment_l = mlflow_client.search_experiments(filter_string=f"name='{site_name}'")
    if len(mlflow_experiment_l) == 0:
        return
    mlflow_experiment = mlflow_experiment_l[0]
    
    results = []
    my_bar = st.progress(0, text='Loading Data')
    with TemporaryDirectory() as temp_dir:
        n_exps = len(experiments)
        for i, exp_code in enumerate(experiments):
            my_bar.progress((i/n_exps))
            exp_name = experiments[exp_code]['name']
            mlflow_parent_run_l = mlflow_client.search_runs(experiment_ids=[mlflow_experiment.experiment_id], filter_string=f"run_name='{exp_name}'")
            if len(mlflow_parent_run_l) == 0:
                break
            mlflow_parent_run = mlflow_parent_run_l[0]
            result_file_name = f'metrics_results_{site_name}-{exp_name}-global.csv'
            result_file = mlflow_client.download_artifacts(mlflow_parent_run.info.run_id, f'results/{result_file_name}', temp_dir)
            results_df = pd.read_csv(result_file)
            results_df['site'] = site_name
            results_df['exp_name'] = exp_name
            results_df['exp_code'] = exp_code
            results_df['base_architecture'] = experiments[exp_code]['base_architecture']
            results_df['opt_condition'] = experiments[exp_code]['opt_condition']
            results_df['sar_condition'] = experiments[exp_code]['sar_condition']
            if 'no_prevmap' in exp_name:
                results_df['prev_map'] = False
            else:
                results_df['prev_map'] = True
                
            if 'opt' in exp_name:
                results_df['type'] = 'Optical'
            elif 'sar' in exp_name:
                results_df['type'] = 'SAR'
            elif 'pixel_level' in exp_name:
                results_df['type'] = 'Pixel Level Fusion'
            elif 'feature_middle' in exp_name:
                results_df['type'] = 'Feature Level (Middle) Fusion'
            elif 'feature_late' in exp_name:
                results_df['type'] = 'Feature Level (Late) Fusion'
            else:
                results_df['type'] = ''
                
            if 'siamese' in exp_name:
                results_df['layout'] = 'siamese'
            elif 'sar' in exp_name:
                results_df['type'] = 'SAR'
            
            
            results.append(results_df)
    results = pd.concat(results)
    my_bar.empty()
    return results
            
        
        