import pandas as pd
import mlflow

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