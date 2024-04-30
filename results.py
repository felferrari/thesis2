import streamlit as st
from pathlib import Path
from hydra import compose, initialize
import mlflow
#import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

conf_path = Path('conf')
exps_path = conf_path / 'exp'
sites_path = conf_path / 'site'

experiments = {}
for exp_file in  exps_path.glob('exp_*.*'):
    exp_code = exp_file.stem
    with initialize(version_base=None, config_path="conf", job_name="test_app"):
        cfg = compose(config_name="config", overrides=[f"+exp={exp_code}"])
        experiments[exp_code] = dict(cfg.exp)
        
sites = {}
for site_file in  sites_path.glob('s*.*'):
    site_code = site_file.stem
    with initialize(version_base=None, config_path="conf", job_name="test_app"):
        cfg = compose(config_name="config", overrides=[f"+site={site_code}"])
        sites[site_code] = dict(cfg.site)

def plot_training(site, experiment):
    mlflow_client = mlflow.client.MlflowClient()
    mlflow_experiment_l = mlflow_client.search_experiments(filter_string=f"name='{site['name']}'")
    if len(mlflow_experiment_l) == 0:
        st.write(f'No Data Experiment')
        return
    mlflow_experiment = mlflow_experiment_l[0]
    mlflow_parent_run_l = mlflow_client.search_runs(experiment_ids=[mlflow_experiment.experiment_id], filter_string=f"run_name='{experiment['name']}'")
    if len(mlflow_parent_run_l) == 0:
        st.write(f'No Run Experiment')
        return
    mlflow_parent_run = mlflow_parent_run_l[0]
    st.write(f'Experiment: {mlflow_parent_run.info.run_name}')
    mlflow_runs = mlflow_client.search_runs(experiment_ids=[mlflow_experiment.experiment_id], filter_string=f"params.parent_run_id = '{mlflow_parent_run.info.run_id}'")
    data = []
    for model_run in mlflow_runs:
        historic = mlflow_client.get_metric_history(model_run.info.run_id, 'val_f1_score_1')
        for hist_i in historic:
            data.append([hist_i.step, hist_i.value, model_run.info.run_name])
    data = pd.DataFrame(data, columns = ['step', 'value', 'model'])
    fig = px.line(data, x ='step', y='value', color='model')
    st.plotly_chart(fig, theme=None, use_container_width=True)

for site_code in sites:
    st.header(sites[site_code]['name'])
    for exp_code in experiments:
        plot_training(sites[site_code], experiments[exp_code])
        