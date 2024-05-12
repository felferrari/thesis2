import streamlit as st
import matplotlib.pyplot as plt
from src.utils.mlflow import get_exp_metric
import seaborn as sns

st.set_page_config(layout="wide")

def plot_training(site_name, exp_name, metric):
    metric = get_exp_metric(site_name, exp_name, metric)
    if metric is None:
        st.write(f'Data incomplete for Site:{site_name}, Exps: [{exp_name}]')
        return
    
    metric = metric.rename(columns={
        'step': 'Epoch',
        'value': 'F1-Score',
        'model': 'run'
    })
   
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots()
    sns.lineplot(x='Epoch', y='F1-Score', hue = 'run', legend = 'full', data = metric)
    plt.title(f'Site: {site_name} Model:{exp_name} ({exp_code})')
    st.pyplot(fig)
    plt.close(fig)

for site_code in st.session_state['sites'] :
    st.header(f"{st.session_state['sites'][site_code]['name']} ({site_code})")
    for exp_code in st.session_state['experiments'] :
        plot_training(st.session_state['sites'][site_code]['name'], st.session_state['experiments'][exp_code]['name'], 'train_f1_score_1')
        