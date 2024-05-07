import streamlit as st
import matplotlib.pyplot as plt
from src.utils.mlflow import get_exps_metric
import seaborn as sns

def plot_training_comp(site_name, experiments, exp_codes, metric):
   metrics = get_exps_metric(site_name, exp_codes, metric, experiments)
   if metrics is None:
      st.write(f'Data incomplete for Site:{site_name}, Exps: [{exp_codes}]')
   
   metrics = metrics.rename(columns={
      'step': 'Epoch',
      'value': 'F1-Score',
      'model': 'run',
      'exp_name': 'Model',
   })
   
   sns.set_theme(style="darkgrid")
   fig, ax = plt.subplots()
   sns.lineplot(x='Epoch', y='F1-Score', hue = 'Model', legend = 'full', data = metrics)
   st.pyplot(fig)
        

for site_code in st.session_state['sites']:
    st.header(f"{st.session_state['sites'][site_code]['name']} ({site_code})")
    plot_training_comp(st.session_state['sites'][site_code]['name'], st.session_state['experiments'], ['exp_101', 'exp_103'], 'train_f1_score_1')
    # plot_training_comp(st.session_state['sites'][site_code], st.session_state['experiments'], ['exp_101', 'exp_103'], 'train_f1_score_0', 'No Deforestation Class F1-Score (Training Dataset)')
    # plot_training_comp(st.session_state['sites'][site_code], st.session_state['experiments'], ['exp_101', 'exp_103'], 'train_f1_score_2', 'Previous Deforestation Class F1-Score (Training Dataset)')
    # plot_training_comp(st.session_state['sites'][site_code], st.session_state['experiments'], ['exp_151', 'exp_153'], 'train_f1_score_1', 'Deforestation Class F1-Score (Training Dataset)')
    # plot_training_comp(st.session_state['sites'][site_code], st.session_state['experiments'], ['exp_151', 'exp_153'], 'train_f1_score_0', 'No Deforestation Class F1-Score (Training Dataset)')
    # plot_training_comp(st.session_state['sites'][site_code], st.session_state['experiments'], ['exp_151', 'exp_153'], 'train_f1_score_2', 'Previous Deforestation Class F1-Score (Training Dataset)')
    # plot_training_comp(st.session_state['sites'][site_code], st.session_state['experiments'], ['exp_151', 'exp_153'], 'train_f1_score_2', 'Previous Deforestation Class F1-Score (Training Dataset)')
        