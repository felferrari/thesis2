import streamlit as st
import matplotlib.pyplot as plt
from src.utils.mlflow import get_exps_metric
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter


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
   fig, ax = plt.subplots(figsize=(8,6))
   sns.lineplot(x='Epoch', y='F1-Score', hue = 'Model', legend = 'full', data = metrics)
   plt.title(f'{metric} Site: {site_name}, Exp: {exp_codes}')
   sns.move_legend(ax, "lower right")
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   #plt.ylim([0,1])
   st.pyplot(fig)
        

for site_code in st.session_state['sites']:
    st.header(f"{st.session_state['sites'][site_code]['name']} ({site_code})")
    plot_training_comp(st.session_state['sites'][site_code]['name'], st.session_state['experiments'], ['exp_101', 'exp_103'], 'train_f1_score_0')
    plot_training_comp(st.session_state['sites'][site_code]['name'], st.session_state['experiments'], ['exp_101', 'exp_103'], 'train_f1_score_1')
    plot_training_comp(st.session_state['sites'][site_code]['name'], st.session_state['experiments'], ['exp_101', 'exp_103'], 'train_f1_score_2')
    plot_training_comp(st.session_state['sites'][site_code]['name'], st.session_state['experiments'], ['exp_151', 'exp_153'], 'train_f1_score_0')
    plot_training_comp(st.session_state['sites'][site_code]['name'], st.session_state['experiments'], ['exp_151', 'exp_153'], 'train_f1_score_1')
    plot_training_comp(st.session_state['sites'][site_code]['name'], st.session_state['experiments'], ['exp_151', 'exp_153'], 'train_f1_score_2')
        