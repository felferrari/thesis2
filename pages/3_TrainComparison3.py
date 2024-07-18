import streamlit as st
import matplotlib.pyplot as plt
from src.utils.mlflow import get_exps_metric
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import pandas as pd

#st.set_page_config(layout="wide")

def plot_training_comp(site_name, experiments, exp_codes, metric_l, title, save, font_scale):
   if site_name == '' or len(exp_codes)==0 or len(metric_l) == 0:
      return
   
   met_dict = {
      'train_f1_score_0': 'No Deforestation [Training]',
      'train_f1_score_1': 'Deforestation [Training]',
      'train_f1_score_2': 'Previous Deforestation [Training]',
      'val_f1_score_0': 'No Deforestation [Validation]',
      'val_f1_score_1': 'Deforestation [Validation]',
      'val_f1_score_2': 'Previous Deforestation [Validation]',
   }
   
   metrics = None
   for metric_i in metric_l:
      if metrics is None:
         metrics = get_exps_metric(site_name, exp_codes, metric_i, experiments)
         metrics['metric'] = met_dict[metric_i]
      else:
         mt = get_exps_metric(site_name, exp_codes, metric_i, experiments)
         mt['metric'] =  met_dict[metric_i]
         metrics = pd.concat([metrics,  mt])
         
   if metrics is None:
      st.write(f'Data incomplete for Site:{site_name}, Exps: [{exp_codes}]')
   
   metrics = metrics.rename(columns={
      'step': 'Epoch',
      'value': 'F1-Score',
      'model': 'run',
      'exp_full_name': 'Model',
      'metric': 'Class [Sub-Dataset]',
   })
   
   site_code = {
      'Apui': 1,
      'Para': 2
   }
   
   title = f"Site {site_code[site_name]}" if title else ''
   
   sns.set_theme(font_scale=font_scale)
   fig, ax = plt.subplots(figsize=(8,6))
   sns.lineplot(x='Epoch', y='F1-Score', hue = 'Class [Sub-Dataset]', legend = 'full', data = metrics, errorbar='sd')
   plt.title(title)
   sns.move_legend(ax, "lower right")
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.ylim([0,1])
   st.pyplot(fig)
   if save:
      plt.savefig(f'figures/train-s{site_code[site_name]}-{exp_codes}-{metric_l}', dpi=300)
   plt.close(fig)
        

sites_names = [''] + [st.session_state['sites'][site]['name'] for site in st.session_state['sites']]

site = st.selectbox('Site:', sites_names)

options = st.multiselect('Experiments:', [''] + list(st.session_state['experiments'].keys()))

font_scale = st.slider('Font Scale:', 0.5, 5.0, 1.0, step=0.1)

title = st.checkbox('Title', True)

save = st.checkbox('Save', False)

metric = st.multiselect('Metric:', [
   'train_f1_score_0',
   'train_f1_score_1',
   'train_f1_score_2',
   'val_f1_score_0',
   'val_f1_score_1',
   'val_f1_score_2',
])

plot_training_comp(site, st.session_state['experiments'], options, metric, title, save, font_scale)
   # plot_training_comp(st.session_state['sites'][site_code]['name'], st.session_state['experiments'], ['exp_101', 'exp_103'], 'train_f1_score_1')
   # plot_training_comp(st.session_state['sites'][site_code]['name'], st.session_state['experiments'], ['exp_101', 'exp_103'], 'train_f1_score_2')
   # plot_training_comp(st.session_state['sites'][site_code]['name'], st.session_state['experiments'], ['exp_151', 'exp_153'], 'train_f1_score_0')
   # plot_training_comp(st.session_state['sites'][site_code]['name'], st.session_state['experiments'], ['exp_151', 'exp_153'], 'train_f1_score_1')
   # plot_training_comp(st.session_state['sites'][site_code]['name'], st.session_state['experiments'], ['exp_151', 'exp_153'], 'train_f1_score_2')
        