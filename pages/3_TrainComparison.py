import streamlit as st
import matplotlib.pyplot as plt
from src.utils.mlflow import get_exps_metric
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

#st.set_page_config(layout="wide")

def plot_training_comp(site_name, experiments, exp_codes, metric, font_scale, if_title, if_save):
   if site_name == '' or len(exp_codes)==0 or metric == '':
      return
   metrics = get_exps_metric(site_name, exp_codes, metric, experiments)
   if metrics is None:
      st.write(f'Data incomplete for Site:{site_name}, Exps: [{exp_codes}]')
   
   metrics = metrics.rename(columns={
      'step': 'Epoch',
      'value': 'F1-Score',
      'model': 'run',
      'full_name': 'Model',
   })
   
   ms_data = metric.split('_')
   ds_name = {
      'val': 'Validation',
      'train': 'Training'
   }
   
   cl_name = {
      '0': 'No Deforestation',
      '1': 'Deforestation',
      '2': 'Previous Deforestation'
   }
   
   site_code = {
      'Apui': 1,
      'Para': 2
   }

   
   title = f'F1-Score for Class {cl_name[ms_data[-1]]} - {ds_name[ms_data[0]]} Subdataset (Site {site_code[site_name]})' if if_title else ''
   
   sns.set_theme(font_scale=font_scale)
   fig, ax = plt.subplots(figsize=(8,6))
   sns.lineplot(x='Epoch', y='F1-Score', hue = 'Model', legend = 'full', data = metrics, errorbar='sd')
   plt.title(title, pad=15)
   sns.move_legend(ax, "lower right")
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.ylim([0,1])
   st.pyplot(fig)
   if if_save:
      plt.savefig(f'figures/train-{ds_name[ms_data[0]]}-class{ms_data[-1]}-s{site_code[site_name]}-{exp_codes}', dpi=300, bbox_inches='tight')
   plt.close(fig)
        

sites_names = [''] + [st.session_state['sites'][site]['name'] for site in st.session_state['sites']]

site = st.selectbox('Site:', sites_names)

options = st.multiselect('Experiments:', list(st.session_state['experiments'].keys()))

font_scale = st.slider('Font Scale:', 0.5, 5.0, 1.0, step=0.1)

title = st.checkbox('Title', True)

save = st.checkbox('Save', False)

metric = st.selectbox('Metric:', [
   '',
   'train_f1_score_0',
   'train_f1_score_1',
   'train_f1_score_2',
   'val_f1_score_0',
   'val_f1_score_1',
   'val_f1_score_2',
])

plot_training_comp(site, st.session_state['experiments'], options, metric, font_scale, title, save)
   # plot_training_comp(st.session_state['sites'][site_code]['name'], st.session_state['experiments'], ['exp_101', 'exp_103'], 'train_f1_score_1')
   # plot_training_comp(st.session_state['sites'][site_code]['name'], st.session_state['experiments'], ['exp_101', 'exp_103'], 'train_f1_score_2')
   # plot_training_comp(st.session_state['sites'][site_code]['name'], st.session_state['experiments'], ['exp_151', 'exp_153'], 'train_f1_score_0')
   # plot_training_comp(st.session_state['sites'][site_code]['name'], st.session_state['experiments'], ['exp_151', 'exp_153'], 'train_f1_score_1')
   # plot_training_comp(st.session_state['sites'][site_code]['name'], st.session_state['experiments'], ['exp_151', 'exp_153'], 'train_f1_score_2')
        