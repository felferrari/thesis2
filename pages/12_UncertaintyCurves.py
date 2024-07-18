import streamlit as st
import matplotlib.pyplot as plt
from src.utils.mlflow import get_uncertainty_data
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import numpy as np

#st.set_page_config(layout="wide")

def plot_uncertainty(site_code, results, metric_l, title, save, font_scale):
   uncertainty_results = results.copy()
   
   base_results = uncertainty_results.copy()
   base_results[:] = base_results[base_results['percentile'] == 0]
   base_results['percentile'] = uncertainty_results['percentile']
   
   metric_d = {
      'F1-Score': 'f1score',
      'Precision': 'precision',
      'Recall': 'recall',
   }
   
   metric = metric_d[metric_l]
   
   sns.set_theme(font_scale=font_scale)
   fig, ax = plt.subplots(1,2,figsize=(12,5))
   #fig.tight_layout()
   if title:
      fig.suptitle(f"{uncertainty_results['full_name'][0]} Site {site_code[1]}", y=1)
   
   sns.lineplot(data = uncertainty_results, x ='percentile', y = metric, color = 'blue', label='Pos Audition Result', ax=ax[0])
   sns.lineplot(data = base_results, x ='percentile', y = metric, color='red', linestyle= ':', label = 'Pre Audition Result', ax=ax[0])
   sns.lineplot(data = uncertainty_results, x ='percentile', y = f'{metric}_high', color = 'darkorange', linestyle= '--', label = 'High Uncertainty (Audited Pixels)', ax=ax[0])
   sns.lineplot(data = uncertainty_results, x ='percentile', y = f'{metric}_low', color = 'darkorange', linestyle= '-.', label = 'Low Uncertainty (No Change)', ax=ax[0])
   
   sns.lineplot(data = uncertainty_results, x ='percentile', y = f'entropy', color = 'blue', ax=ax[1])
   ax[0].set_xticks(np.arange(0,11))
   ax[0].axvline(x=3, color = 'k', linestyle = '--')
   ax[0].set_xlabel('Revised Pixels (%)')
   ax[0].set_ylim([0,1])
   ax[0].set_ylabel(metric_l)
   ax[0].title.set_text('Audition Results')
   
   ax[1].set_xticks(np.arange(0,11))
   ax[1].set_xlabel('Revised Pixels (%)')
   ax[1].set_ylabel('Entropy Threshold')
   ax[1].title.set_text('Entropy Thresholds')
   st.pyplot(fig)
   if save:
      plt.savefig(f"figures/uncertainty-curve-s{site_code}-{uncertainty_results['exp_code'][0]}-{metric}", dpi=300)
   plt.close(fig)
        

sites_names = [''] + [st.session_state['sites'][site]['name'] for site in st.session_state['sites']]

site_name = st.selectbox('Site:', sites_names)



exp_code = st.selectbox('Experiments:', [''] + list(st.session_state['experiments'].keys()))

font_scale = st.slider('Font Scale:', 0.5, 5.0, 1.2, step=0.1)

title = st.checkbox('Title', True)

save = st.checkbox('Save', False)

metric = st.selectbox('Metric:', [
   'F1-Score',
   'Precision',
   'Recall',
])


if site_name != '':
   site_code = [site for site in st.session_state['sites'] if st.session_state['sites'][site]['name'] == site_name][0]
   
   if exp_code != '':
   
      results = get_uncertainty_data(site_name, st.session_state['experiments'], [exp_code])

      plot_uncertainty(site_code, results, metric, title, save, font_scale)
        