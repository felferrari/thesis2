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
   
   uncertainty_results = uncertainty_results.rename(columns={
      'name': 'Model',
   })
   
   metric_d = {
      'F1-Score': 'f1score',
      'Precision': 'precision',
      'Recall': 'recall',
   }
   
   metric = metric_d[metric_l]
   
   sns.set_theme(font_scale=font_scale)
   fig, ax = plt.subplots(1,1,figsize=(6,5))
   #fig.tight_layout()
   
   sns.lineplot(data = uncertainty_results, x ='percentile', y = metric, hue = 'Model')
   ax.set_xticks(np.arange(0,11))
   sns.move_legend(ax, "lower right") #, bbox_to_anchor=(1, 1))
   ax.axvline(x=3, color = 'k', linestyle = '--')
   ax.set_xlabel('Revised Pixels (%)')
   ax.set_ylim([0,1])
   ax.set_ylabel(metric_l)
   if title:
      ax.title.set_text(f"{metric_l} from Audition - Site {site_code[1]}")
   
   st.pyplot(fig)
   if save:
      plt.savefig(f"figures/uncertainty-curve-{site_code}-{list(set(uncertainty_results['exp_code']))}-{metric}", dpi=300)
   plt.close(fig)
        

sites_names = [''] + [st.session_state['sites'][site]['name'] for site in st.session_state['sites']]

site_name = st.selectbox('Site:', sites_names)



exp_codes = st.multiselect('Experiments:', [''] + list(st.session_state['experiments'].keys()))

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
   
   if len(exp_codes) > 0:
   
      results = get_uncertainty_data(site_name, st.session_state['experiments'], exp_codes)

      plot_uncertainty(site_code, results, metric, title, save, font_scale)
        