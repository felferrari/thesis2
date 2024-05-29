import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow import get_uncertainty_data
import seaborn.objects  as so
import numpy as np

st.set_page_config(layout="wide")

def plot_uncertainty_graphic(results, metric, metric_header):
   uncertainty_results = results.copy()
   
   fig, ax = plt.subplots(1,1,figsize=(15,10))
   fig.tight_layout()
   
   sns.lineplot(data = uncertainty_results, x ='percentile', y = metric, hue = 'exp_name')
   ax.set_xticks(np.arange(0,11))
   ax.axvline(x=3, color = 'k', linestyle = '--')
   ax.set_xlabel('Revised Pixels (%)')
   ax.set_ylabel(metric_header)
   st.pyplot(fig)
   plt.close(fig)
   

for site_code in st.session_state['sites']:
   sns.set_theme(style="darkgrid")
   
   site_name = st.session_state['sites'][site_code]['name']
   #results_site = get_site_results(site_name, st.session_state['experiments'])
   #st.table(results_site)
   
   st.header(site_name)
   
   #siamese comparison no cloud
   exp_list = [
      [101, 103],
      [151, 153],
      [101, 103, 311, 312, 313, 411, 412],
      [151, 153, 361, 362, 363, 461, 462],
   ]
   for exps in exp_list:
      exp_codes = [f'exp_{code}' for code in exps]
      
      results = get_uncertainty_data(site_name, st.session_state['experiments'], exp_codes)
      plot_uncertainty_graphic(results, 'f1score', 'F1-Score')
   
   
   