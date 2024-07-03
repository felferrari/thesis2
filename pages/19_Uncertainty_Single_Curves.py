import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow import get_uncertainty_data
import seaborn.objects  as so
import numpy as np

st.set_page_config(layout="wide")

def plot_uncertainty_graphic(results, metric, metric_header):
   uncertainty_results = results.copy()
   
   base_results = uncertainty_results.copy()
   base_results[:] = base_results[base_results['percentile'] == 0]
   base_results['percentile'] = uncertainty_results['percentile']
   

   
   # fig, ax = plt.subplots(1,1,figsize=(9,6))
   # fig.tight_layout()
   
   # sns.lineplot(data = uncertainty_results, x ='percentile', y = metric, color = 'blue', label='Pos Audition Result')
   # sns.lineplot(data = base_results, x ='percentile', y = metric, color='red', linestyle= ':', label = 'Pre Audition Result')
   # sns.lineplot(data = uncertainty_results, x ='percentile', y = f'{metric}_high', color = 'darkorange', linestyle= '--', label = 'High Uncertainty (Audited Pixels)')
   # sns.lineplot(data = uncertainty_results, x ='percentile', y = f'{metric}_low', color = 'darkorange', linestyle= '-.', label = 'Low Uncertainty (No Change)')
   # ax.set_xticks(np.arange(0,11))
   # ax.axvline(x=3, color = 'k', linestyle = '--')
   # ax.set_xlabel('Revised Pixels (%)')
   # ax.set_ylabel(metric_header)
   # st.pyplot(fig)
   # plt.close(fig)
   
   fig, ax = plt.subplots(1,2,figsize=(16,5))
   #fig.tight_layout()
   
   fig.suptitle(uncertainty_results['full_name'][0])
   
   sns.lineplot(data = uncertainty_results, x ='percentile', y = metric, color = 'blue', label='Pos Audition Result', ax=ax[0])
   sns.lineplot(data = base_results, x ='percentile', y = metric, color='red', linestyle= ':', label = 'Pre Audition Result', ax=ax[0])
   sns.lineplot(data = uncertainty_results, x ='percentile', y = f'{metric}_high', color = 'darkorange', linestyle= '--', label = 'High Uncertainty (Audited Pixels)', ax=ax[0])
   sns.lineplot(data = uncertainty_results, x ='percentile', y = f'{metric}_low', color = 'darkorange', linestyle= '-.', label = 'Low Uncertainty (No Change)', ax=ax[0])
   
   sns.lineplot(data = uncertainty_results, x ='percentile', y = f'entropy', color = 'blue', ax=ax[1])
   ax[0].set_xticks(np.arange(0,11))
   ax[0].axvline(x=3, color = 'k', linestyle = '--')
   ax[0].set_xlabel('Revised Pixels (%)')
   ax[0].set_ylim([0,1])
   ax[0].set_ylabel(metric_header)
   ax[0].title.set_text('Audition Results')
   
   ax[1].set_xticks(np.arange(0,11))
   ax[1].set_xlabel('Revised Pixels (%)')
   ax[1].set_ylabel('Entropy')
   ax[1].title.set_text('Entropy Thresholds')
   st.pyplot(fig)
   plt.close(fig)

for site_code in st.session_state['sites']:
   sns.set_theme(style="darkgrid")
   
   site_name = st.session_state['sites'][site_code]['name']
   #results_site = get_site_results(site_name, st.session_state['experiments'])
   #st.table(results_site)
   
   st.header(site_name)
   
   #siamese comparison no cloud
   for exp_code in st.session_state['experiments'] :
      
      results = get_uncertainty_data(site_name, st.session_state['experiments'], [exp_code])
      plot_uncertainty_graphic(results, 'f1score', 'F1-Score')
   
   
   