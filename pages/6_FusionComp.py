import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow import get_site_results
import seaborn.objects  as so
import numpy as np

st.set_page_config(layout="wide")

def plot_simple_graphic(results, metric, metric_header):
   
   global_results = results[results['cond'] == 'global']
   
   global_results_resunet = global_results[global_results['base_architecture'] == 'resunet']
   global_results_swin = global_results[global_results['base_architecture'] == 'transformer']
   
   global_results_resunet.loc[:, 'max'] = global_results_resunet[metric].max()
   global_results_swin.loc[:, 'max'] = global_results_swin[metric].max()
   
   sns.set_theme(font_scale=1.5)
   sns.set_palette('tab10')
   
   fig, ax = plt.subplots(1,2,figsize=(15,5))
   fig.tight_layout()
   alpha = 0.4
   
   ax[0].axvspan(-0.5, 3.5, facecolor='g', alpha=alpha) #, label='Cloud-Free')
   ax[0].axvspan(3.5, 4.5, facecolor='r', alpha=alpha) #, label='No Optical')
   ax[0].axvspan(4.5, 8.5, facecolor='b', alpha=alpha) #, label='Diverse Cloud')
   ax[0].axvspan(8.5, 10.5, facecolor='y', alpha=alpha) #, label='Diverse Cloud (Pre-trained)')
   
   ax[1].axvspan(-0.5, 3.5, facecolor='g', alpha=alpha, label='Cloud-Free')
   ax[1].axvspan(3.5, 4.5, facecolor='r', alpha=alpha, label='No Optical')
   ax[1].axvspan(4.5, 8.5, facecolor='b', alpha=alpha, label='Diverse Cloud')
   ax[1].axvspan(8.5, 10.5, facecolor='y', alpha=alpha, label='Diverse Cloud (Pre-trained)')
   
   sns.barplot(data = global_results_resunet, x ='exp_name', y = metric, hue = 'name', ax = ax[0], edgecolor='k', errorbar = None, legend=False)
   sns.barplot(data = global_results_swin, x ='exp_name', y = metric, hue = 'name', ax = ax[1], edgecolor='k', errorbar = None, legend=True)
   sns.lineplot(data = global_results_resunet, x ='exp_name', y = 'max', color= 'k', ax= ax[0], label = 'Best Result', legend=False)
   sns.lineplot(data = global_results_swin, x ='exp_name', y = 'max', color= 'k', ax = ax[1], label = 'Best Result', legend=True)
   
   sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   ax[0].set_ylim([0,1])
   ax[1].set_ylim([0,1])
   
   ax[0].set_xticklabels([])
   ax[1].set_xticklabels([])
   ax[1].set_yticklabels([])
   plt.xticks(ha='left', rotation = 325)
   ax[0].set_xlabel('Model')
   ax[1].set_xlabel('Model')
   ax[0].set_ylabel(metric_header)
   ax[1].set_ylabel(None)
   ax[0].set_title(f'Base Architecture: ResUnet')
   ax[1].set_title(f'Base Architecture: Swin')
   st.pyplot(fig)
   plt.close(fig)
   
def plot_cloudcond_graphic(results, metric, metric_header):
   
   cloud_cond_results = results.copy()
   cloud_cond_results = cloud_cond_results.reset_index(drop=True)
   cloud_cond_results = cloud_cond_results.drop(cloud_cond_results[cloud_cond_results['cond'].str.contains('entropy')].index)
   
   cloud_cond_results = cloud_cond_results.reset_index(drop=True)
   cloud_cond_results = cloud_cond_results.drop(cloud_cond_results[
      (cloud_cond_results['cond'].str.contains('cloud-free')) &
      (cloud_cond_results['exp_name'].str.contains('no_cloud')) 
      ].index)
   
   cloud_cond_results = cloud_cond_results.reset_index(drop=True)
   cloud_cond_results = cloud_cond_results.drop(cloud_cond_results[
      (cloud_cond_results['cond'].str.contains('cloud-free')) &
      (cloud_cond_results['exp_name'].str.contains('_sar_'))
      ].index)
   
   cloud_cond_results['exp_name'] = cloud_cond_results['exp_name'].str.replace('resunet_', '')
   cloud_cond_results['exp_name'] = cloud_cond_results['exp_name'].str.replace('swin_', '')
   cloud_cond_results['exp_name'] = cloud_cond_results['exp_name'].str.replace('_combined', '')
   cloud_cond_results['exp_name'] = cloud_cond_results['exp_name'].str.replace('_concat', '')
   cloud_cond_results['exp_name'] = cloud_cond_results['exp_name'].str.replace('_pretrained', '_pt')
   
   cloud_cond_results['cond'] = cloud_cond_results['cond'].str.replace('global', 'All Pixels')
   cloud_cond_results['cond'] = cloud_cond_results['cond'].str.replace('cloudy', 'Cloudy Pixels')
   cloud_cond_results['cond'] = cloud_cond_results['cond'].str.replace('cloud-free', 'Cloud-Free Pixels')
   
   global_results_resunet = cloud_cond_results[cloud_cond_results['base_architecture'] == 'resunet']
   global_results_swin = cloud_cond_results[cloud_cond_results['base_architecture'] == 'transformer']
   
   global_results_resunet.loc[:, 'max'] = global_results_resunet[metric].max()
   resunet_max = global_results_resunet[metric].max()
   global_results_swin.loc[:, 'max'] = global_results_swin[metric].max()
   swin_max = global_results_swin[metric].max()
   
   global_results_resunet = global_results_resunet.reset_index(drop=True)
   global_results_swin = global_results_swin.reset_index(drop=True)
   
   sns.set_theme(font_scale=1.5)
   sns.set_palette('tab10')
   
   fig, ax = plt.subplots(1,2,figsize=(15,5))
   fig.tight_layout()
   alpha = 0.4
   
   ax[0].axvspan(-0.5, 3.5, facecolor='g', alpha=alpha) #, label='Cloud-Free')
   ax[0].axvspan(3.5, 4.5, facecolor='r', alpha=alpha) #, label='No Optical')
   ax[0].axvspan(4.5, 8.5, facecolor='b', alpha=alpha) #, label='Diverse Cloud')
   ax[0].axvspan(8.5, 10.5, facecolor='y', alpha=alpha) #, label='Diverse Cloud (Pre-trained)')
   
   ax[1].axvspan(-0.5, 3.5, facecolor='g', alpha=alpha, label='Cloud-Free')
   ax[1].axvspan(3.5, 4.5, facecolor='r', alpha=alpha, label='No Optical')
   ax[1].axvspan(4.5, 8.5, facecolor='b', alpha=alpha, label='Diverse Cloud')
   ax[1].axvspan(8.5, 10.5, facecolor='y', alpha=alpha, label='Diverse Cloud (Pre-trained)')

   sns.barplot(data = global_results_resunet, x ='exp_name', y = metric, hue = 'cond', ax = ax[0], edgecolor='k', errorbar = None, legend=False)
   sns.barplot(data = global_results_swin, x ='exp_name', y = metric, hue = 'cond', ax = ax[1], edgecolor='k', errorbar = None, legend=True)
   sns.lineplot(data = global_results_resunet, x ='exp_name', y = 'max', color= 'k', ax= ax[0], label = 'Best Result', legend=False)
   sns.lineplot(data = global_results_swin, x ='exp_name', y = 'max', color= 'k', ax = ax[1], label = 'Best Result', legend=True)
   
   ax[0].axhline(y=resunet_max, color = 'k')
   ax[1].axhline(y=swin_max, color = 'k')
   
   sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   ax[0].set_ylim([0,1])
   ax[1].set_ylim([0,1])
   ax[1].set_yticklabels([])
   plt.sca(ax[0])
   plt.xticks(ha='left', rotation = 325)
   plt.sca(ax[1])
   plt.xticks(ha='left', rotation = 325)
   ax[0].set_xlabel('Model')
   ax[1].set_xlabel('Model')
   ax[0].set_ylabel(metric_header)
   ax[1].set_ylabel(None)
   ax[0].set_title(f'Base Architecture: ResUnet')
   ax[1].set_title(f'Base Architecture: Swin')
   st.pyplot(fig)
   plt.close(fig)

for site_code in st.session_state['sites']:
   sns.set_theme(style="darkgrid")
   
   site_name = st.session_state['sites'][site_code]['name']
   #results_site = get_site_results(site_name, st.session_state['experiments'])
   #st.table(results_site)
   
   st.header(site_name)
   
   #siamese comparison no cloud
   exps = [101, 301, 302, 303, 103, 102, 311, 312, 313, 411, 412, 151, 351, 352, 353, 153, 152, 361, 362, 363, 461, 462]
   exp_codes = [f'exp_{code}' for code in exps]
   
   results = get_site_results(site_name, st.session_state['experiments'], exp_codes)
   
   plot_simple_graphic(results, 'f1score', 'F1-Score')
   plot_simple_graphic(results, 'precision', 'Precision')
   plot_simple_graphic(results, 'recall', 'Recall')
   
   plot_cloudcond_graphic(results, 'f1score', 'F1-Score')
   plot_cloudcond_graphic(results, 'precision', 'Precision')
   plot_cloudcond_graphic(results, 'recall', 'Recall')
   
   