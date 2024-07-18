import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow import get_site_results
import numpy as np

#st.set_page_config(layout="wide")

def plot_simple_graphic(results, metric, metric_header, site_code):
   
   global_results = results[results['cond'] == 'global']
   
   global_results_resunet = global_results[global_results['base_architecture'] == 'resunet']
   global_results_swin = global_results[global_results['base_architecture'] == 'transformer']
   
   global_results_resunet.loc[:, 'max'] = global_results_resunet[metric].max()
   resunet_max = global_results_resunet[metric].max()
   global_results_swin.loc[:, 'max'] = global_results_swin[metric].max()
   swin_max = global_results_swin[metric].max()
   
   sns.set_theme(font_scale=2)
   sns.set_palette('tab10')
   
   fig, ax = plt.subplots(1,1,figsize=(13,7))
   fig.tight_layout()
   alpha = 0.4
   
   fig.suptitle(f'{metric_header} - Base Architecture: ResUnet (Site {site_code[1]})', y=1.1)
   
   plt.plot([], [], ' ', label="--Dataset--")
   ax.axvspan(-0.5, 4.5, facecolor='g', alpha=alpha, label='Cloud-Free')
   ax.axvspan(4.5, 5.5, facecolor='r', alpha=alpha, label='No Optical')
   ax.axvspan(5.5, 10.5, facecolor='b', alpha=alpha, label='Diverse')
   ax.axvspan(10.5, 13.5, facecolor='y', alpha=alpha, label='Diverse [Pre-trained]')
   plt.plot([], [], ' ', label="--Model--")
   
   sns.barplot(data = global_results_resunet, x ='exp_name', y = metric, hue = 'name', ax = ax, edgecolor='k', errorbar = None, legend=True, estimator='sum')
   sns.lineplot(data = global_results_resunet, x ='exp_name', y = 'max', color= 'b', ax= ax, label = 'Best Result', legend=True, linestyle = 'dotted', linewidth = 2)
   
   ax.axhline(y=resunet_max, color = 'b', linestyle = 'dotted', linewidth = 2)
   
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax.get_legend().get_texts(), fontsize=14) 
   ax.set_ylim([0.3, 1.05])
   
   ax.set_xticklabels([])
   plt.xticks(ha='left', rotation = 325)
   ax.set_xlabel('Model')
   ax.set_ylabel(metric_header)
   
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=18, fmt='%.2f', fontweight='bold', padding = 2, color = 'k')
   st.pyplot(fig)
   plt.savefig(f'figures/fusion-resunet-{site_code}-{metric}.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   
   sns.set_theme(font_scale=2)
   sns.set_palette('tab10')
   
   fig, ax = plt.subplots(1,1,figsize=(13,7))
   fig.tight_layout()
   alpha = 0.4
   
   fig.suptitle(f'{metric_header} - Base Architecture: Swin (Site {site_code[1]})', y=1.1)
   
   plt.plot([], [], ' ', label="--Dataset--")
   ax.axvspan(-0.5, 4.5, facecolor='g', alpha=alpha, label='Cloud-Free')
   ax.axvspan(4.5, 5.5, facecolor='r', alpha=alpha, label='No Optical')
   ax.axvspan(5.5, 10.5, facecolor='b', alpha=alpha, label='Diverse')
   ax.axvspan(10.5, 13.5, facecolor='y', alpha=alpha, label='Diverse [Pre-trained]')
   plt.plot([], [], ' ', label="--Model--")
   
   sns.barplot(data = global_results_swin, x ='exp_name', y = metric, hue = 'name', ax = ax, edgecolor='k', errorbar = None, legend=True, estimator='sum')
   sns.lineplot(data = global_results_swin, x ='exp_name', y = 'max', color= 'b', ax= ax, label = 'Best Result', legend=True, linestyle = 'dotted', linewidth = 2)
   
   ax.axhline(y=swin_max, color = 'b', linestyle = 'dotted', linewidth = 2)
   
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax.get_legend().get_texts(), fontsize=13) 
   ax.set_ylim([0.3, 1.05])
   
   ax.set_xticklabels([])
   plt.xticks(ha='left', rotation = 325)
   ax.set_xlabel('Model')
   ax.set_ylabel(metric_header)
   
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=18, fmt='%.2f', fontweight='bold', padding = 2, color = 'k')
   st.pyplot(fig)
   plt.savefig(f'figures/fusion-swin-{site_code}-{metric}.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   
def plot_cloudcond_graphic(results, metric, metric_header, site_code):
   
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
   
   cloud_cond_results['full_name'] = cloud_cond_results['full_name'].str.replace('ResUnet ', '')
   cloud_cond_results['full_name'] = cloud_cond_results['full_name'].str.replace('SWIN ', '')
   cloud_cond_results['full_name'] = cloud_cond_results['full_name'].str.replace('[CLOUD-FREE]', ' ')
   cloud_cond_results['full_name'] = cloud_cond_results['full_name'].str.replace('[CLOUD-DIVERSE]', '  ')
   cloud_cond_results['full_name'] = cloud_cond_results['full_name'].str.replace('[PRE-TRAINED]', '   ')
   cloud_cond_results['full_name'] = cloud_cond_results['full_name'].str.replace('[AVERAGE-12]', '    ')
   
   
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
   
   
   sns.set_theme(font_scale=2)
   sns.set_palette('tab10')
   
   fig, ax = plt.subplots(1,1,figsize=(13,6))
   #fig.tight_layout()
   alpha = 0.4
   
   fig.suptitle(f'{metric_header} Base Architecture: ResUnet (Site {site_code[1]})', y= 1.1)
   
   plt.plot([], [], ' ', label="--Dataset--")
   ax.axvspan(-0.5, 4.5, facecolor='g', alpha=alpha, label='Cloud-Free')
   ax.axvspan(4.5, 5.5, facecolor='r', alpha=alpha, label='No Optical')
   ax.axvspan(5.5, 10.5, facecolor='b', alpha=alpha, label='Diverse')
   ax.axvspan(10.5, 13.5, facecolor='y', alpha=alpha, label='Diverse [Pre-trained]')
   plt.plot([], [], ' ', label="--Cloud Condition--")

   sns.barplot(data = global_results_resunet, x ='full_name', y = metric, hue = 'cond', edgecolor='k', errorbar = None, legend=True, width = 0.9, estimator='sum')
   sns.lineplot(data = global_results_resunet, x ='full_name', y = 'max', color= 'b', label = 'Best Result', legend=True, linestyle = 'dotted', linewidth = 3)
   
   ax.axhline(y=resunet_max, color = 'b', linestyle = 'dotted', linewidth = 3)
   
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax.get_legend().get_texts(), fontsize=13) 
   plt.xticks(ha='left', rotation = 325)
   ax.set_ylim([0.3, 1.1])
   ax.set_xlabel('Model')
   ax.set_ylabel(metric_header)
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=15, fmt='%.2f', fontweight='bold', padding = 5, color = 'k', rotation=90)
   st.pyplot(fig)
   plt.savefig(f'figures/fusion-clouds-resunet-{site_code}-{metric}.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   
   
   sns.set_theme(font_scale=2)
   sns.set_palette('tab10')
   
   fig, ax = plt.subplots(1,1,figsize=(13,6))
   #fig.tight_layout()
   alpha = 0.4
   
   fig.suptitle(f'{metric_header} Base Architecture: Swin (Site {site_code[1]})', y= 1.1)
   
   plt.plot([], [], ' ', label="--Dataset--")
   ax.axvspan(-0.5, 4.5, facecolor='g', alpha=alpha, label='Cloud-Free')
   ax.axvspan(4.5, 5.5, facecolor='r', alpha=alpha, label='No Optical')
   ax.axvspan(5.5, 10.5, facecolor='b', alpha=alpha, label='Diverse')
   ax.axvspan(10.5, 13.5, facecolor='y', alpha=alpha, label='Diverse [Pre-trained]')
   plt.plot([], [], ' ', label="--Cloud Condition--")

   sns.barplot(data = global_results_swin, x ='full_name', y = metric, hue = 'cond', edgecolor='k', errorbar = None, legend=True, width = 0.9, estimator='sum')
   sns.lineplot(data = global_results_swin, x ='full_name', y = 'max', color= 'b', label = 'Best Result', legend=True, linestyle = 'dotted', linewidth = 3)
   
   ax.axhline(y=swin_max, color = 'b', linestyle = 'dotted', linewidth = 3)
   
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax.get_legend().get_texts(), fontsize=15) 
   plt.xticks(ha='left', rotation = 325)
   ax.set_ylim([0.3, 1.1])
   ax.set_xlabel('Model')
   ax.set_ylabel(metric_header)
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=15, fmt='%.2f', fontweight='bold', padding = 5, color = 'k', rotation=90)
   st.pyplot(fig)
   plt.savefig(f'figures/fusion-clouds-swin-{site_code}-{metric}.png', dpi=300, bbox_inches='tight')
   plt.close(fig)

for site_code in st.session_state['sites']:
   sns.set_theme(style="darkgrid")
   
   site_name = st.session_state['sites'][site_code]['name']
   #results_site = get_site_results(site_name, st.session_state['experiments'])
   #st.table(results_site)
   
   st.header(site_name)
   
   #siamese comparison no cloud
   exps = [101, 301, 302, 303, 306, 103, 102, 311, 312, 313, 316, 411, 412, 413, 151, 351, 352, 353, 356, 153, 152, 361, 362, 363, 366, 461, 462, 463]
   exp_codes = [f'exp_{code}' for code in exps]
   
   results = get_site_results(site_name, st.session_state['experiments'], exp_codes)
   
   plot_simple_graphic(results, 'f1score', 'F1-Score', site_code)
   plot_simple_graphic(results, 'precision', 'Precision', site_code)
   plot_simple_graphic(results, 'recall', 'Recall', site_code)
   
   plot_cloudcond_graphic(results, 'f1score', 'F1-Score', site_code)
   plot_cloudcond_graphic(results, 'precision', 'Precision', site_code)
   plot_cloudcond_graphic(results, 'recall', 'Recall', site_code)
   
   