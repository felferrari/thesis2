import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow import get_site_results
import seaborn.objects  as so
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
   
   sns.set_theme(font_scale=1.5)
   sns.set_palette('tab10')
   
   fig, ax = plt.subplots(1,2,figsize=(15,5))
   fig.tight_layout()
   #fig.tight_layout(pad=1.4, w_pad=0.5, h_pad=1.0)
   alpha = 0.4
   
   fig.suptitle(f'{metric_header} for Site {site_code[1]}', y=1.1)
   
   ax[0].axvspan(-0.5, 4.5, facecolor='g', alpha=alpha) #, label='Cloud-Free')
   ax[0].axvspan(4.5, 5.5, facecolor='r', alpha=alpha) #, label='No Optical')
   ax[0].axvspan(5.5, 10.5, facecolor='b', alpha=alpha) #, label='Diverse Cloud')
   ax[0].axvspan(10.5, 13.5, facecolor='y', alpha=alpha) #, label='Diverse Cloud (Pre-trained)')
   
   ax[1].axvspan(-0.5, 4.5, facecolor='g', alpha=alpha, label='Cloud-Free')
   ax[1].axvspan(4.5, 5.5, facecolor='r', alpha=alpha, label='No Optical')
   ax[1].axvspan(5.5, 10.5, facecolor='b', alpha=alpha, label='Diverse Cloud')
   ax[1].axvspan(10.5, 13.5, facecolor='y', alpha=alpha, label='Diverse Cloud (Pre-trained)')
   
   sns.barplot(data = global_results_resunet, x ='exp_name', y = metric, hue = 'name', ax = ax[0], edgecolor='k', errorbar = None, legend=False, estimator='sum')
   sns.barplot(data = global_results_swin, x ='exp_name', y = metric, hue = 'name', ax = ax[1], edgecolor='k', errorbar = None, legend=True, estimator='sum')
   sns.lineplot(data = global_results_resunet, x ='exp_name', y = 'max', color= 'b', ax= ax[0], label = 'Best Result', legend=False, linestyle = 'dotted')
   sns.lineplot(data = global_results_swin, x ='exp_name', y = 'max', color= 'b', ax = ax[1], label = 'Best Result', legend=True, linestyle = 'dotted')
   
   ax[0].axhline(y=resunet_max, color = 'b', linestyle = 'dotted')
   ax[1].axhline(y=swin_max, color = 'b', linestyle = 'dotted')
   
   sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   ax[0].set_ylim([0.3, 1.05])
   ax[1].set_ylim([0.3, 1.05])
   
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
   
   for bars in ax[0].containers:
      ax[0].bar_label(bars, fontsize=11, fmt='%.2f', fontweight='bold', padding = 2, color = 'k')
   for bars in ax[1].containers:
      ax[1].bar_label(bars, fontsize=11, fmt='%.2f', fontweight='bold', padding = 2, color = 'k')
   st.pyplot(fig)
   plt.savefig(f'figures/fusion-{site_code}-{metric}.png', dpi=300, bbox_inches='tight')
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
   
   sns.set_theme(font_scale=1.4)
   sns.set_palette('tab10')
   
   fig, ax = plt.subplots(1,2,figsize=(16,14))
   #fig.tight_layout()
   alpha = 0.4
   
   fig.suptitle(f'{metric_header} for Site {site_code[1]}', y=0.95)
   
   ax[0].axhspan(-0.5, 4.5, facecolor='g', alpha=alpha) #, label='Cloud-Free')
   ax[0].axhspan(4.5, 5.5, facecolor='r', alpha=alpha) #, label='No Optical')
   ax[0].axhspan(5.5, 10.5, facecolor='b', alpha=alpha) #, label='Diverse Cloud')
   ax[0].axhspan(10.5, 13.5, facecolor='y', alpha=alpha) #, label='Diverse Cloud (Pre-trained)')
   
   ax[1].axhspan(-0.5, 4.5, facecolor='g', alpha=alpha, label='Cloud-Free')
   ax[1].axhspan(4.5, 5.5, facecolor='r', alpha=alpha, label='No Optical')
   ax[1].axhspan(5.5, 10.5, facecolor='b', alpha=alpha, label='Diverse Cloud')
   ax[1].axhspan(10.5, 13.5, facecolor='y', alpha=alpha, label='Diverse Cloud (Pre-trained)')

   sns.barplot(data = global_results_resunet, y ='full_name', x = metric, hue = 'cond', orient = 'y',ax = ax[0], edgecolor='k', errorbar = None, legend=False, width = 0.9, estimator='sum')
   sns.barplot(data = global_results_swin, y ='full_name', x = metric, hue = 'cond', orient = 'y', ax = ax[1], edgecolor='k', errorbar = None, legend=True, estimator='sum')
   sns.lineplot(data = global_results_resunet, y ='full_name', x = 'max', color= 'b', ax= ax[0], label = 'Best Result', legend=False, linestyle = 'dotted')
   sns.lineplot(data = global_results_swin, y ='full_name', x = 'max', color= 'b', ax = ax[1], label = 'Best Result', legend=True, linestyle = 'dotted')
   
   ax[0].axvline(x=resunet_max, color = 'b', linestyle = 'dotted')
   ax[1].axvline(x=swin_max, color = 'b', linestyle = 'dotted')
   
   sns.move_legend(ax[1], "upper left", bbox_to_anchor=(-2.25, 1))
   plt.setp(ax[1].get_legend().get_texts(), fontsize='12') 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   ax[0].set_xlim([0,1.05])
   ax[1].set_xlim([0,1.05])
   ax[1].set_yticklabels([])
   # plt.sca(ax[0])
   #plt.xticks(ha='left', rotation = 325)
   # plt.sca(ax[1])
   # plt.xticks(ha='left', rotation = 325)
   #ax[0].tick_params(axis='x', which='major', labelsize=11)
   #ax[1].tick_params(axis='x', which='major', labelsize=11)
   ax[0].set_ylabel('Model')
   #ax[1].set_xlabel('Model')
   ax[0].set_xlabel(metric_header)
   ax[1].set_xlabel(metric_header)
   ax[1].set_ylabel(None)
   ax[0].set_title(f'Base Architecture: ResUnet')
   ax[1].set_title(f'Base Architecture: Swin')
   for bars in ax[0].containers:
      ax[0].bar_label(bars, fontsize=12, fmt='%.2f', fontweight='bold', padding = 2, color = 'k')
   for bars in ax[1].containers:
      ax[1].bar_label(bars, fontsize=12, fmt='%.2f', fontweight='bold', padding = 2, color = 'k')
   st.pyplot(fig)
   plt.savefig(f'figures/fusion-clouds-{site_code}-{metric}.png', dpi=300, bbox_inches='tight')
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
   
   