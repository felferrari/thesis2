import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow import get_uncertainty_data
import seaborn.objects  as so
import numpy as np
import pandas as pd

# st.set_page_config(layout="wide")

def plot_simple_graphic(results, metric, metric_header, site_code):
   
   global_results = results.copy()
   
   original_results = global_results[global_results['percentile'] == 0.0]
   original_results['Audited'] = False
   audited_results = global_results[global_results['percentile'] == 3.0]
   audited_results['Audited'] = True
   
   mix_results = pd.concat([original_results, audited_results])
   
   mix_results['full_name'] = mix_results['full_name'].str.replace('ResUnet ', '')
   mix_results['full_name'] = mix_results['full_name'].str.replace('SWIN ', '')
   mix_results['full_name'] = mix_results['full_name'].str.replace('[CLOUD-FREE]', ' ')
   mix_results['full_name'] = mix_results['full_name'].str.replace('[CLOUD-DIVERSE]', '  ')
   mix_results['full_name'] = mix_results['full_name'].str.replace('[PRE-TRAINED]', '   ')
   mix_results['full_name'] = mix_results['full_name'].str.replace('[AVERAGE-12]', '    ')
   
   global_results_resunet = mix_results[mix_results['base_architecture'] == 'resunet']
   global_results_swin = mix_results[mix_results['base_architecture'] == 'transformer']
   
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
   ax.axvspan(4.5, 5.5, facecolor='r', alpha=alpha, label='Non-Optical')
   ax.axvspan(5.5, 10.5, facecolor='b', alpha=alpha, label='Diverse')
   ax.axvspan(10.5, 13.5, facecolor='y', alpha=alpha, label='Diverse [Pre-trained]')
   plt.plot([], [], ' ', label="--Audited--")

   sns.barplot(data = global_results_resunet, x ='full_name', y = metric, hue = 'Audited', edgecolor='k', errorbar = None, legend=True, width = 0.9, estimator='sum')
   sns.lineplot(data = global_results_resunet, x ='full_name', y = 'max', color= 'b', label = 'Best Result', legend=True, linestyle = 'dotted', linewidth = 3)
   
   ax.axhline(y=resunet_max, color = 'b', linestyle = 'dotted', linewidth = 3)
   
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax.get_legend().get_texts(), fontsize=15) 
   plt.xticks(ha='left', rotation = 325)
   ax.set_ylim([0.3, 1.12])
   ax.set_xlabel('Model')
   ax.set_ylabel(metric_header)
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=20, fmt='%.2f', fontweight='bold', padding = 5, color = 'k', rotation=90)
   st.pyplot(fig)
   plt.savefig(f'figures/entropy-fusion-clouds-resunet-{site_code}-{metric}.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   
   sns.set_theme(font_scale=2)
   sns.set_palette('tab10')
   
   fig, ax = plt.subplots(1,1,figsize=(13,6))
   #fig.tight_layout()
   alpha = 0.4
   
   fig.suptitle(f'{metric_header} Base Architecture: Swin (Site {site_code[1]})', y= 1.1)
   
   plt.plot([], [], ' ', label="--Dataset--")
   ax.axvspan(-0.5, 4.5, facecolor='g', alpha=alpha, label='Cloud-Free')
   ax.axvspan(4.5, 5.5, facecolor='r', alpha=alpha, label='Non-Optical')
   ax.axvspan(5.5, 10.5, facecolor='b', alpha=alpha, label='Diverse')
   ax.axvspan(10.5, 13.5, facecolor='y', alpha=alpha, label='Diverse [Pre-trained]')
   plt.plot([], [], ' ', label="--Audited--")

   sns.barplot(data = global_results_swin, x ='full_name', y = metric, hue = 'Audited', edgecolor='k', errorbar = None, legend=True, width = 0.9, estimator='sum')
   sns.lineplot(data = global_results_swin, x ='full_name', y = 'max', color= 'b', label = 'Best Result', legend=True, linestyle = 'dotted', linewidth = 3)
   
   ax.axhline(y=swin_max, color = 'b', linestyle = 'dotted', linewidth = 3)
   
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax.get_legend().get_texts(), fontsize=15) 
   plt.xticks(ha='left', rotation = 325)
   ax.set_ylim([0.3, 1.12])
   ax.set_xlabel('Model')
   ax.set_ylabel(metric_header)
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=20, fmt='%.2f', fontweight='bold', padding = 5, color = 'k', rotation=90)
   st.pyplot(fig)
   plt.savefig(f'figures/entropy-fusion-clouds-swin-{site_code}-{metric}.png', dpi=300, bbox_inches='tight')
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
   
   results = get_uncertainty_data(site_name, st.session_state['experiments'], exp_codes)
   
   plot_simple_graphic(results, 'f1score', 'F1-Score', site_code)
   plot_simple_graphic(results, 'precision', 'Precision', site_code)
   plot_simple_graphic(results, 'recall', 'Recall', site_code)
   
   
   