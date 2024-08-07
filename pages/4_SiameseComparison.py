import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow import get_site_results, get_site_time_results
import numpy as np
import pandas as pd

#st.set_page_config(layout="wide")

def plot_siamese_com(results, metric, metric_name, site_code):
   sns.set_theme(style="darkgrid")
   
   opt_results = results[
      (results['name'] == 'Optical') &
      (results['cond'] == 'global') 
   ]
   
   
   opt_results = opt_results.rename(columns={
      'base_architecture': 'Base Architecture',
      'siamese': 'Temporal Aggregation',
   })
   
   opt_results['Base Architecture'] = opt_results['Base Architecture'].replace('resunet', 'ResUnet')
   opt_results['Base Architecture'] = opt_results['Base Architecture'].replace('transformer', 'Swin')
   opt_results['Temporal Aggregation'] = opt_results['Temporal Aggregation'].astype('str')
   opt_results['Temporal Aggregation'] = opt_results['Temporal Aggregation'].replace('False', 'Single-stream')
   opt_results['Temporal Aggregation'] = opt_results['Temporal Aggregation'].replace('True', 'Multi-stream')
   
   sns.set_theme(font_scale=1.8)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(figsize=(8,5))
   sns.barplot(data = opt_results, x ='Base Architecture', y = metric, hue = 'Temporal Aggregation', edgecolor='k', errorbar = None, gap=0.05, estimator='sum')
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax.get_legend().get_texts(), fontsize='16') 
   plt.setp(ax.get_legend().get_title(), fontsize='18') 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.ylim([0.3,1.05])
   plt.ylabel(metric_name)
   plt.title(f'Temporal Aggregation Comparison - CLOUD-FREE (Site {site_code[1]})', pad=35, fontsize=24)
   plt.xticks(ha='center')
   plt.yticks(np.linspace(0.3, 1.0, 8))
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=18, fmt='%.2f', fontweight='bold')
   # for container in ax.containers:
   #    ax.bar_label(container, fontsize=10)
   st.pyplot(fig)
   plt.savefig(f'figures/siamese-opt-{site_code}-{metric_name}.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   
   
   sar_results = results[
      (results['name'] == 'SAR') &
      (results['cond'] == 'global') 
   ]
   
   #sar_results = sar_results.sort_values(by=['siamese', 'sar_condition', 'exp_code'])
   
   sar_results = sar_results.rename(columns={
      'base_architecture': 'Base Architecture',
      'sar_condition': 'SAR Data',
      'siamese': 'Temporal Aggregation',
   })
   
   sar_results['Base Architecture'] = sar_results['Base Architecture'].replace('resunet', 'ResUnet')
   sar_results['Base Architecture'] = sar_results['Base Architecture'].replace('transformer', 'Swin')
   sar_results['SAR Data'] = sar_results['SAR Data'].replace('avg2', 'AVERAGE-2')
   sar_results['SAR Data'] = sar_results['SAR Data'].replace('single2', 'SINGLE-2')
   sar_results['SAR Data'] = sar_results['SAR Data'].replace('combined', 'AVERAGE-12')
   
   # sar_results.loc[sar_results['Temporal Aggregation'] == False, 'SAR Data'] = sar_results.loc[sar_results['Temporal Aggregation'] ==False, 'SAR Data'] + ' (Single-stream)'
   # sar_results.loc[sar_results['Temporal Aggregation'] == True, 'SAR Data'] = sar_results.loc[sar_results['Temporal Aggregation'] ==True, 'SAR Data'] + ' (Multi-stream)'
   
   sar_results.loc[sar_results['Temporal Aggregation'] == False, 'SAR Data'] =  'Single-stream [' + sar_results.loc[sar_results['Temporal Aggregation'] ==False, 'SAR Data'] + ']'
   sar_results.loc[sar_results['Temporal Aggregation'] == True, 'SAR Data'] = 'Multi-stream [' + sar_results.loc[sar_results['Temporal Aggregation'] ==True, 'SAR Data'] + ']'
   
   
   sar_results = sar_results.rename(columns={
      'SAR Data': 'Temporal Aggregation [SAR Dataset]',
   })
   
   sns.set_theme(font_scale=2.5)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(figsize=(10,7))
   fig.tight_layout()
   sns.barplot(data = sar_results, x ='Base Architecture', y = metric, hue = 'Temporal Aggregation [SAR Dataset]', edgecolor='k', errorbar = None, gap=0.1, estimator='sum')
   plt.title(f'Temporal Aggregation Comparison - SAR datasets (Site {site_code[1]})', pad=35, fontsize=36)
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax.get_legend().get_texts(), fontsize=18) 
   plt.setp(ax.get_legend().get_title(), fontsize=20) 
   #sns.move_legend(ax, "upper right", bbox_to_anchor=(1.0, 1.3))
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.ylim([0.3,1.05])
   plt.ylabel(metric_name)
   plt.xticks(ha='center', rotation = 0)
   plt.yticks(np.linspace(0.3, 1.0, 8))
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=17, fmt='%.2f', fontweight='bold') #, padding = -20, color = 'w')
   st.pyplot(fig)
   plt.savefig(f'figures/siamese-sar-{site_code}-{metric_name}.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   
def plot_siamese_time_com(results):
   sns.set_theme(style="darkgrid")
   
   opt_results = results[
      (results['name'] == 'Optical')
   ]
   
   opt_results = opt_results.rename(columns={
      'base_architecture': 'Base Architecture',
      'siamese': 'Temporal Aggregation',
      'value': 'Average Time',
   })
   
   opt_results['Base Architecture'] = opt_results['Base Architecture'].replace('resunet', 'ResUnet')
   opt_results['Base Architecture'] = opt_results['Base Architecture'].replace('transformer', 'Swin')
   opt_results['Temporal Aggregation'] = opt_results['Temporal Aggregation'].astype('str')
   opt_results['Temporal Aggregation'] = opt_results['Temporal Aggregation'].replace('False', 'Single-stream')
   opt_results['Temporal Aggregation'] = opt_results['Temporal Aggregation'].replace('True', 'Multi-stream')
   
   sns.set_theme(font_scale=1.5)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(1, 2, figsize=(12,6))
   sns.barplot(data = opt_results[opt_results['stage'] == 'Training'], x ='Base Architecture', y = 'Average Time', hue = 'Temporal Aggregation', ax = ax[0], edgecolor='k', errorbar = 'sd', gap=0.05, estimator='mean', legend = False)
   sns.barplot(data = opt_results[opt_results['stage'] == 'Prediction'], x ='Base Architecture', y = 'Average Time', hue = 'Temporal Aggregation', ax = ax[1], edgecolor='k', errorbar = 'sd', gap=0.05, estimator='mean', legend = True)
   sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
   #plt.setp(ax.get_legend().get_texts(), fontsize='16') 
   #plt.setp(ax.get_legend().get_title(), fontsize='18') 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.suptitle('Training and Prediction times for 10 epochs (CLOUD-FREE dataset)')
   ax[0].set_ylim([0,3])
   ax[1].set_ylim([0,3])
   ax[0].set_ylabel('Average Training Time (s)')
   ax[1].set_ylabel('Average Prediction Time (s)')
   ax[0].set_title('Training Time')
   ax[1].set_title('Prediction Time')
   # plt.title(f'Average Training Time (10 Epochs) - CLOUD-FREE', pad=35, fontsize=24)
   # plt.xticks(ha='center')
   #plt.yticks(np.linspace(0.3, 1.0, 8))
   for ax_i in ax:
      for bars in ax_i.containers:
         ax_i.bar_label(bars, fontsize=16, fmt='%.2f', padding = 10, fontweight='bold')
   # for container in ax.containers:
   #    ax.bar_label(container, fontsize=10)
   plt.tight_layout()
   st.pyplot(fig)
   plt.savefig(f'figures/time-siamese-opt.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   
   sar_results = results[
      (results['name'] == 'SAR')
   ]
   
   
   sar_results = sar_results.rename(columns={
      'base_architecture': 'Base Architecture',
      'sar_condition': 'SAR Data',
      'siamese': 'Temporal Aggregation',
      'value': 'Average Time',
   })
   
   sar_results['Base Architecture'] = sar_results['Base Architecture'].replace('resunet', 'ResUnet')
   sar_results['Base Architecture'] = sar_results['Base Architecture'].replace('transformer', 'Swin')
   sar_results['SAR Data'] = sar_results['SAR Data'].replace('avg2', 'AVERAGE-2')
   sar_results['SAR Data'] = sar_results['SAR Data'].replace('single2', 'SINGLE-2')
   sar_results['SAR Data'] = sar_results['SAR Data'].replace('combined', 'AVERAGE-12')
   
   # sar_results.loc[sar_results['Temporal Aggregation'] == False, 'SAR Data'] = sar_results.loc[sar_results['Temporal Aggregation'] ==False, 'SAR Data'] + ' (Single-stream)'
   # sar_results.loc[sar_results['Temporal Aggregation'] == True, 'SAR Data'] = sar_results.loc[sar_results['Temporal Aggregation'] ==True, 'SAR Data'] + ' (Multi-stream)'
   
   sar_results.loc[sar_results['Temporal Aggregation'] == False, 'SAR Data'] =  'Single-stream [' + sar_results.loc[sar_results['Temporal Aggregation'] ==False, 'SAR Data'] + ']'
   sar_results.loc[sar_results['Temporal Aggregation'] == True, 'SAR Data'] = 'Multi-stream [' + sar_results.loc[sar_results['Temporal Aggregation'] ==True, 'SAR Data'] + ']'
   
   
   sar_results = sar_results.rename(columns={
      'SAR Data': 'Temporal Aggregation [SAR Dataset]',
   })
   
   sns.set_theme(font_scale=1.5)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(1, 2, figsize=(12,6))
   sns.barplot(data = sar_results[sar_results['stage'] == 'Training'], x ='Base Architecture', y = 'Average Time', hue = 'Temporal Aggregation [SAR Dataset]', ax = ax[0], edgecolor='k', errorbar = 'sd', gap=0.05, estimator='mean', legend = False)
   sns.barplot(data = sar_results[sar_results['stage'] == 'Prediction'], x ='Base Architecture', y = 'Average Time', hue = 'Temporal Aggregation [SAR Dataset]', ax = ax[1], edgecolor='k', errorbar = 'sd', gap=0.05, estimator='mean', legend = True)
   sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax[1].get_legend().get_texts(), fontsize='10') 
   plt.setp(ax[1].get_legend().get_title(), fontsize='12') 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.suptitle('Training and Prediction times for 10 epochs (SAR datasets)')
   ax[0].set_ylim([0,3])
   ax[1].set_ylim([0,3])
   ax[0].set_ylabel('Average Training Time (s)')
   ax[1].set_ylabel('Average Prediction Time (s)')
   ax[0].set_title('Training Time')
   ax[1].set_title('Prediction Time')
   # plt.title(f'Average Training Time (10 Epochs) - CLOUD-FREE', pad=35, fontsize=24)
   # plt.xticks(ha='center')
   #plt.yticks(np.linspace(0.3, 1.0, 8))
   for ax_i in ax:
      for bars in ax_i.containers:
         ax_i.bar_label(bars, fontsize=16, fmt='%.2f', padding = 10, fontweight='bold', rotation=90)
   # for container in ax.containers:
   #    ax.bar_label(container, fontsize=10)
   plt.tight_layout()
   st.pyplot(fig)
   plt.savefig(f'figures/time-siamese-sar.png', dpi=300, bbox_inches='tight')
   plt.close(fig)

time_results = None
for site_code in st.session_state['sites']:
   sns.set_theme(style="darkgrid")
   
   site_name = st.session_state['sites'][site_code]['name']
   #results_site = get_site_results(site_name, st.session_state['experiments'])
   #st.table(results_site)
   
   st.header(site_name)
   
   #siamese comparison no cloud
   exps = [101, 103, 106, 107, 156, 157, 151, 153, 204, 205, 206, 254, 255, 256]
   exp_codes = [f'exp_{code}' for code in exps]
   
   results = get_site_results(site_name, st.session_state['experiments'], exp_codes)
   
   
   plot_siamese_com(results, 'f1score', 'F1-Score', site_code)
   plot_siamese_com(results, 'precision', 'Precision', site_code)
   plot_siamese_com(results, 'recall', 'Recall', site_code)
   
   if time_results is None:
      time_results = get_site_time_results(site_name, st.session_state['experiments'], exp_codes)
   else:
      time_results = pd.concat([time_results, get_site_time_results(site_name, st.session_state['experiments'], exp_codes)])
   
plot_siamese_time_com(time_results)
   
   