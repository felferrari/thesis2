import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow import get_site_time_results
import numpy as np
import pandas as pd

#st.set_page_config(layout="wide")

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
   fig, ax = plt.subplots(1, 2, figsize=(12,5))
   sns.barplot(data = opt_results[opt_results['stage'] == 'Training'], x ='Base Architecture', y = 'Average Time', hue = 'Temporal Aggregation', ax = ax[0], edgecolor='k', errorbar = 'sd', err_kws = {'alpha': 0.5}, capsize=0.05, gap=0.05, estimator='mean', legend = False)
   sns.barplot(data = opt_results[opt_results['stage'] == 'Prediction'], x ='Base Architecture', y = 'Average Time', hue = 'Temporal Aggregation', ax = ax[1], edgecolor='k', errorbar = 'sd', err_kws = {'alpha': 0.5}, capsize=0.05, gap=0.05, estimator='mean', legend = True)
   sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
   #plt.setp(ax.get_legend().get_texts(), fontsize='16') 
   #plt.setp(ax.get_legend().get_title(), fontsize='18') 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.suptitle('Models\' times - Temporal Aggregation - Temporal Aggregation (CLOUD-FREE dataset)')
   ax[0].set_ylim([0,3])
   ax[1].set_ylim([0,3])
   ax[0].set_ylabel('Average Time (s)')
   ax[1].set_ylabel('Average Time (s)')
   ax[0].set_title('Training')
   ax[1].set_title('Prediction')
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
   fig, ax = plt.subplots(1, 2, figsize=(12,5))
   sns.barplot(data = sar_results[sar_results['stage'] == 'Training'], x ='Base Architecture', y = 'Average Time', hue = 'Temporal Aggregation [SAR Dataset]', ax = ax[0], edgecolor='k', errorbar = 'sd', err_kws = {'alpha': 0.5}, capsize=0.05, gap=0.05, estimator='mean', legend = False)
   sns.barplot(data = sar_results[sar_results['stage'] == 'Prediction'], x ='Base Architecture', y = 'Average Time', hue = 'Temporal Aggregation [SAR Dataset]', ax = ax[1], edgecolor='k', errorbar = 'sd', err_kws = {'alpha': 0.5}, capsize=0.05, gap=0.05, estimator='mean', legend = True)
   sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax[1].get_legend().get_texts(), fontsize='10') 
   plt.setp(ax[1].get_legend().get_title(), fontsize='12') 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.suptitle('Models\' times - Temporal Aggregation (SAR datasets)')
   ax[0].set_ylim([0,3])
   ax[1].set_ylim([0,3])
   ax[0].set_ylabel('Average Time (s)')
   ax[1].set_ylabel('Average Time (s)')
   ax[0].set_title('Training')
   ax[1].set_title('Prediction')
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
   
def plot_prevmap_time_com(results):
   sns.set_theme(style="darkgrid")
   
   opt_results = results[
      (results['name'] == 'Optical')
   ]
   
   
   opt_results = opt_results.rename(columns={
      'base_architecture': 'Base Architecture',
      'prev_map': 'Previous Map',
      'siamese': 'Siamese',
      'value': 'Average Time',
   })
   
   opt_results['Base Architecture'] = opt_results['Base Architecture'].replace('resunet', 'ResUnet')
   opt_results['Base Architecture'] = opt_results['Base Architecture'].replace('transformer', 'Swin')
   
   opt_results.loc[opt_results['Siamese'] == True, 'Base Architecture'] = opt_results.loc[opt_results['Siamese'] ==True, 'Base Architecture'] + ' (Multi-stream)'
   opt_results.loc[opt_results['Siamese'] == False, 'Base Architecture'] = opt_results.loc[opt_results['Siamese'] ==False, 'Base Architecture'] + ' (Single-stream)'

   
   sns.set_theme(font_scale=1.4)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(2, 1, figsize=(10,7))
   sns.barplot(data = opt_results[opt_results['stage'] == 'Training'], x ='Base Architecture', y = 'Average Time', hue = 'Previous Map', ax = ax[0], edgecolor='k', errorbar = 'sd', err_kws = {'alpha': 0.5}, capsize=0.05, gap=0.05, estimator='mean', legend = True)
   sns.barplot(data = opt_results[opt_results['stage'] == 'Prediction'], x ='Base Architecture', y = 'Average Time', hue = 'Previous Map', ax = ax[1], edgecolor='k', errorbar = 'sd', err_kws = {'alpha': 0.5}, capsize=0.05, gap=0.05, estimator='mean', legend = False)
   sns.move_legend(ax[0], "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax[0].get_legend().get_texts(), fontsize='12') 
   plt.setp(ax[0].get_legend().get_title(), fontsize='14') 
   plt.setp(ax[0].get_xticklabels(), fontsize=15) 
   plt.setp(ax[1].get_xticklabels(), fontsize=15) 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.suptitle('Models\' times - Previous Deforestation Map (CLOUD-FREE dataset)')
   ax[0].set_ylim([0,3])
   ax[1].set_ylim([0,3])
   ax[0].set_ylabel('Average Time (s)')
   ax[1].set_ylabel('Average Time (s)')
   ax[0].set_title('Training')
   ax[1].set_title('Prediction')
   
   ax[0].set_xlabel('')
   ax[0].set_xticks([])
   
   # plt.sca(ax[0])
   # plt.xticks(ha='left', rotation = 345)
   plt.sca(ax[1])
   plt.xticks(ha='left', rotation = 350)
   # plt.title(f'Average Training Time (10 Epochs) - CLOUD-FREE', pad=35, fontsize=24)
   # plt.xticks(ha='center')
   #plt.yticks(np.linspace(0.3, 1.0, 8))
   for ax_i in ax:
      for bars in ax_i.containers:
         ax_i.bar_label(bars, fontsize=14, fmt='%.2f', padding = 3, fontweight='bold', rotation=0)
   # for container in ax.containers:
   #    ax.bar_label(container, fontsize=10)
   plt.tight_layout()
   st.pyplot(fig)
   plt.savefig(f'figures/time-prevmap-opt.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   
   
   
   sar_results = results[
      (results['name'] == 'SAR') 
   ]
   
   #f1_global_sar_results = f1_global_sar_results.sort_values(by=['siamese', 'sar_condition', 'exp_code'])
   
   sar_results = sar_results.rename(columns={
      'base_architecture': 'Base Architecture',
      'sar_condition': 'SAR Data',
      'prev_map': 'Previous Map',
      'siamese': 'Siamese',
      'value': 'Average Time',
   })
   
   sar_results['Base Architecture'] = sar_results['Base Architecture'].replace('resunet', 'ResUnet')
   sar_results['Base Architecture'] = sar_results['Base Architecture'].replace('transformer', 'Swin')
   sar_results['SAR Data'] = sar_results['SAR Data'].replace('avg2', 'AVERAGE-2')
   sar_results['SAR Data'] = sar_results['SAR Data'].replace('single2', 'SINGLE-2')
   sar_results['SAR Data'] = sar_results['SAR Data'].replace('combined', 'AVERAGE-12')
   
   #f1_global_sar_results.loc[f1_global_sar_results['Siamese'] == True, 'SAR Data'] = f1_global_sar_results.loc[f1_global_sar_results['Siamese'] ==True, 'SAR Data'] + ' (Siamese)'
   sar_results.loc[sar_results['Siamese'] == True, 'Base Architecture'] = sar_results.loc[sar_results['Siamese'] ==True, 'Base Architecture'] + ' (Multi-stream)'
   sar_results.loc[sar_results['Siamese'] == False, 'Base Architecture'] = sar_results.loc[sar_results['Siamese'] ==False, 'Base Architecture'] + ' (Single-stream)'
   sar_results['Base Architecture'] = sar_results['Base Architecture'] + ' [' + sar_results['SAR Data'] + ']'
   
   
   
   sns.set_theme(font_scale=1.3)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(2, 1, figsize=(10,7))
   sns.barplot(data = sar_results[sar_results['stage'] == 'Training'], x ='Base Architecture', y = 'Average Time', hue = 'Previous Map', ax = ax[0], edgecolor='k', errorbar = 'sd', err_kws = {'alpha': 0.5}, capsize=0.05, gap=0.05, estimator='mean', legend = True)
   sns.barplot(data = sar_results[sar_results['stage'] == 'Prediction'], x ='Base Architecture', y = 'Average Time', hue = 'Previous Map', ax = ax[1], edgecolor='k', errorbar = 'sd', err_kws = {'alpha': 0.5}, capsize=0.05, gap=0.05, estimator='mean', legend = False)
   sns.move_legend(ax[0], "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax[0].get_legend().get_texts(), fontsize='10') 
   plt.setp(ax[0].get_legend().get_title(), fontsize='12') 
   plt.setp(ax[0].get_xticklabels(), fontsize=14) 
   plt.setp(ax[1].get_xticklabels(), fontsize=14) 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.suptitle('Models\' times - Previous Deforestation Map (SAR datasets)')
   ax[0].set_ylim([0,3])
   ax[1].set_ylim([0,3])
   ax[0].set_ylabel('Average Time (s)')
   ax[1].set_ylabel('Average Time (s)')
   ax[0].set_title('Training')
   ax[1].set_title('Prediction')
   ax[0].set_xticks([])
   ax[0].set_xlabel('')
   
   # plt.sca(ax[0])
   # plt.xticks(ha='left', rotation = 325)
   plt.sca(ax[1])
   plt.xticks(ha='left', rotation = 345)
   # plt.title(f'Average Training Time (10 Epochs) - CLOUD-FREE', pad=35, fontsize=24)
   # plt.xticks(ha='center')
   #plt.yticks(np.linspace(0.3, 1.0, 8))
   for ax_i in ax:
      for bars in ax_i.containers:
         ax_i.bar_label(bars, fontsize=11, fmt='%.2f', padding = 3, fontweight='bold', rotation=0)
   # for container in ax.containers:
   #    ax.bar_label(container, fontsize=10)
   plt.tight_layout()
   st.pyplot(fig)
   plt.savefig(f'figures/time-prevmap-sar.png', dpi=300, bbox_inches='tight')
   plt.close(fig)

def plot_fusion_time_com(results):
   sns.set_theme(style="darkgrid")
   
   opt_results = results.copy()
   
   
   opt_results = opt_results.rename(columns={
      'base_architecture': 'Base Architecture',
      'prev_map': 'Previous Map',
      'siamese': 'Siamese',
      'value': 'Average Time',
      'name': 'Model',
   })
   
   opt_results['Base Architecture'] = opt_results['Base Architecture'].replace('resunet', 'ResUnet')
   opt_results['Base Architecture'] = opt_results['Base Architecture'].replace('transformer', 'Swin')
   
   sns.set_theme(font_scale=1.5)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(2, 1, figsize=(10,7))
   sns.barplot(data = opt_results[opt_results['stage'] == 'Training'], x ='Base Architecture', y = 'Average Time', hue = 'Model', ax = ax[0], edgecolor='k', errorbar = 'sd', err_kws = {'alpha': 0.5}, capsize=0.05, gap=0.05, estimator='mean', legend = True)
   sns.barplot(data = opt_results[opt_results['stage'] == 'Prediction'], x ='Base Architecture', y = 'Average Time', hue = 'Model', ax = ax[1], edgecolor='k', errorbar = 'sd', err_kws = {'alpha': 0.5}, capsize=0.05, gap=0.05, estimator='mean', legend = False)
   sns.move_legend(ax[0], "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax[0].get_legend().get_texts(), fontsize='10') 
   plt.setp(ax[0].get_legend().get_title(), fontsize='12') 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.suptitle('Models\' times - Fusion Models')
   ax[0].set_ylim([0,4.55])
   ax[1].set_ylim([0,4.55])
   ax[0].set_ylabel('Average Time (s)')
   ax[1].set_ylabel('Average Time (s)')
   ax[0].set_title('Training')
   ax[1].set_title('Prediction')
   
   plt.sca(ax[0])
   plt.xticks(ha='left')
   plt.sca(ax[1])
   plt.xticks(ha='left')
   # plt.title(f'Average Training Time (10 Epochs) - CLOUD-FREE', pad=35, fontsize=24)
   # plt.xticks(ha='center')
   #plt.yticks(np.linspace(0.3, 1.0, 8))
   for ax_i in ax:
      for bars in ax_i.containers:
         ax_i.bar_label(bars, fontsize=14, fmt='%.2f', padding = 3, fontweight='bold', rotation=0)
   # for container in ax.containers:
   #    ax.bar_label(container, fontsize=10)
   plt.tight_layout()
   st.pyplot(fig)
   plt.savefig(f'figures/time-fusion.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   




time_results = None
for site_code in st.session_state['sites']:
   sns.set_theme(style="darkgrid")
   
   site_name = st.session_state['sites'][site_code]['name']
   st.header(site_name)
   
   exps = [101, 103, 106, 107, 156, 157, 151, 153, 204, 205, 206, 254, 255, 256]
   exp_codes = [f'exp_{code}' for code in exps]
   
   if time_results is None:
      time_results = get_site_time_results(site_name, st.session_state['experiments'], exp_codes)
   else:
      time_results = pd.concat([time_results, get_site_time_results(site_name, st.session_state['experiments'], exp_codes)])
   
plot_siamese_time_com(time_results)
   
time_results = None
for site_code in st.session_state['sites']:
   sns.set_theme(style="darkgrid")
   
   site_name = st.session_state['sites'][site_code]['name']
   
   st.header(site_name, divider='red')
   
   exps =        [101, 151, 104, 154, 201, 204, 251, 254]
   exps = exps + [103, 105, 202, 203, 205, 206, 153, 155, 252, 253, 255, 256]
   exp_codes = [f'exp_{code}' for code in exps]

   if time_results is None:
      time_results = get_site_time_results(site_name, st.session_state['experiments'], exp_codes)
   else:
      time_results = pd.concat([time_results, get_site_time_results(site_name, st.session_state['experiments'], exp_codes)])
   
plot_prevmap_time_com(time_results)

time_results = None
for site_code in st.session_state['sites']:
   sns.set_theme(style="darkgrid")
   
   site_name = st.session_state['sites'][site_code]['name']
   #results_site = get_site_results(site_name, st.session_state['experiments'])
   #st.table(results_site)
   
   st.header(site_name)
   
   #siamese comparison no cloud
   exps = [101, 301, 302, 303, 306, 103, 151, 351, 352, 353, 356, 153]
   exp_codes = [f'exp_{code}' for code in exps]
   
   if time_results is None:
      time_results = get_site_time_results(site_name, st.session_state['experiments'], exp_codes)
   else:
      time_results = pd.concat([time_results, get_site_time_results(site_name, st.session_state['experiments'], exp_codes)])
   
plot_fusion_time_com(time_results)