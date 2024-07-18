import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow import get_exps_metric, get_site_time_results
import numpy as np
import pandas as pd

#st.set_page_config(layout="wide")

def plot_siamese_epochs_com(results, site_code):
   sns.set_theme(style="darkgrid")
   
   opt_results = results[
      (results['name'] == 'Optical')
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
   
   opt_results = opt_results.groupby(['full_name', 'Base Architecture', 'Temporal Aggregation', 'model']).size().reset_index(name='Epochs')
   
   sns.set_theme(font_scale=1.5)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(figsize=(12,5))
   sns.barplot(data = opt_results, x ='Base Architecture', y = 'Epochs', hue = 'Temporal Aggregation', ax = ax, edgecolor='k', errorbar = 'sd', capsize=0.05, gap=0.05, estimator='mean', legend = True)
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   #plt.setp(ax.get_legend().get_texts(), fontsize='16') 
   #plt.setp(ax.get_legend().get_title(), fontsize='18') 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.suptitle(f'Training Epochs - Temporal Aggregation (Optical Models) - Site {site_code[-1]}')
   #ax.set_ylim([0,140])
   ax.set_ylabel('Epochs')
   # plt.title(f'Average Training Time (10 Epochs) - CLOUD-FREE', pad=35, fontsize=24)
   # plt.xticks(ha='center')
   #plt.yticks(np.linspace(0.3, 1.0, 8))
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=16, fmt='%.1f', padding = 3, fontweight='bold')
   # for container in ax.containers:
   #    ax.bar_label(container, fontsize=10)
   plt.tight_layout()
   st.pyplot(fig)
   plt.savefig(f'figures/epochs-siamese-opt-{site_code}.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   
   sar_results = results[
      (results['name'] == 'SAR')
   ]
   
   sar_results = sar_results.rename(columns={
      'base_architecture': 'Base Architecture',
      'sar_condition': 'SAR Data',
      'siamese': 'Temporal Aggregation',
      'value': 'Millions of Paramters',
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
   
   sar_results = sar_results.groupby(['full_name', 'Base Architecture', 'Temporal Aggregation [SAR Dataset]', 'model']).size().reset_index(name='Epochs')
   
   sns.set_theme(font_scale=1.5)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(figsize=(12,5))
   sns.barplot(data = sar_results, x ='Base Architecture', y = 'Epochs', hue = 'Temporal Aggregation [SAR Dataset]', ax = ax, edgecolor='k', errorbar = 'sd', capsize=0.05, gap=0.05, estimator='mean', legend = True)
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax.get_legend().get_texts(), fontsize='12') 
   plt.setp(ax.get_legend().get_title(), fontsize='14') 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.suptitle(f'Training Epochs - Temporal Aggregation (SAR Models) - Site {site_code[-1]}')
   #ax.set_ylim([0,35])
   ax.set_ylabel('Epochs')
   # ax.set_ylabel('Average Time (s)')
   # ax.set_title('Training')
   # plt.title(f'Average Training Time (10 Epochs) - CLOUD-FREE', pad=35, fontsize=24)
   # plt.xticks(ha='center')
   #plt.yticks(np.linspace(0.3, 1.0, 8))
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=15, fmt='%.1f', padding = 3, fontweight='bold', rotation=0)
   # for container in ax.containers:
   #    ax.bar_label(container, fontsize=10)
   plt.tight_layout()
   st.pyplot(fig)
   plt.savefig(f'figures/epochs-siamese-sar-{site_code}.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   
def plot_prevmap_epochs_com(results, site_code):
   sns.set_theme(style="darkgrid")
   
   opt_results = results[
      (results['name'] == 'Optical')
   ]
   
   opt_results = opt_results.rename(columns={
      'base_architecture': 'Base Architecture',
      'prev_map': 'Previous Map',
      'siamese': 'Siamese',
   })
   
   opt_results['Base Architecture'] = opt_results['Base Architecture'].replace('resunet', 'ResUnet')
   opt_results['Base Architecture'] = opt_results['Base Architecture'].replace('transformer', 'Swin')
   
   opt_results.loc[opt_results['Siamese'] == True, 'Base Architecture'] = opt_results.loc[opt_results['Siamese'] ==True, 'Base Architecture'] + ' (Multi-stream)'
   opt_results.loc[opt_results['Siamese'] == False, 'Base Architecture'] = opt_results.loc[opt_results['Siamese'] ==False, 'Base Architecture'] + ' (Single-stream)'

   opt_results = opt_results.groupby(['full_name', 'Base Architecture', 'Previous Map', 'model']).size().reset_index(name='Epochs')
   
   sns.set_theme(font_scale=1.4)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(figsize=(10,6))
   sns.barplot(data = opt_results, x ='Base Architecture', y = 'Epochs', hue = 'Previous Map', ax = ax, edgecolor='k', errorbar = 'sd', capsize=0.05, gap=0.05, estimator='mean', legend = True)
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax.get_legend().get_texts(), fontsize='12') 
   plt.setp(ax.get_legend().get_title(), fontsize='14') 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.suptitle(f'Training Epochs - Previous Deforestation Map (Optical Models) - Site {site_code[-1]}')
   plt.xticks(ha='left', rotation = 350)
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=14, fmt='%.1f', padding = 3, fontweight='bold', rotation=0)
   plt.tight_layout()
   st.pyplot(fig)
   plt.savefig(f'figures/epochs-prevmap-opt-{site_code}.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   
   
   
   sar_results = results[
      (results['name'] == 'SAR') 
   ]
   
   sar_results = sar_results.rename(columns={
      'base_architecture': 'Base Architecture',
      'sar_condition': 'SAR Data',
      'prev_map': 'Previous Map',
      'siamese': 'Siamese',
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
   
   sar_results = sar_results.groupby(['full_name', 'Base Architecture', 'Previous Map', 'model']).size().reset_index(name='Epochs')
   
   sns.set_theme(font_scale=1.3)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(figsize=(10,6))
   sns.barplot(data = sar_results, x ='Base Architecture', y = 'Epochs', hue = 'Previous Map', ax = ax, edgecolor='k', errorbar = 'sd', capsize=0.05, gap=0.05, estimator='mean', legend = True)
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax.get_legend().get_texts(), fontsize='12') 
   plt.setp(ax.get_legend().get_title(), fontsize='14') 
   plt.setp(ax.get_xticklabels(), fontsize=14) 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.suptitle(f'Training Epochs - Previous Deforestation Map (SAR Models) - Site {site_code[-1]}')
   
   plt.xticks(ha='left', rotation = 345)
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=11, fmt='%.1f', padding = 3, fontweight='bold', rotation=0)
   # for container in ax.containers:
   #    ax.bar_label(container, fontsize=10)
   plt.tight_layout()
   st.pyplot(fig)
   plt.savefig(f'figures/epochs-prevmap-sar-{site_code}.png', dpi=300, bbox_inches='tight')
   plt.close(fig)

def plot_fusion_size_com(results, times, site_code):
   sns.set_theme(style="darkgrid")
   
   new_results = results.copy()
   
   new_results = new_results.rename(columns={
      'base_architecture': 'Base Architecture',
      'prev_map': 'Previous Map',
      'siamese': 'Siamese',
      'name': 'Model',
   })
   
   new_results['Base Architecture'] = new_results['Base Architecture'].replace('resunet', 'ResUnet')
   new_results['Base Architecture'] = new_results['Base Architecture'].replace('transformer', 'Swin')
   
   new_results = new_results.groupby(['full_name', 'Base Architecture', 'Previous Map', 'Model', 'model'], sort=False).size().reset_index(name='Epochs')
   
   new_times = times.copy()
   new_times = new_times.rename(columns={
      'base_architecture': 'Base Architecture',
      'prev_map': 'Previous Map',
      'siamese': 'Siamese',
      'name': 'Model',
   })
   
   new_times['Base Architecture'] = new_times['Base Architecture'].replace('resunet', 'ResUnet')
   new_times['Base Architecture'] = new_times['Base Architecture'].replace('transformer', 'Swin')
   
   new_times = new_times.groupby(['full_name'])['value'].mean().reset_index(name='time')
   
   new_results = pd.merge(new_results, new_times, on='full_name')
   
   new_results['full_time'] = new_results['Epochs'] * new_results['time'] * 200 / 3600
   
   sns.set_theme(font_scale=1.5)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(figsize=(10,6))
   sns.barplot(data = new_results, x ='Base Architecture', y = 'full_time', hue = 'Model', ax = ax, edgecolor='k', errorbar = 'sd', err_kws = {'alpha': 0.5}, capsize=0.05, gap=0.05, estimator='mean', legend = True)
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax.get_legend().get_texts(), fontsize='10') 
   plt.setp(ax.get_legend().get_title(), fontsize='12') 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.suptitle(f'Total Training Times - Fusion Models')
   plt.xticks(ha='left')
   plt.ylabel('Average Total Training Time (hours)')
   # plt.title(f'Average Training Time (10 Epochs) - CLOUD-FREE', pad=35, fontsize=24)
   # plt.xticks(ha='center')
   #plt.yticks(np.linspace(0.3, 1.0, 8))
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=13, fmt='%.1f', padding = 3, fontweight='bold', rotation=0)
   # for container in ax.containers:
   #    ax.bar_label(container, fontsize=10)
   plt.tight_layout()
   st.pyplot(fig)
   plt.savefig(f'figures/train-time-fusion.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   

# epochs_results = None
# for site_code in st.session_state['sites']:
#    sns.set_theme(style="darkgrid")
   
#    site_name = st.session_state['sites'][site_code]['name']
#    st.header(site_name)
   
#    exps = [101, 103, 106, 107, 156, 157, 151, 153, 204, 205, 206, 254, 255, 256]
#    exp_codes = [f'exp_{code}' for code in exps]
   
#    epochs_results = get_exps_metric(site_name, exp_codes, 'train_f1_score_0', st.session_state['experiments'])
   
#    plot_siamese_epochs_com(epochs_results, site_code)
   
# epochs_results = None


# for site_code in st.session_state['sites']:
#    sns.set_theme(style="darkgrid")
   
#    site_name = st.session_state['sites'][site_code]['name']
   
#    st.header(site_name, divider='red')
   
#    exps =        [101, 151, 104, 154, 201, 204, 251, 254]
#    exps = exps + [103, 105, 202, 203, 205, 206, 153, 155, 252, 253, 255, 256]
#    exp_codes = [f'exp_{code}' for code in exps]

#    epochs_results = get_exps_metric(site_name, exp_codes, 'train_f1_score_0', st.session_state['experiments'])
   
#    plot_prevmap_epochs_com(epochs_results, site_code)

epochs, times = None, None
for site_code in st.session_state['sites']:
   sns.set_theme(style="darkgrid")
   
   site_name = st.session_state['sites'][site_code]['name']
   #results_site = get_site_results(site_name, st.session_state['experiments'])
   #st.table(results_site)
   
   st.header(site_name)
   
   #siamese comparison no cloud
   exps = [101, 301, 302, 303, 306, 103, 151, 351, 352, 353, 356, 153]
   exp_codes = [f'exp_{code}' for code in exps]
   
   if epochs is None:
      epochs = get_exps_metric(site_name, exp_codes, 'train_f1_score_0', st.session_state['experiments'])
   else:
      epochs = pd.concat([epochs, get_exps_metric(site_name, exp_codes, 'train_f1_score_0', st.session_state['experiments'])]) 
      
   if times is None:
      times = get_site_time_results(site_name, st.session_state['experiments'], exp_codes)
   else:
      times = pd.concat([times, get_site_time_results(site_name, st.session_state['experiments'], exp_codes)]) 
   
   
plot_fusion_size_com(epochs, times, site_code)