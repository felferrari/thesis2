import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow import get_site_size_results
import numpy as np
import pandas as pd

#st.set_page_config(layout="wide")

def plot_siamese_size_com(results):
   sns.set_theme(style="darkgrid")
   
   opt_results = results[
      (results['name'] == 'Optical')
   ]
   opt_results['value'] = opt_results['value'] / 1000000
   
   opt_results = opt_results.rename(columns={
      'base_architecture': 'Base Architecture',
      'siamese': 'Temporal Aggregation',
      'value': 'Millions of Paramters',
   })
   
   opt_results['Base Architecture'] = opt_results['Base Architecture'].replace('resunet', 'ResUnet')
   opt_results['Base Architecture'] = opt_results['Base Architecture'].replace('transformer', 'Swin')
   opt_results['Temporal Aggregation'] = opt_results['Temporal Aggregation'].astype('str')
   opt_results['Temporal Aggregation'] = opt_results['Temporal Aggregation'].replace('False', 'Single-stream')
   opt_results['Temporal Aggregation'] = opt_results['Temporal Aggregation'].replace('True', 'Multi-stream')
   
   
   
   sns.set_theme(font_scale=1.5)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(figsize=(12,5))
   sns.barplot(data = opt_results, x ='Base Architecture', y = 'Millions of Paramters', hue = 'Temporal Aggregation', ax = ax, edgecolor='k', errorbar = 'sd', gap=0.05, estimator='mean', legend = True)
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   #plt.setp(ax.get_legend().get_texts(), fontsize='16') 
   #plt.setp(ax.get_legend().get_title(), fontsize='18') 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.suptitle('Trainable Parameters - Temporal Aggregation (Optical Models)')
   ax.set_ylim([0,35])
   ax.set_ylabel('Trainable Parameters (Millions)')
   # plt.title(f'Average Training Time (10 Epochs) - CLOUD-FREE', pad=35, fontsize=24)
   # plt.xticks(ha='center')
   #plt.yticks(np.linspace(0.3, 1.0, 8))
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=16, fmt='%.1f', padding = 3, fontweight='bold')
   # for container in ax.containers:
   #    ax.bar_label(container, fontsize=10)
   plt.tight_layout()
   st.pyplot(fig)
   plt.savefig(f'figures/size-siamese-opt.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   
   sar_results = results[
      (results['name'] == 'SAR')
   ]
   sar_results['value'] = sar_results['value'] / 1000000
   
   
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
   
   sns.set_theme(font_scale=1.5)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(figsize=(12,5))
   sns.barplot(data = sar_results, x ='Base Architecture', y = 'Millions of Paramters', hue = 'Temporal Aggregation [SAR Dataset]', ax = ax, edgecolor='k', errorbar = 'sd', gap=0.05, estimator='mean', legend = True)
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax.get_legend().get_texts(), fontsize='12') 
   plt.setp(ax.get_legend().get_title(), fontsize='14') 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.suptitle('Trainable Parameters - Temporal Aggregation (SAR Models)')
   ax.set_ylim([0,35])
   ax.set_ylabel('Trainable Parameters (Millions)')
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
   plt.savefig(f'figures/size-siamese-sar.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   
def plot_prevmap_size_com(results):
   sns.set_theme(style="darkgrid")
   
   opt_results = results[
      (results['name'] == 'Optical')
   ]
   
   opt_results['value'] = opt_results['value'] / 1000000
   
   opt_results = opt_results.rename(columns={
      'base_architecture': 'Base Architecture',
      'prev_map': 'Previous Map',
      'siamese': 'Siamese',
      'value': 'Millions of Paramters',
   })
   
   opt_results['Base Architecture'] = opt_results['Base Architecture'].replace('resunet', 'ResUnet')
   opt_results['Base Architecture'] = opt_results['Base Architecture'].replace('transformer', 'Swin')
   
   opt_results.loc[opt_results['Siamese'] == True, 'Base Architecture'] = opt_results.loc[opt_results['Siamese'] ==True, 'Base Architecture'] + ' (Multi-stream)'
   opt_results.loc[opt_results['Siamese'] == False, 'Base Architecture'] = opt_results.loc[opt_results['Siamese'] ==False, 'Base Architecture'] + ' (Single-stream)'

   
   sns.set_theme(font_scale=1.4)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(figsize=(10,6))
   sns.barplot(data = opt_results, x ='Base Architecture', y = 'Millions of Paramters', hue = 'Previous Map', ax = ax, edgecolor='k', errorbar = 'sd', gap=0.05, estimator='mean', legend = True)
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax.get_legend().get_texts(), fontsize='12') 
   plt.setp(ax.get_legend().get_title(), fontsize='14') 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.suptitle('Trainable Parameters - Previous Deforestation Map (Optical Models)')
   ax.set_ylim([0,35])
   ax.set_ylabel('Trainable Parameters (Millions)')
   
   plt.xticks(ha='left', rotation = 350)
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=14, fmt='%.1f', padding = 3, fontweight='bold', rotation=0)
   plt.tight_layout()
   st.pyplot(fig)
   plt.savefig(f'figures/size-prevmap-opt.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   
   
   
   sar_results = results[
      (results['name'] == 'SAR') 
   ]
   
   sar_results['value'] = sar_results['value'] / 1000000
   
   sar_results = sar_results.rename(columns={
      'base_architecture': 'Base Architecture',
      'sar_condition': 'SAR Data',
      'prev_map': 'Previous Map',
      'siamese': 'Siamese',
      'value': 'Millions of Paramters',
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
   fig, ax = plt.subplots(figsize=(10,6))
   sns.barplot(data = sar_results, x ='Base Architecture', y = 'Millions of Paramters', hue = 'Previous Map', ax = ax, edgecolor='k', errorbar = 'sd', gap=0.05, estimator='mean', legend = True)
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax.get_legend().get_texts(), fontsize='12') 
   plt.setp(ax.get_legend().get_title(), fontsize='14') 
   plt.setp(ax.get_xticklabels(), fontsize=14) 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.suptitle('Trainable Parameters - Previous Deforestation Map (SAR Models)')
   ax.set_ylim([0,35])
   ax.set_ylabel('Trainable Parameters (Millions)')
   
   plt.xticks(ha='left', rotation = 345)
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=11, fmt='%.1f', padding = 3, fontweight='bold', rotation=0)
   # for container in ax.containers:
   #    ax.bar_label(container, fontsize=10)
   plt.tight_layout()
   st.pyplot(fig)
   plt.savefig(f'figures/size-prevmap-sar.png', dpi=300, bbox_inches='tight')
   plt.close(fig)

def plot_fusion_size_com(results):
   sns.set_theme(style="darkgrid")
   
   opt_results = results.copy()
   
   opt_results['value'] = opt_results['value'] / 1000000
   
   
   opt_results = opt_results.rename(columns={
      'base_architecture': 'Base Architecture',
      'prev_map': 'Previous Map',
      'siamese': 'Siamese',
      'value': 'Millions of Paramters',
      'name': 'Model',
   })
   
   opt_results['Base Architecture'] = opt_results['Base Architecture'].replace('resunet', 'ResUnet')
   opt_results['Base Architecture'] = opt_results['Base Architecture'].replace('transformer', 'Swin')
   
   sns.set_theme(font_scale=1.5)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(figsize=(10,6))
   sns.barplot(data = opt_results, x ='Base Architecture', y = 'Millions of Paramters', hue = 'Model', ax = ax, edgecolor='k', errorbar = 'sd', gap=0.05, estimator='mean', legend = True)
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax.get_legend().get_texts(), fontsize='10') 
   plt.setp(ax.get_legend().get_title(), fontsize='12') 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.suptitle('Trainable Parameters - Fusion Models')
   ax.set_ylim([0,60])
   ax.set_ylabel('Trainable Parameters (Millions)')

   plt.xticks(ha='left')
   # plt.title(f'Average Training Time (10 Epochs) - CLOUD-FREE', pad=35, fontsize=24)
   # plt.xticks(ha='center')
   #plt.yticks(np.linspace(0.3, 1.0, 8))
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=13, fmt='%.1f', padding = 3, fontweight='bold', rotation=0)
   # for container in ax.containers:
   #    ax.bar_label(container, fontsize=10)
   plt.tight_layout()
   st.pyplot(fig)
   plt.savefig(f'figures/size-fusion.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   

size_results = None
for site_code in st.session_state['sites']:
   sns.set_theme(style="darkgrid")
   
   site_name = st.session_state['sites'][site_code]['name']
   st.header(site_name)
   
   exps = [101, 103, 106, 107, 156, 157, 151, 153, 204, 205, 206, 254, 255, 256]
   exp_codes = [f'exp_{code}' for code in exps]
   
   if size_results is None:
      size_results = get_site_size_results(site_name, st.session_state['experiments'], exp_codes)
   else:
      size_results = pd.concat([size_results, get_site_size_results(site_name, st.session_state['experiments'], exp_codes)])
   
plot_siamese_size_com(size_results)
   
size_results = None
for site_code in st.session_state['sites']:
   sns.set_theme(style="darkgrid")
   
   site_name = st.session_state['sites'][site_code]['name']
   
   st.header(site_name, divider='red')
   
   exps =        [101, 151, 104, 154, 201, 204, 251, 254]
   exps = exps + [103, 105, 202, 203, 205, 206, 153, 155, 252, 253, 255, 256]
   exp_codes = [f'exp_{code}' for code in exps]

   if size_results is None:
      size_results = get_site_size_results(site_name, st.session_state['experiments'], exp_codes)
   else:
      size_results = pd.concat([size_results, get_site_size_results(site_name, st.session_state['experiments'], exp_codes)])
   
plot_prevmap_size_com(size_results)

size_results = None
for site_code in st.session_state['sites']:
   sns.set_theme(style="darkgrid")
   
   site_name = st.session_state['sites'][site_code]['name']
   #results_site = get_site_results(site_name, st.session_state['experiments'])
   #st.table(results_site)
   
   st.header(site_name)
   
   #siamese comparison no cloud
   exps = [101, 301, 302, 303, 306, 103, 151, 351, 352, 353, 356, 153]
   exp_codes = [f'exp_{code}' for code in exps]
   
   if size_results is None:
      size_results = get_site_size_results(site_name, st.session_state['experiments'], exp_codes)
   else:
      size_results = pd.concat([size_results, get_site_size_results(site_name, st.session_state['experiments'], exp_codes)])
   
plot_fusion_size_com(size_results)