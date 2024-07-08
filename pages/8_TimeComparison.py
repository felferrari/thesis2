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
   fig, ax = plt.subplots(1, 2, figsize=(12,5))
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
   
def plot_prevmap_time_com(results):
   sns.set_theme(style="darkgrid")

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