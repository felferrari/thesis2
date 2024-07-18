import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow import get_site_results
import numpy as np

#st.set_page_config(layout="wide")

def plot_prevmap_com(results, metric, metric_name, site_code):
   opt_results = results[
      (results['name'] == 'Optical') &
      (results['cond'] == 'global') 
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
   
   sns.set_theme(font_scale=2.2)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(figsize=(10,6))
   sns.barplot(data = opt_results, x ='Base Architecture', y = metric, edgecolor='k', hue = 'Previous Map', errorbar = None, gap=0.08, estimator='sum')
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.title(f'Previous Deforestation Map Comparison - CLOUD-FREE (Site {site_code[1]})', pad = 35, fontsize= 30)
   plt.setp(ax.get_xticklabels(), fontsize=20) 
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.ylim([0.3,1.05])
   plt.ylabel(metric_name)
   plt.xlabel('Base Model (Temporal Aggregation)')
   plt.yticks(np.linspace(0.3, 1.0, 8))
   #plt.xticks(ha='center')
   plt.xticks(ha='left', rotation = 345)
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=18, fmt='%.2f', fontweight='bold')
   st.pyplot(fig)
   plt.savefig(f'figures/prevmap-opt-{site_code}-{metric_name}.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   
   
   sar_results = results[
      (results['name'] == 'SAR') &
      (results['cond'] == 'global') 
   ]
   
   #f1_global_sar_results = f1_global_sar_results.sort_values(by=['siamese', 'sar_condition', 'exp_code'])
   
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
   
   sns.set_theme(font_scale=2.2)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(figsize=(10,6))
   #sns.barplot(data = sar_results, x ='Base Architecture', y = metric, edgecolor='k', hue = 'Previous Map', errorbar = None, gap=0.1)
   sns.barplot(data = sar_results, x ='Base Architecture', y = metric, edgecolor='k', hue = 'Previous Map', gap=0.1, estimator='sum')
   plt.title(f'Previous Deforestation Map Comparison - SAR datasets (Site {site_code[1]})', pad= 35, fontsize=25)
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.setp(ax.get_xticklabels(), fontsize=18) 
   #sns.move_legend(ax, "upper right", bbox_to_anchor=(1.0, 1.3))
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.ylabel(metric_name)
   plt.xlabel('Base Model (Temporal Aggregation) [SAR Dataset]')
   plt.ylim([0.3,1.05])
   plt.yticks(np.linspace(0.3, 1.0, 8))
   plt.xticks(ha='left', rotation = 345)
   for bars in ax.containers:
      ax.bar_label(bars, fontsize=14, fmt='%.2f', fontweight='bold')
   st.pyplot(fig)
   plt.savefig(f'figures/prevmap-sar-{site_code}-{metric_name}.png', dpi=300, bbox_inches='tight')
   plt.close(fig)
   
   # sar_results = results[
   #    (results['name'] == 'SAR') &
   #    (results['cond'] == 'global') 
   # ]
   
   # #f1_global_sar_results = f1_global_sar_results.sort_values(by=['siamese', 'sar_condition', 'exp_code'])
   
   # sar_results = sar_results.rename(columns={
   #    'base_architecture': 'Base Architecture',
   #    'sar_condition': 'SAR Data',
   #    'prev_map': 'Previous Map',
   #    'siamese': 'Siamese',
   # })
   
   # sar_results['Base Architecture'] = sar_results['Base Architecture'].replace('resunet', 'ResUnet')
   # sar_results['Base Architecture'] = sar_results['Base Architecture'].replace('transformer', 'Swin')
   # sar_results['SAR Data'] = sar_results['SAR Data'].replace('avg2', 'AVERAGE-2')
   # sar_results['SAR Data'] = sar_results['SAR Data'].replace('single2', 'SINGLE-2')
   # sar_results['SAR Data'] = sar_results['SAR Data'].replace('combined', 'AVERAGE-12')

   # sns.set_theme(font_scale=1)
   # sns.set_palette('tab10')
   # fig, ax = plt.subplots( 2, 3, figsize=(10,6))
   # data_temp = sar_results[sar_results['Base Architecture'] == 'ResUnet']
   
   # sns.barplot(data = data_temp[(data_temp['Siamese']==False) & (data_temp['SAR Data']=='AVERAGE-12')], ax = ax[0,0] , x ='Base Architecture', y = metric, edgecolor='k', hue = 'Previous Map', gap=0.1, estimator='sum', legend = False)
   # sns.barplot(data = data_temp[(data_temp['Siamese']==True) & (data_temp['SAR Data']=='AVERAGE-2')], ax = ax[0,1] , x ='Base Architecture', y = metric, edgecolor='k', hue = 'Previous Map', gap=0.1, estimator='sum', legend = False)
   # sns.barplot(data = data_temp[(data_temp['Siamese']==True) & (data_temp['SAR Data']=='SINGLE-2')], ax = ax[0,2] , x ='Base Architecture', y = metric, edgecolor='k', hue = 'Previous Map', gap=0.1, estimator='sum')

   # data_temp = sar_results[sar_results['Base Architecture'] == 'Swin']
   # sns.barplot(data = data_temp[(data_temp['Siamese']==False) & (data_temp['SAR Data']=='AVERAGE-12')], ax = ax[1,0] , x ='Base Architecture', y = metric, edgecolor='k', hue = 'Previous Map', gap=0.1, estimator='sum', legend = False)
   # sns.barplot(data = data_temp[(data_temp['Siamese']==True) & (data_temp['SAR Data']=='AVERAGE-2')], ax = ax[1,1] , x ='Base Architecture', y = metric, edgecolor='k', hue = 'Previous Map', gap=0.1, estimator='sum', legend = False)
   # sns.barplot(data = data_temp[(data_temp['Siamese']==True) & (data_temp['SAR Data']=='SINGLE-2')], ax = ax[1,2] , x ='Base Architecture', y = metric, edgecolor='k', hue = 'Previous Map', gap=0.1, estimator='sum', legend = False)

   # sns.move_legend(ax[0,2], "upper left", bbox_to_anchor=(1, 1))
   
   # ax[0,0].set_title(f'Single-Stream [AVERAGE-12]')
   # ax[0,1].set_title(f'Multi-Stream [AVERAGE-2]')
   # ax[0,2].set_title(f'Multi-Stream [SINGLE-2]')
   # ax[1,0].set_title(f'Single-Stream [AVERAGE-12]')
   # ax[1,1].set_title(f'Multi-Stream [AVERAGE-2]')
   # ax[1,2].set_title(f'Multi-Stream [SINGLE-2]')
   
   # for ax_i in ax.flatten():
   #    ax_i.set_xlabel(None)
   #    ax_i.set_xticklabels([])
   #    ax_i.set_ylabel(None)
   #    ax_i.set_yticklabels([])
   #    ax_i.set_yticks(np.linspace(0.3, 1.0, 8))
   #    ax_i.set_ylim([0.3, 1.05])
   #    for bars in ax_i.containers:
   #       ax_i.bar_label(bars, fontsize=14, fmt='%.2f', fontweight='bold')
   
   # ax[0,0].set_ylabel(f'ResUnet \n {metric_name}')
   # ax[0,0].set_yticklabels(np.round(np.linspace(0.3, 1.0, 8), 1))
   
   # ax[1,0].set_ylabel(f'Swin \n {metric_name}')
   # ax[1,0].set_yticklabels(np.round(np.linspace(0.3, 1.0, 8), 1))

   # plt.tight_layout()
   
   # st.pyplot(fig)
   # plt.savefig(f'figures/prevmap-sar-{site_code}-{metric_name}.png', dpi=300, bbox_inches='tight')
   # plt.close(fig)
   
for site_code in st.session_state['sites']:
   sns.set_theme(style="darkgrid")
   
   site_name = st.session_state['sites'][site_code]['name']
   #results_site = get_site_results(site_name, st.session_state['experiments'])
   #st.table(results_site)
   
   st.header(site_name, divider='red')
   
   #siamese comparison no cloud
   #exps = [101, 151, 103, 153, 104, 154, 105, 155, 201, 202, 203, 204, 251, 252, 253, 254, 205, 206, 255, 256]
   exps =        [101, 151, 104, 154, 201, 204, 251, 254]
   exps = exps + [103, 105, 202, 203, 205, 206, 153, 155, 252, 253, 255, 256]
   exp_codes = [f'exp_{code}' for code in exps]
   
   results = get_site_results(site_name, st.session_state['experiments'], exp_codes)
   
   plot_prevmap_com(results, 'f1score', 'F1-Score', site_code)
   plot_prevmap_com(results, 'precision', 'Precision', site_code)
   plot_prevmap_com(results, 'recall', 'Recall', site_code)
   
   