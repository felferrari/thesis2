import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow import get_site_results


st.set_page_config(layout="wide")

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
   
   opt_results.loc[opt_results['Siamese'] == True, 'Base Architecture'] = opt_results.loc[opt_results['Siamese'] ==True, 'Base Architecture'] + ' (Siamese)'
   
   sns.set_theme(font_scale=1)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(figsize=(8,6))
   sns.barplot(data = opt_results, x ='Base Architecture', y = metric, edgecolor='k', hue = 'Previous Map', errorbar = None, gap=0.08)
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   plt.title(f'Previous Map Optical (Cloud-Free) Comparison - Site {site_code[1]} ({metric_name})')
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.ylim([0,1])
   plt.ylabel(metric_name)
   plt.xlabel('Model')
   plt.xticks(ha='center')
   st.pyplot(fig)
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
   sar_results['SAR Data'] = sar_results['SAR Data'].replace('avg2', '2 Average')
   sar_results['SAR Data'] = sar_results['SAR Data'].replace('single2', '2 Single')
   sar_results['SAR Data'] = sar_results['SAR Data'].replace('combined', '12 Average')
   
   #f1_global_sar_results.loc[f1_global_sar_results['Siamese'] == True, 'SAR Data'] = f1_global_sar_results.loc[f1_global_sar_results['Siamese'] ==True, 'SAR Data'] + ' (Siamese)'
   sar_results.loc[sar_results['Siamese'] == True, 'Base Architecture'] = sar_results.loc[sar_results['Siamese'] ==True, 'Base Architecture'] + ' (Siamese)'
   sar_results['Base Architecture'] = sar_results['Base Architecture'] + ' [' + sar_results['SAR Data'] + ']'
   
   sns.set_theme(font_scale=1)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(figsize=(8,6))
   #sns.barplot(data = sar_results, x ='Base Architecture', y = metric, edgecolor='k', hue = 'Previous Map', errorbar = None, gap=0.1)
   sns.barplot(data = sar_results, x ='Base Architecture', y = metric, edgecolor='k', hue = 'Previous Map', gap=0.1)
   plt.title(f'Previous Map SAR Comparison - Site {site_code[1]} ({metric_name})')
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   #sns.move_legend(ax, "upper right", bbox_to_anchor=(1.0, 1.3))
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.ylabel(metric_name)
   plt.xlabel('Model')
   plt.ylim([0,1])
   plt.xticks(ha='left', rotation = 325)
   st.pyplot(fig)
   plt.close(fig)

for site_code in st.session_state['sites']:
   sns.set_theme(style="darkgrid")
   
   site_name = st.session_state['sites'][site_code]['name']
   #results_site = get_site_results(site_name, st.session_state['experiments'])
   #st.table(results_site)
   
   st.header(site_name, divider='red')
   
   #siamese comparison no cloud
   exps = [101, 151, 103, 153, 104, 154, 105, 155, 201, 202, 203, 204, 251, 252, 253, 254, 205, 206, 255, 256]
   exp_codes = [f'exp_{code}' for code in exps]
   
   results = get_site_results(site_name, st.session_state['experiments'], exp_codes)
   
   plot_prevmap_com(results, 'f1score', 'F1-Score', site_code)
   plot_prevmap_com(results, 'precision', 'Precision', site_code)
   plot_prevmap_com(results, 'recall', 'Recall', site_code)
   
   