import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow import get_site_results

st.set_page_config(layout="wide")

def plot_siamese_com(results, metric, metric_name, site_code):
   sns.set_theme(style="darkgrid")
   
   opt_results = results[
      (results['name'] == 'Optical') &
      (results['cond'] == 'global') 
   ]
   
   
   opt_results = opt_results.rename(columns={
      'base_architecture': 'Base Architecture',
      'siamese': 'Siamese',
   })
   
   opt_results['Base Architecture'] = opt_results['Base Architecture'].replace('resunet', 'ResUnet')
   opt_results['Base Architecture'] = opt_results['Base Architecture'].replace('transformer', 'Swin')
   
   sns.set_theme(font_scale=1.8)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(figsize=(8,6))
   sns.barplot(data = opt_results, x ='Base Architecture', y = metric, hue = 'Siamese', edgecolor='k', errorbar = None, gap=0.05)
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.ylim([0.3,1])
   plt.ylabel(metric_name)
   plt.title(f'Siamese Optical Networks - Site {site_code[1]} ({metric_name})', pad=40)
   plt.xticks(ha='center')
   # for container in ax.containers:
   #    ax.bar_label(container, fontsize=10)
   st.pyplot(fig)
   plt.close(fig)
   
   
   sar_results = results[
      (results['name'] == 'SAR') &
      (results['cond'] == 'global') 
   ]
   
   #sar_results = sar_results.sort_values(by=['siamese', 'sar_condition', 'exp_code'])
   
   sar_results = sar_results.rename(columns={
      'base_architecture': 'Base Architecture',
      'sar_condition': 'SAR Data',
      'siamese': 'Siamese',
   })
   
   sar_results['Base Architecture'] = sar_results['Base Architecture'].replace('resunet', 'ResUnet')
   sar_results['Base Architecture'] = sar_results['Base Architecture'].replace('transformer', 'Swin')
   sar_results['SAR Data'] = sar_results['SAR Data'].replace('avg2', '2 Average Images')
   sar_results['SAR Data'] = sar_results['SAR Data'].replace('single2', '2 Single Images')
   sar_results['SAR Data'] = sar_results['SAR Data'].replace('combined', '12 Average Images')
   
   sar_results.loc[sar_results['Siamese'] == True, 'SAR Data'] = sar_results.loc[sar_results['Siamese'] ==True, 'SAR Data'] + ' (Siamese)'
   
   sns.set_theme(font_scale=1.8)
   sns.set_palette('tab10')
   fig, ax = plt.subplots(figsize=(8,6))
   fig.tight_layout()
   sns.barplot(data = sar_results, x ='Base Architecture', y = metric, hue = 'SAR Data', edgecolor='k', errorbar = None, gap=0.1)
   plt.title(f'Siamese SAR Comparison - Site {site_code[1]}', pad = 40)
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   #sns.move_legend(ax, "upper right", bbox_to_anchor=(1.0, 1.3))
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.ylim([0.3,1.])
   plt.ylabel(metric_name)
   plt.xticks(ha='center', rotation = 0)
   st.pyplot(fig)
   plt.close(fig)
   


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
   
   