import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow import get_site_results




for site_code in st.session_state['sites']:
   sns.set_theme(style="darkgrid")
   
   site_name = st.session_state['sites'][site_code]['name']
   #results_site = get_site_results(site_name, st.session_state['experiments'])
   #st.table(results_site)
   
   st.header(site_name)
   
   #siamese comparison no cloud
   exps = [101, 151, 103, 153, 104, 154, 105, 155, 202, 203, 252, 253, 205, 206, 255, 256]
   exp_codes = [f'exp_{code}' for code in exps]
   
   results = get_site_results(site_name, st.session_state['experiments'], exp_codes)
   
   f1_results = results[results['metric'] == 'f1score']
   f1_global_opt_results = f1_results[
      (f1_results['type'] == 'Optical') &
      (f1_results['cond'] == 'global') 
   ]
   
   
   f1_global_opt_results = f1_global_opt_results.rename(columns={
      'value': 'F1-Score',
      'base_architecture': 'Base Architecture',
      'prev_map': 'Previous Map',
   })
   
   f1_global_opt_results['Base Architecture'] = f1_global_opt_results['Base Architecture'].replace('resunet', 'ResUnet')
   f1_global_opt_results['Base Architecture'] = f1_global_opt_results['Base Architecture'].replace('transformer', 'Swin')
   
   fig, ax = plt.subplots(figsize=(8,6))
   sns.barplot(data = f1_global_opt_results, x ='Base Architecture', y = 'F1-Score', hue = 'Previous Map', errorbar = None)
   plt.title(f'Previous Map Optical (Cloud-Free) Comparison (Site: {site_name})')
   #sns.move_legend(ax, "lower right")
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.ylim([0.6,1])
   plt.xticks(ha='center')
   st.pyplot(fig)
   plt.close(fig)
   
   
   f1_global_sar_results = f1_results[
      (f1_results['type'] == 'SAR') &
      (f1_results['cond'] == 'global') 
   ]
   
   #f1_global_sar_results = f1_global_sar_results.sort_values(by=['siamese', 'sar_condition', 'exp_code'])
   
   f1_global_sar_results = f1_global_sar_results.rename(columns={
      'value': 'F1-Score',
      'base_architecture': 'Base Architecture',
      'sar_condition': 'SAR Data',
      'prev_map': 'Previous Map',
      'siamese': 'Siamese',
   })
   
   f1_global_sar_results['Base Architecture'] = f1_global_sar_results['Base Architecture'].replace('resunet', 'ResUnet')
   f1_global_sar_results['Base Architecture'] = f1_global_sar_results['Base Architecture'].replace('transformer', 'Swin')
   f1_global_sar_results['SAR Data'] = f1_global_sar_results['SAR Data'].replace('avg2', '2 Average Images')
   f1_global_sar_results['SAR Data'] = f1_global_sar_results['SAR Data'].replace('single2', '2 Single Images')
   f1_global_sar_results['SAR Data'] = f1_global_sar_results['SAR Data'].replace('combined', '12 Average Images')
   
   #f1_global_sar_results.loc[f1_global_sar_results['Siamese'] == True, 'SAR Data'] = f1_global_sar_results.loc[f1_global_sar_results['Siamese'] ==True, 'SAR Data'] + ' (Siamese)'
   f1_global_sar_results.loc[f1_global_sar_results['Siamese'] == True, 'Base Architecture'] = f1_global_sar_results.loc[f1_global_sar_results['Siamese'] ==True, 'Base Architecture'] + ' (Siamese)'
   f1_global_sar_results['Base Architecture'] = f1_global_sar_results['Base Architecture'] + ' [' + f1_global_sar_results['SAR Data'] + ']'
   
   fig, ax = plt.subplots(figsize=(8,6))
   sns.barplot(data = f1_global_sar_results, x ='Base Architecture', y = 'F1-Score', hue = 'Previous Map', errorbar = None)
   plt.title(f'Previous Map SAR Comparison (Site: {site_name})')
   #sns.move_legend(ax, "upper right", bbox_to_anchor=(1.0, 1.3))
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.ylim([0,1.09])
   plt.xticks(ha='left', rotation = 325)
   st.pyplot(fig)
   plt.close(fig)