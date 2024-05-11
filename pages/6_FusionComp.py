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
   exps = [101, 301, 302, 303, 103, 102, 311, 312, 313, 411, 412, 151, 351, 352, 353, 153, 152, 361, 362, 363, 461, 462]
   exp_codes = [f'exp_{code}' for code in exps]
   
   results = get_site_results(site_name, st.session_state['experiments'], exp_codes)
   
   f1_results = results[results['metric'] == 'f1score']
   f1_results = f1_results[f1_results['cond'] == 'global']
   f1_results_resunet = f1_results[f1_results['base_architecture'] == 'resunet']
   f1_results_resunet.loc[:, 'max'] = f1_results_resunet['value'].max()
   
   f1_results_resunet.loc[f1_results_resunet['exp_name'].str.contains('opt'), 'name'] = 'Optical'
   f1_results_resunet.loc[f1_results_resunet['exp_name'].str.contains('sar'), 'name'] = 'SAR'
   f1_results_resunet.loc[f1_results_resunet['exp_name'].str.contains('pixel_level'), 'name'] = 'Pixel Level'
   f1_results_resunet.loc[f1_results_resunet['exp_name'].str.contains('feature_middle'), 'name'] = 'Feature (Middle) Level'
   f1_results_resunet.loc[f1_results_resunet['exp_name'].str.contains('feature_late'), 'name'] = 'Feature (Late) Level'
   #f1_results_resunet.loc[f1_results_resunet['exp_name'].str.contains('pretrained'), 'name'] = f1_results_resunet.loc[f1_results_resunet['exp_name'].str.contains('pretrained'), 'name'] + ' [Pre-trained]'
   
   fig, ax = plt.subplots(figsize=(12,6))
   ax.axvspan(-0.5, 3.5, facecolor='g', alpha=0.25, label='Cloud-Free')
   ax.axvspan(3.5, 4.5, facecolor='r', alpha=0.25, label='No Optical')
   ax.axvspan(4.5, 8.5, facecolor='b', alpha=0.25, label='Diverse Cloud')
   ax.axvspan(8.5, 10.5, facecolor='y', alpha=0.25, label='Diverse Cloud (Pre-trained)')
   sns.barplot(data = f1_results_resunet, x ='exp_name', y = 'value', hue = 'name', units = 'opt_condition', edgecolor='k', errorbar = None, legend='full')
   sns.lineplot(data = f1_results_resunet, x ='exp_name', y = 'max', label = 'Maximum', legend='full')
   #g = sns.FacetGrid(f1_results, col="base_architecture")
   #g.map(sns.barplot, "exp_name", "value")
   plt.title(f'Results Comparison (Site: {site_name})')
   sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   plt.ylim([0,1])
   plt.gca().set_xticklabels([])
   plt.xticks(ha='left', rotation = 325)
   plt.xlabel('Model')
   plt.ylabel('F1-Score')
   st.pyplot(fig)
   plt.close(fig)
   