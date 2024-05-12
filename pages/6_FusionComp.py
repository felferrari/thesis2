import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow import get_site_results
import seaborn.objects  as so

st.set_page_config(layout="wide")

def plot_graphic(results, metric, metric_header):
   
   f1_results = results[results['metric'] == metric]
   f1_results = f1_results[f1_results['cond'] == 'global']
   
   
   f1_results.loc[f1_results['exp_name'].str.contains('opt'), 'name'] = 'Optical'
   f1_results.loc[f1_results['exp_name'].str.contains('sar'), 'name'] = 'SAR'
   f1_results.loc[f1_results['exp_name'].str.contains('pixel_level'), 'name'] = 'Pixel Level'
   f1_results.loc[f1_results['exp_name'].str.contains('feature_middle'), 'name'] = 'Feature (Middle) Level'
   f1_results.loc[f1_results['exp_name'].str.contains('feature_late'), 'name'] = 'Feature (Late) Level'
   #f1_results.loc[f1_results['exp_name'].str.contains('pretrained'), 'name'] = f1_results.loc[f1_results['exp_name'].str.contains('pretrained'), 'name'] + ' [Pre-trained]'
   
   f1_results_resunet = f1_results[f1_results['base_architecture'] == 'resunet']
   f1_results_swin = f1_results[f1_results['base_architecture'] == 'transformer']
   
   f1_results_resunet.loc[:, 'max'] = f1_results_resunet['value'].max()
   f1_results_swin.loc[:, 'max'] = f1_results_swin['value'].max()
   
   sns.set_theme(font_scale=1.5)
   
   fig, ax = plt.subplots(1,2,figsize=(15,5))
   fig.tight_layout()
   alpha = 0.4
   
   ax[0].axvspan(-0.5, 3.5, facecolor='g', alpha=alpha) #, label='Cloud-Free')
   ax[0].axvspan(3.5, 4.5, facecolor='r', alpha=alpha) #, label='No Optical')
   ax[0].axvspan(4.5, 8.5, facecolor='b', alpha=alpha) #, label='Diverse Cloud')
   ax[0].axvspan(8.5, 10.5, facecolor='y', alpha=alpha) #, label='Diverse Cloud (Pre-trained)')
   
   ax[1].axvspan(-0.5, 3.5, facecolor='g', alpha=alpha, label='Cloud-Free')
   ax[1].axvspan(3.5, 4.5, facecolor='r', alpha=alpha, label='No Optical')
   ax[1].axvspan(4.5, 8.5, facecolor='b', alpha=alpha, label='Diverse Cloud')
   ax[1].axvspan(8.5, 10.5, facecolor='y', alpha=alpha, label='Diverse Cloud (Pre-trained)')
   
   sns.barplot(data = f1_results_resunet, x ='exp_name', y = 'value', hue = 'name', ax = ax[0], edgecolor='k', errorbar = None, legend=False)
   bar_p = sns.barplot(data = f1_results_swin, x ='exp_name', y = 'value', hue = 'name', ax = ax[1], edgecolor='k', errorbar = None, legend=False)
   sns.lineplot(data = f1_results_resunet, x ='exp_name', y = 'max', color= 'k', ax= ax[0], label = 'Best Result', legend=False)
   line_p = sns.lineplot(data = f1_results_swin, x ='exp_name', y = 'max', color= 'k', ax = ax[1], label = 'Best Result', legend=False)
   ax[1].legend(handles = [bar_p, line_p])
   #g = sns.FacetGrid(f1_results, col="base_architecture")
   #g.map(sns.barplot, "exp_name", "value")
   
   sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   ax[0].set_ylim([0,1])
   ax[1].set_ylim([0,1])
   
   ax[0].set_xticklabels([])
   ax[1].set_xticklabels([])
   ax[1].set_yticklabels([])
   plt.xticks(ha='left', rotation = 325)
   ax[0].set_xlabel('Model')
   ax[1].set_xlabel('Model')
   ax[0].set_ylabel(metric_header)
   ax[1].set_ylabel(None)
   ax[0].set_title(f'Base Architecture: ResUnet')
   ax[1].set_title(f'Base Architecture: Swin')
   st.pyplot(fig)
   plt.close(fig)
   
   

for site_code in st.session_state['sites']:
   sns.set_theme(style="darkgrid")
   
   site_name = st.session_state['sites'][site_code]['name']
   #results_site = get_site_results(site_name, st.session_state['experiments'])
   #st.table(results_site)
   
   st.header(site_name)
   
   #siamese comparison no cloud
   exps = [101, 301, 302, 303, 103, 102, 311, 312, 313, 411, 412, 151, 351, 352, 353, 153, 152, 361, 362, 363, 461] #, 462]
   exp_codes = [f'exp_{code}' for code in exps]
   
   results = get_site_results(site_name, st.session_state['experiments'], exp_codes)
   
   plot_graphic(results, 'f1score', 'F1-Score')
   
   