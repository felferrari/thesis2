import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow import get_site_results




for site_code in st.session_state['sites']:
   site_name = st.session_state['sites'][site_code]['name']
   results_site = get_site_results(site_name, st.session_state['experiments'])
   #st.table(results_site)
   
   st.header(site_name)
   
   results_global_f1 = results_site[results_site['metric']== 'f1score']
   results_global_f1 = results_global_f1[results_global_f1['cond']== 'global']
   results_global_f1 = results_global_f1.sort_values(by = 'exp_code')
   results_global_f1 = results_global_f1.drop(labels = ['Unnamed: 0', 'metric', 'cond'], axis=1)
   
   
   # Global No cloud optical values
   optical_no_cloud = results_global_f1[results_global_f1['opt_condition'] == 'no_cloud']
   optical_no_cloud = optical_no_cloud[optical_no_cloud['type'] == 'Optical']
   optical_no_cloud = optical_no_cloud[optical_no_cloud['prev_map'] == True]
   
   sns.set_theme(style="darkgrid")
   
   
   fig, ax = plt.subplots(figsize=(8,6))
   sns.barplot(data = results_global_f1, x ='type', y = 'value', errorbar = None)
   plt.title(f'Site: {site_name}')
   #sns.move_legend(ax, "lower right")
   #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   #plt.ylim([0,1])
   plt.xticks(rotation=325, ha='left')
   st.pyplot(fig)
   plt.close(fig)