import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow import get_uncertainty_proportions_data
import seaborn.objects  as so
import numpy as np

st.set_page_config(layout="wide")

def plot_simple_graphic(results, metric, metric_header, site_code):
   
   global_results = results.copy()
   global_results = global_results.groupby(['full_name', 'cond']).sum()['pixels']
   global_results = global_results.reset_index()
   
   


for site_code in st.session_state['sites']:
   sns.set_theme(style="darkgrid")
   
   site_name = st.session_state['sites'][site_code]['name']
   #results_site = get_site_results(site_name, st.session_state['experiments'])
   #st.table(results_site)
   
   st.header(site_name)
   
   #siamese comparison no cloud
   exps = [101, 301, 302, 303, 306, 103, 102, 311, 312, 313, 316, 411, 412, 413, 151, 351, 352, 353, 356, 153, 152, 361, 362, 363, 366, 461, 462, 463]
   exp_codes = [f'exp_{code}' for code in exps]
   
   results = get_uncertainty_proportions_data(site_name, st.session_state['experiments'], exp_codes)
   
   plot_simple_graphic(results, 'f1score', 'F1-Score', site_code)
   plot_simple_graphic(results, 'precision', 'Precision', site_code)
   plot_simple_graphic(results, 'recall', 'Recall', site_code)
   
   
   