import streamlit as st
from pathlib import Path
from hydra import compose, initialize

conf_path = Path('conf')
exps_path =  conf_path / 'exp'
sites_path =  conf_path / 'site'

experiments = {}
for exp_file in  sorted(exps_path.glob('exp_*.*')):
    exp_code = exp_file.stem
    with initialize(version_base=None, config_path=str(conf_path), job_name="test_app"):
        cfg = compose(config_name="config", overrides=[f"+exp={exp_code}"])
        experiments[exp_code] = dict(cfg.exp)
        
sites = {}
for site_file in  sorted(sites_path.glob('s*.*')):
    site_code = site_file.stem
    with initialize(version_base=None, config_path=str(conf_path), job_name="test_app"):
        cfg = compose(config_name="config", overrides=[f"+site={site_code}"])
        sites[site_code] = dict(cfg.site)
        
st.session_state['experiments'] = experiments
st.session_state['sites'] = sites

st.set_page_config(
    page_title="Results",
    page_icon="👋",
)