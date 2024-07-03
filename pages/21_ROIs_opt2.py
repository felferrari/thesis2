import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.mlflow import get_rois_images
from src.utils.roi import resize_img
import seaborn.objects  as so
import numpy as np
from PIL import Image, ImageFont, ImageDraw 

st.set_page_config(layout="wide")

def plot_opt_rois(site_name, experiments, images):
   
   # font = ImageFont.truetype(<font-file>, <font-size>)
   font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', size = 70)  
   
   exps = list(images.keys())
   rois = list(images[exps[0]].keys())
   #combs = list(images[exps[0]][rois[0]].keys())
   for roi in rois:
      st.header(roi, divider='blue')
      for exp in exps:  
         st.subheader(exp, divider='red')
         combs = list(images[exp][roi].keys())
         for comb in combs:
            st.text(comb)

            full_image = Image.new('RGB', (4*800,800))
            
            img_name = f"opt_{site_name}-{experiments[exp]['name']}-{comb.split('_')[1]}_roi_{roi.split('_')[1]}_opt_0"
            full_image.paste(resize_img(images[exp][roi][comb][img_name]), (0*800, 0*800))
            
            img_name = f"opt_{site_name}-{experiments[exp]['name']}-{comb.split('_')[1]}_roi_{roi.split('_')[1]}_opt_1"
            full_image.paste(resize_img(images[exp][roi][comb][img_name]), (1*800, 0*800))
            
            img_name = f"ref_{site_name}-{experiments[exp]['name']}-{comb.split('_')[1]}_roi_{roi.split('_')[1]}"
            full_image.paste(images[exp][roi][comb][img_name], (2*800, 0*800))
            
            img_name = f"def_class_{site_name}-{experiments[exp]['name']}-{comb.split('_')[1]}_roi_{roi.split('_')[1]}"
            full_image.paste(images[exp][roi][comb][img_name], (3*800, 0*800))
            
            
            draw = ImageDraw.Draw(full_image)
            draw.text((800/2, 30), "Optical First", (0,0,0), font = font, anchor ='mt')
            draw.text((3*800/2, 30), "Optical Last", (0,0,0), font = font, anchor ='mt')
            draw.text((5*800/2, 30), "Reference", (0,0,0), font = font, anchor ='mt')
            draw.text((7*800/2, 30), "Prediction", (0,0,0), font = font, anchor ='mt')
            
            
            st.image(full_image)

for site_code in st.session_state['sites']:
   sns.set_theme(style="darkgrid")
   
   site_name = st.session_state['sites'][site_code]['name']
   #results_site = get_site_results(site_name, st.session_state['experiments'])
   #st.table(results_site)
   
   st.title(site_name)
   exp_codes =  [101, 102, 104, 201, 204, 301, 302, 303, 306, 311, 312, 313, 316, 401, 402, 403, 411, 412, 413]
   exp_codes += [151, 152, 154, 251, 254, 351, 352, 353, 356, 361, 362, 363, 366, 451, 452, 453, 461, 462, 463]
   
   #siamese comparison no cloud
   # for exp_code in exp_codes:
      
   images = get_rois_images(st.session_state['sites'][site_code], st.session_state['experiments'], exp_codes)
   plot_opt_rois(site_name, st.session_state['experiments'], images)
   
   
   