import streamlit as st
from src.utils.roi import resize_img
import numpy as np
from PIL import Image, ImageFont, ImageDraw 
from src.utils.mlflow import get_rois_images



#st.set_page_config(layout="wide")

def plot_sar_rois(site_name, experiments, images, rois_codes, save, font_size):
   
   # font = ImageFont.truetype(<font-file>, <font-size>)
   font1 = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', size = int(90*font_size))
   font2 = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', size = int(60*font_size))
   
   exps = list(images.keys())
   rois = list(images[exps[0]].keys())
   #combs = list(images[exps[0]][rois[0]].keys())
   for roi in rois:
      if not roi in rois_codes:
         continue
      st.header(roi, divider='blue')
      for exp in exps:  
         st.subheader(exp, divider='red')
         combs = list(images[exp][roi].keys())
         for comb in combs:
            st.text(comb)

            full_image = Image.new('RGB', (3*1200,1*1200))
            
            draw = ImageDraw.Draw(full_image)
            
            for i in range(2):
               for j in range(6):
                  img_name = f"sar_{site_name}-{experiments[exp]['name']}-{comb.split('_')[1]}_roi_{roi.split('_')[1]}_sar_{3*i+j}"
                  full_image.paste(resize_img(images[exp][roi][comb][img_name]).resize((650 ,650), Image.LANCZOS), (j*600, i*600))
                  draw.text((350+j*600, (i*600)+15), f"{6*i+j}", (0,0,0), font = font2, anchor ='mt')
               
            if save:
               full_image.save(f'figures/sample-sar-{exp}-{roi}-{comb}.png')
            
            st.image(full_image)

sites_names = [''] + [st.session_state['sites'][site]['name'] for site in st.session_state['sites']]

site_name = st.selectbox('Site:', sites_names)

if site_name != '':
   site_code = [site for site in st.session_state['sites'] if st.session_state['sites'][site]['name'] == site_name][0]

exps_l = [exp_k for exp_k in st.session_state['experiments'] if st.session_state['experiments'][exp_k]['sar_condition'] == 'combined']
exp_codes = st.selectbox('Experiments:', exps_l)
exp_codes = [exp_codes]

font_size = st.slider('Font Size:', 0.5, 5.0, 1.2, step=0.1)

save = st.checkbox('Save', False)

# if site_name != '' and len(exp_codes) > 0:
#    images = get_rois_images(st.session_state['sites'][site_code], st.session_state['experiments'], exp_codes)
   
#    exps = list(images.keys())
#    rois = list(images[exps[0]].keys())
   
#    rois_codes = st.multiselect('ROIs:', rois)
   
#    if len(rois_codes)>0:
   
#       plot_sar_rois(site_name, st.session_state['experiments'], images, rois_codes, title, save, font_size)
        
if site_name != '' and len(exp_codes) > 0:
   rois = [f'roi_{i}' for i in range(len(st.session_state['sites'][site_code]['rois']))]
   roi_codes = st.multiselect('ROIs:', rois)
   if len(roi_codes) >0 and len(exp_codes) > 0:
      images = get_rois_images(st.session_state['sites'][site_code], st.session_state['experiments'], exp_codes, roi_codes)
   
      plot_sar_rois(site_name, st.session_state['experiments'], images, roi_codes, save, font_size)
      