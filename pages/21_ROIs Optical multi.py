import streamlit as st
from src.utils.roi import resize_img
import numpy as np
from PIL import Image, ImageFont, ImageDraw 
from src.utils.mlflow import get_rois_images



st.set_page_config(layout="wide")

def plot_opt_rois(site_name, site_code, experiments, images, rois_codes, save, font_size, legend, output, reference):
   
   # font = ImageFont.truetype(<font-file>, <font-size>)
   font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', size = int(90*font_size))
   font2 = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', size = int(60*font_size))
   
   exps = list(images.keys())
   rois = list(images[exps[0]].keys())
   #combs = list(images[exps[0]][rois[0]].keys())
   for roi in rois:
      if not roi in rois_codes:
         continue
      st.header(roi, divider='blue')
      
      combs = list(images[exps[0]][roi].keys())
      for comb in combs:
         st.text(comb)
         ref_i = 0
         if reference is not None:
            ref_i += 1
         leg = 0
         if output == 'error':
            leg += 1
            
         full_image = Image.new('RGB', (int((2+ref_i+leg+len(exps))*1200),1200))
         
         img_name = f"opt_{site_name}-{experiments[exps[0]]['name']}-{comb.split('_')[1]}_roi_{roi.split('_')[1]}_opt_0"
         full_image.paste(resize_img(images[exps[0]][roi][comb][img_name]), (0*1200, 0*1200))
         
         img_name = f"opt_{site_name}-{experiments[exps[0]]['name']}-{comb.split('_')[1]}_roi_{roi.split('_')[1]}_opt_1"
         full_image.paste(resize_img(images[exps[0]][roi][comb][img_name]), (1*1200, 0*1200))
         
         draw = ImageDraw.Draw(full_image)
         draw.text((1200/2, 30), "Optical First", (0,0,0), font = font, anchor ='mt')
         draw.text((3*1200/2, 30), "Optical Last", (0,0,0), font = font, anchor ='mt')
         
         
         if reference is not None:
            img_name = f"{reference}_{site_name}-{experiments[exps[0]]['name']}-{comb.split('_')[1]}_roi_{roi.split('_')[1]}"
            full_image.paste(images[exps[0]][roi][comb][img_name], (2*1200, 0*1200))
            draw.text((5*1200/2, 30), "Reference", (0,0,0), font = font, anchor ='mt')
         
         i_last = None
         for i, exp in enumerate(exps):
            i_last = i
            # img_name = f"{output}_{site_name}-{experiments[exp]['name']}-{comb.split('_')[1]}_roi_{roi.split('_')[1]}"
            # full_image.paste(images[exp][roi][comb][img_name], ((3+i)*1200, 0*1200))
            if len(images[exp][roi]) == 1:
               img_name = f"{output}_{site_name}-{experiments[exp]['name']}-0_roi_{roi.split('_')[1]}"
               full_image.paste(images[exp][roi]['comb_0'][img_name], ((2+ref_i+i)*1200, 0*1200))
            else:
               img_name = f"{output}_{site_name}-{experiments[exp]['name']}-{comb.split('_')[1]}_roi_{roi.split('_')[1]}"
               full_image.paste(images[exp][roi][comb][img_name], ((2+ref_i+i)*1200, 0*1200))
            name = experiments[exp]['full_name'].split(' ')[legend]
            draw.text(((5+2*ref_i+2*i)*1200/2, 30), name, (0,0,0), font = font, anchor ='mt')
            
         if output == 'error':
            legend_img = Image.open('legend.png')
            legend_img = legend_img.resize((1200 ,1200), Image.LANCZOS)
            full_image.paste(legend_img, ((3+ref_i+i)*1200, 0*1200))
            # draw.rectangle(((3+ref_i+i)*1200, 0*1200, int((2+ref_i+leg+len(exps))*1200),1200), fill='white')
            # draw.text(((6+2*ref_i+2*i)*1200/2+ 250, 230), 'Discard', (0,0,0), font = font, anchor ='lm')
            # draw.text(((6+2*ref_i+2*i)*1200/2+ 250, 430), 'True Positive', (0,0,0), font = font, anchor ='lm')
            # draw.text(((6+2*ref_i+2*i)*1200/2+ 250, 630), 'True Negative', (0,0,0), font = font, anchor ='lm')
            # draw.text(((6+2*ref_i+2*i)*1200/2+ 250, 830), 'False Positive', (0,0,0), font = font, anchor ='lm')
            # draw.text(((6+2*ref_i+2*i)*1200/2+ 250, 1030), 'False Negative', (0,0,0), font = font, anchor ='lm')
         
         if save:
            full_image.save(f'figures/sample-opt-{output}-{exps}-{roi}-{comb}-{site_code}.png')
         st.image(full_image)

sites_names = [''] + [st.session_state['sites'][site]['name'] for site in st.session_state['sites']]

site_name = st.selectbox('Site:', sites_names)

if site_name != '':
   site_code = [site for site in st.session_state['sites'] if st.session_state['sites'][site]['name'] == site_name][0]

exps_l = [exp_k for exp_k in st.session_state['experiments']] # if st.session_state['experiments'][exp_k]['opt_condition'] != 'no_opt']
exp_codes = st.multiselect('Experiments:', exps_l)
#exp_codes = [exp_codes]

font_size = st.slider('Font Size:', 0.5, 5.0, 1.2, step=0.1)

legend = st.selectbox('Models Legend:', ['Base Model', 'Model'], 1)

legends = {
   'Base Model' : 0,
   'Model' : 1,
}

reference = st.selectbox('Reference:', ['Prediction', 'Labels', 'None'], 0)

references = {
   'Prediction' : 'ref',
   'Labels' : 'true',
   'None': None
}

output = st.selectbox('Output:', ['Prediction', 'Error Map', 'Entropy'], 0)

outputs = {
   'Prediction' : 'def_class',
   'Error Map' : 'error',
   'Entropy' : 'entropy',
}

save = st.checkbox('Save', False)

if site_name != '' and len(exp_codes) > 0:
   rois = [f'roi_{i}' for i in range(len(st.session_state['sites'][site_code]['rois']))]
   roi_codes = st.multiselect('ROIs:', rois)
   if len(roi_codes) >0 and len(exp_codes) > 0:
      images = get_rois_images(st.session_state['sites'][site_code], st.session_state['experiments'], exp_codes, roi_codes)
   
      plot_opt_rois(site_name, site_code, st.session_state['experiments'], images, roi_codes, save, font_size, legends[legend], outputs[output], references[reference])
        