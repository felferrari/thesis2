import streamlit as st
from src.utils.roi import resize_img
import numpy as np
from PIL import Image, ImageFont, ImageDraw 
from src.utils.mlflow import get_rois_images



st.set_page_config(layout="wide")

class CompositeImage():
   def __init__(self, base_size:int, n_rows, n_images, circle_params = None):
      n_cols = n_images // n_rows
      if n_images % n_rows > 0:
         n_cols += 1
      
      self.n_cols = n_cols
      self.n_rows = n_rows
      self.base_size = base_size
      self.circle_params = circle_params
      
      self.image = Image.new('RGB', (base_size*n_cols, base_size*n_rows), color = (255,255,255))
      self.draw = ImageDraw.Draw(self.image)
      
   def pasteImage(self, index, img, data_type:str, legend = '', font = None):
      x = index % self.n_cols
      y = index // self.n_cols
         
      self.image.paste(resize_img(img), (x*self.base_size, y*self.base_size))
      
      if len(legend) > 0:
         self.draw.text((self.base_size * x + self.base_size / 2, self.base_size * y + 30), legend, (0,0,0), font = font, anchor ='mt')
         
      if self.circle_params is not None and data_type != 'legend':
         center = (int(x * self.base_size + self.circle_params[0] * self.base_size / 100), int(y * self.base_size + self.circle_params[1] * self.base_size / 100))
         radius = int(self.base_size * self.circle_params[2] / 100)
         if data_type == 'map':
            center = (center[0]-30, center[1])
            radius = int(radius*0.9)
         pos = ((center[0] - radius, center[1] - radius), (center[0] + radius, center[1] + radius))
         self.draw.ellipse(pos, outline= (243, 152, 55), width = 10)
         
      

def plot_opt_rois(site_name, site_code, experiments, images, rois_codes, save, font_size, legend, output, reference, n_lines, filler, circle_params = None):
   
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
         
         n_images =  len(exps) + 2 + filler
         if reference is not None: 
            n_images += 1
            if reference == 'true':
               n_images += 1
         if output == 'error': n_images += 1 
         prev_images = 2
         
         comb_images = CompositeImage(1200, n_lines, n_images, circle_params)
         
         img_name = f"opt_{site_name}-{experiments[exps[0]]['name']}-{comb.split('_')[1]}_roi_{roi.split('_')[1]}_opt_0"
         comb_images.pasteImage(0, resize_img(images[exps[0]][roi][comb][img_name]), 'image', 'First Image', font)
         
         img_name = f"opt_{site_name}-{experiments[exps[0]]['name']}-{comb.split('_')[1]}_roi_{roi.split('_')[1]}_opt_1"
         comb_images.pasteImage(1, resize_img(images[exps[0]][roi][comb][img_name]), 'image', 'Last Image', font)
         
         if reference is not None:
            prev_images += 1
            img_name = f"{reference}_{site_name}-{experiments[exps[0]]['name']}-{comb.split('_')[1]}_roi_{roi.split('_')[1]}"
            # comb_images.pasteImage(2, images[exps[0]][roi][comb][img_name], 'map', 'Reference', font)
            
            if reference == 'ref':
               comb_images.pasteImage(2, images[exps[0]][roi][comb][img_name], 'map', 'Reference', font)
            
            elif reference == 'true':
               comb_images.pasteImage(2, images[exps[0]][roi][comb][img_name], 'image', 'Labels', font)
               prev_images += 1
               legend_img = Image.open('reference_legend.png')
               legend_img = legend_img.resize((1200 ,1200), Image.LANCZOS)
               comb_images.pasteImage(3, legend_img, 'legend')
         
         for i, exp in enumerate(exps):
            name = experiments[exp]['full_name'].split(' ')[legend]
            if output == 'error':
               data_type = 'image'
            else:
               data_type = 'map'
            if len(images[exp][roi]) == 1:
               img_name = f"{output}_{site_name}-{experiments[exp]['name']}-0_roi_{roi.split('_')[1]}"
               # full_image.paste(images[exp][roi]['comb_0'][img_name], ((2+ref_i+i)*1200, 0*1200))
               comb_images.pasteImage(i+prev_images, images[exp][roi]['comb_0'][img_name], data_type, name, font)
            else:
               img_name = f"{output}_{site_name}-{experiments[exp]['name']}-{comb.split('_')[1]}_roi_{roi.split('_')[1]}"
               # full_image.paste(images[exp][roi][comb][img_name], ((2+ref_i+i)*1200, 0*1200))
               comb_images.pasteImage(i+prev_images, images[exp][roi][comb][img_name], data_type, name, font)
               
         if output == 'error':
            legend_img = Image.open('error_map_legend.png')
            legend_img = legend_img.resize((1200 ,1200), Image.LANCZOS)
            comb_images.pasteImage(prev_images + len(exps), legend_img, 'legend')
         
         st.image(comb_images.image)
         if save:
            comb_images.image.save(f'figures/sample-opt-{output}-{exps}-{roi}-{comb}-{site_code}.png')


sites_names = [''] + [st.session_state['sites'][site]['name'] for site in st.session_state['sites']]

site_name = st.sidebar.selectbox('Site:', sites_names)

if site_name != '':
   site_code = [site for site in st.session_state['sites'] if st.session_state['sites'][site]['name'] == site_name][0]

exps_l = [exp_k for exp_k in st.session_state['experiments']] # if st.session_state['experiments'][exp_k]['opt_condition'] != 'no_opt']
exp_codes = st.sidebar.multiselect('Experiments:', exps_l)
#exp_codes = [exp_codes]

font_size = st.sidebar.slider('Font Size:', 0.5, 5.0, 1.2, step=0.1)

n_lines = int(st.sidebar.slider('Number of lines:', 1, 5, 1, step=1))

filler = int(st.sidebar.slider('Number of empty images in the end:', 0, 10, 0, step=1))

legend = st.sidebar.selectbox('Models Legend:', ['Base Model', 'Model'], 1)

legends = {
   'Base Model' : 0,
   'Model' : 1,
}

reference = st.sidebar.selectbox('Reference:', ['Reference', 'Labels', 'None'], 0)

references = {
   'Reference' : 'ref',
   'Labels' : 'true',
   'None': None
}

output = st.sidebar.selectbox('Output:', ['Prediction', 'Error Map', 'Entropy'], 0)

outputs = {
   'Prediction' : 'def_class',
   'Error Map' : 'error',
   'Entropy' : 'entropy',
}

save = st.sidebar.checkbox('Save', False)

circle = st.sidebar.checkbox('Circle', False)

if circle:

   circle_x = int(st.sidebar.slider('X (%):', 0, 100, 50, step=1))
   circle_y = int(st.sidebar.slider('Y (%):', 0, 100, 50, step=1))
   radius = int(st.sidebar.slider('Radius (%):', 0, 50, 10, step=1))

if site_name != '' and len(exp_codes) > 0:
   rois = [f'roi_{i}' for i in range(len(st.session_state['sites'][site_code]['rois']))]
   roi_codes = st.sidebar.multiselect('ROIs:', rois)
   if len(roi_codes) >0 and len(exp_codes) > 0:
      images = get_rois_images(st.session_state['sites'][site_code], st.session_state['experiments'], exp_codes, roi_codes)
   
      if not circle:
         plot_opt_rois(site_name, site_code, st.session_state['experiments'], images, roi_codes, save, font_size, legends[legend], outputs[output], references[reference], n_lines, filler)
      else:
         plot_opt_rois(site_name, site_code, st.session_state['experiments'], images, roi_codes, save, font_size, legends[legend], outputs[output], references[reference], n_lines, filler, (circle_x, circle_y, radius))
        