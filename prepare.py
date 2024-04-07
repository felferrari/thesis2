import hydra
from pathlib import Path
from src.utils.ops import load_ml_image
import numpy as np
from tqdm import tqdm

def generate_statistics(cfg_data, data_path, significance = 0):
    
    for k in cfg_data.condition:
        data_path = Path(data_path)
        stats = None
        pbar = tqdm({ii for i in cfg_data.condition[k] for ii in i }, desc=f'Generating Statistics for {k} condition')
        for img_i in pbar:
            img_file = data_path / cfg_data.imgs[img_i]
            data = load_ml_image(img_file)
            clip_values = np.percentile(data, (significance, 100-significance), axis=(0,1))            if stats is None:
                stats = {
                    'means': data.mean(axis = (0,1)),
                    'stds': data.std(axis = (0,1))
                }
            else:
                stats = {
                    'means': np.concatenate((stats['means'], data.mean(axis=(0,1)))),
                    'stds': np.concatenate((stats['stds'], data.std(axis=(0,1))))
                }
        print()
            
            

@hydra.main(version_base=None, config_path='conf', config_name='config.yaml')
def prepare(cfg):
    generate_statistics(cfg.site.original_data.opt.train, cfg.path.data.opt)


if __name__ == "__main__":
    prepare()