import hydra
from pathlib import Path
from src.utils.ops import load_ml_image, load_sb_image
import numpy as np
from tqdm import tqdm
import h5py
import pandas as pd
from shutil import rmtree

def generate_images_statistics(cfg_data, data_path, significance = 0):
    general_stats = dict()
    for k in cfg_data.condition:
        data_path = Path(data_path)
        stats = None
        pbar = tqdm({ii for i in cfg_data.condition[k] for ii in i }, desc=f'Generating Statistics for {k} condition')
        for img_i in pbar:
            img_file = data_path / cfg_data.imgs[img_i]
            data = load_ml_image(img_file)
            clip_values = np.percentile(data, (significance, 100-significance), axis=(0,1))
            data = np.clip(data, clip_values[0], clip_values[1])
            if stats is None:
                stats = {
                    'means': [data.mean(axis = (0,1))],
                    'stds': [data.std(axis = (0,1))],
                    'maxs': [data.max(axis = (0,1))],
                    'mins': [data.min(axis = (0,1))],
                }
            else:
                stats = {
                    'means': np.vstack((stats['means'], data.mean(axis=(0,1)))),
                    'stds': np.vstack((stats['stds'], data.std(axis=(0,1)))),
                    'maxs': np.vstack((stats['maxs'], data.max(axis=(0,1)))),
                    'mins': np.vstack((stats['mins'], data.min(axis=(0,1))))
                }
        if stats is not None:
            general_stats[k] = {
                'bands': np.arange(stats['means'].shape[1]),
                'means': stats['means'].mean(axis=0),
                'stds': stats['stds'].mean(axis=0),
                'maxs': stats['maxs'].max(axis=0),
                'mins': stats['mins'].min(axis=0)
            }
        
    return general_stats
        
@hydra.main(version_base=None, config_path='conf', config_name='config.yaml')
def prepare(cfg):
    prepared_path = Path(cfg.path.prepared)
    
    prepared_path.mkdir(exist_ok=True)
    
    if cfg.preparation.calculate.statistics:
        opt_stats = generate_images_statistics(cfg.site.original_data.opt.train, cfg.path.opt, cfg.preparation.significance)
        sar_stats = generate_images_statistics(cfg.site.original_data.sar.train, cfg.path.sar, cfg.preparation.significance)
        
        opt_df = None
        for cond in opt_stats:
            stats = opt_stats[cond]
            stats['cond'] = cond
            if opt_df is None:
                opt_df = pd.DataFrame(stats)
            else:
                opt_df = pd.concat((opt_df, pd.DataFrame(stats)))
                
        sar_df = None
        for cond in sar_stats:
            stats = sar_stats[cond]
            stats['cond'] = cond
            if sar_df is None:
                sar_df = pd.DataFrame(stats)
            else:
                sar_df = pd.concat((sar_df, pd.DataFrame(stats)))

        opt_df.to_csv(prepared_path / 'opt_stats.csv')
        sar_df.to_csv(prepared_path / 'sar_stats.csv')
        
    train_label = load_sb_image(cfg.path.label.train)


if __name__ == "__main__":
    prepare()