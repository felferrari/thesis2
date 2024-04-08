import hydra
from pathlib import Path
from src.utils.ops import load_sb_image, load_opt_image, load_SAR_image
from src.utils.generate import generate_images_statistics, generate_tiles, generate_labels, generate_prev_map
import h5py
import pandas as pd


        
@hydra.main(version_base=None, config_path='conf', config_name='config.yaml')
def prepare(cfg):
    prepared_path = Path(cfg.path.prepared)
    
    prepared_path.mkdir(exist_ok=True)
    
    if cfg.preparation.calculate.statistics:
        opt_stats = generate_images_statistics(cfg.site.original_data.opt.train, cfg.path.opt, load_opt_image, cfg.preparation.significance)
        sar_stats = generate_images_statistics(cfg.site.original_data.sar.train, cfg.path.sar, load_SAR_image, cfg.preparation.significance)
        
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
        
    if cfg.preparation.generate.tiles:
        print('Generating tiles...')
        generate_tiles(cfg)
    
    if cfg.preparation.generate.labels:
        generate_labels(cfg)
    
    if cfg.preparation.generate.prev_map:
        generate_prev_map(cfg)
    
    train_label = load_sb_image(cfg.path.label.train)


if __name__ == "__main__":
    prepare()