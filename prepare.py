import hydra
from pathlib import Path
from src.utils.ops import load_sb_image, load_opt_image, load_SAR_image, save_geotiff
from src.utils.generate import generate_images_statistics, generate_tiles, generate_labels, generate_prev_map, read_imgs
import h5py
import pandas as pd
import numpy as np
from einops import rearrange
from skimage.util import view_as_windows
from tqdm import tqdm
from shutil import rmtree

        
@hydra.main(version_base=None, config_path='conf', config_name='config.yaml')
def prepare(cfg):
    prepared_path = Path(cfg.path.prepared.base)
    prepared_path.mkdir(exist_ok=True)
    
    if cfg.preparation.calculate.statistics:
        opt_df = generate_images_statistics(cfg.site.original_data.opt.train, cfg.path.opt, load_opt_image, cfg.general.outlier_significance)
        sar_df = generate_images_statistics(cfg.site.original_data.sar.train, cfg.path.sar, load_SAR_image, cfg.general.outlier_significance)
        
        opt_df.to_csv(cfg.path.prepared.opt_statistics)
        sar_df.to_csv(cfg.path.prepared.sar_statistics)
        
    if cfg.preparation.generate.tiles:
        print('Generating tiles...')
        generate_tiles(cfg)
    
    if cfg.preparation.generate.labels:
        generate_labels(cfg)
    
    if cfg.preparation.generate.prev_map:
        generate_prev_map(cfg)
    
    if cfg.preparation.generate.patches:
        np.random.seed(123)
        train_label = load_sb_image(cfg.path.label.train)
        tiles = load_sb_image(cfg.path.tiles)

        shape = train_label.shape
        indexes = view_as_windows(
            rearrange(np.arange(shape[0]*shape[1]), '(h w) -> h w', h = shape[0], w = shape[1]),
            cfg.general.patch_size,
            int((1-cfg.general.train_overlap)*cfg.general.patch_size)        
        )
        indexes = rearrange(indexes, 'n m h w -> (n m) h w')
        
        train_w_indexes = np.squeeze(np.argwhere(np.all(tiles.flatten()[indexes] == 1, axis=(1,2))))
        val_w_indexes = np.squeeze(np.argwhere(np.all(tiles.flatten()[indexes] == 0, axis=(1,2))))
        
        train_windows = indexes[train_w_indexes]
        val_windows = indexes[val_w_indexes]
        
        del indexes, train_w_indexes, val_w_indexes
        
        keep_train = ((train_label.flatten()[train_windows] == 1).sum(axis=(1,2)) / (cfg.general.patch_size**2)) >= cfg.general.min_def_prop
        keep_val = ((train_label.flatten()[val_windows] == 1).sum(axis=(1,2)) / (cfg.general.patch_size**2)) >= cfg.general.min_def_prop
        
        no_keep_train = np.logical_not(keep_train)
        no_keep_val = np.logical_not(keep_val)
        
        keep_train = np.squeeze(np.argwhere(keep_train), axis=-1)
        keep_val = np.squeeze(np.argwhere(keep_val), axis=-1)
        no_keep_train = np.squeeze(np.argwhere(no_keep_train), axis=-1)
        no_keep_val = np.squeeze(np.argwhere(no_keep_val), axis=-1)
        
        no_keep_train = np.random.choice(no_keep_train, len(keep_train))
        no_keep_val = np.random.choice(no_keep_val, len(keep_val))
        
        train_w_indexes = np.concatenate([keep_train, no_keep_train])
        val_w_indexes = np.concatenate([keep_val, no_keep_val])
        
        train_windows = train_windows[train_w_indexes]
        val_windows = val_windows[val_w_indexes]

        train_val_map = np.zeros_like(tiles)
        train_val_map = rearrange(train_val_map, 'h w -> (h w)')
        train_val_map[train_windows] = 2
        train_val_map[val_windows] = 1
        train_val_map = rearrange(train_val_map, '(h w) -> h w', h = shape[0], w = shape[1])
        
        base_image = Path(cfg.path.opt) / cfg.site.original_data.opt.train.imgs[0]
        save_geotiff(base_image, cfg.path.train_val_map, train_val_map, 'byte')
        
        del train_val_map, train_w_indexes, val_w_indexes
        
        opt_imgs = read_imgs(
            folder=cfg.path.opt, 
            imgs=cfg.site.original_data.opt.train.imgs, 
            read_fn=load_opt_image, 
            dtype=np.float16, 
            significance=cfg.general.outlier_significance, 
            factor=1.0
            )
        
        cloud_imgs = read_imgs(
            folder=cfg.path.opt, 
            imgs=cfg.site.original_data.opt.train.imgs, 
            read_fn=load_sb_image, 
            dtype=np.float16, 
            significance=cfg.general.outlier_significance, 
            factor=1.0/100,
            prefix_name='cloud_'
            )
        
        sar_imgs = read_imgs(
            folder=cfg.path.sar, 
            imgs=cfg.site.original_data.sar.train.imgs, 
            read_fn=load_SAR_image, 
            dtype=np.float16, 
            significance=cfg.general.outlier_significance, 
            factor=1.0
            )
        
        train_label = train_label.flatten().astype(np.uint8)
        previous_map = load_sb_image(cfg.path.prev_map.train).flatten().astype(np.float16)
        
        patch_size = cfg.general.patch_size
        
        pbar = tqdm(train_windows, desc='Generating Training Patches')
        train_prep_path = Path(cfg.path.prepared.train)
        if train_prep_path.exists():
            rmtree(train_prep_path)
        train_prep_path.mkdir()
        for i, patch in enumerate(pbar):
            patch_file = train_prep_path / f'{i}.h5'
            opt_patch = np.stack([full_img[patch] for full_img in opt_imgs], axis=0)
            sar_patch = np.stack([full_img[patch] for full_img in sar_imgs], axis=0)
            cloud_patch = np.expand_dims(np.stack([full_img[patch] for full_img in cloud_imgs], axis=0), axis= -1)
            prev_map_patch = np.expand_dims(previous_map[patch], axis = 0)
            label_patch = train_label[patch]
            
            with h5py.File(patch_file, "w") as f:
                f.create_dataset('opt', data=opt_patch, compression='lzf', chunks=opt_patch.shape)
                f.create_dataset('cloud', data=cloud_patch, compression='lzf', chunks=cloud_patch.shape)
                f.create_dataset('sar', data=sar_patch, compression='lzf', chunks=sar_patch.shape)
                f.create_dataset('previous', data=prev_map_patch, compression='lzf', chunks=prev_map_patch.shape)
                f.create_dataset('label', data=label_patch, compression='lzf', chunks=label_patch.shape)
                
        pbar = tqdm(val_windows, desc='Generating Validation Patches')
        val_prep_path = Path(cfg.path.prepared.validation)
        if val_prep_path.exists():
            rmtree(val_prep_path)
        val_prep_path.mkdir()
        for i, patch in enumerate(pbar):
            patch_file = val_prep_path / f'{i}.h5'
            opt_patch = np.stack([full_img[patch] for full_img in opt_imgs], axis=0)
            sar_patch = np.stack([full_img[patch] for full_img in sar_imgs], axis=0)
            cloud_patch = np.stack([full_img[patch] for full_img in cloud_imgs], axis=0)
            prev_map_patch = previous_map[patch]
            label_patch = train_label[patch]
            
            with h5py.File(patch_file, "w") as f:
                f.create_dataset('opt', data=opt_patch, compression='lzf', chunks=opt_patch.shape)
                f.create_dataset('cloud', data=cloud_patch, compression='lzf', chunks=cloud_patch.shape)
                f.create_dataset('sar', data=sar_patch, compression='lzf', chunks=sar_patch.shape)
                f.create_dataset('previous', data=prev_map_patch, compression='lzf', chunks=prev_map_patch.shape)
                f.create_dataset('label', data=label_patch, compression='lzf', chunks=label_patch.shape)
    
if __name__ == "__main__":
    prepare()