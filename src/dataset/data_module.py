
from abc import abstractmethod
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import h5py
from itertools import product
from einops import rearrange
from tqdm import tqdm
from src.utils.ops import load_SAR_image, load_opt_image, remove_outliers, load_sb_image, possible_combinations
from skimage.util import view_as_windows, crop
from torchvision.tv_tensors import Image, Mask
from skimage.transform import rescale
import random
from torchvision.transforms import v2
import torch
from torchvision.transforms.functional import vflip, hflip
class DataModule(LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        ds = TrainDataset(
            cfg = self.cfg,
            mode = 'train'
            )
        
        return DataLoader(
            dataset = ds,
            batch_size=self.cfg.exp.train_params.batch_size,
            num_workers=self.cfg.exp.train_params.train_workers,
            shuffle=True,
            persistent_workers=True,
            
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        ds = TrainDataset(
            cfg = self.cfg,
            mode = 'validation'
            )
        
        return DataLoader(
            dataset = ds,
            batch_size=self.cfg.exp.train_params.batch_size,
            num_workers=self.cfg.exp.train_params.val_workers,
            shuffle=False,
            persistent_workers=True,
        )
        
    
class TrainDataset(Dataset):
    def __init__(self, cfg, mode:str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mode == 'train':
            opt_files = list(Path(cfg.path.prepared.train).glob('opt_*.h5'))
            sar_files = list(Path(cfg.path.prepared.train).glob('sar_*.h5'))
            gen_files = list(Path(cfg.path.prepared.train).glob('gen_*.h5'))
            
            # opt_files.sort()
            # sar_files.sort()
            # gen_files.sort()
            
            # self.files = list(zip(opt_files, sar_files, gen_files))
        elif mode == 'validation':
            opt_files = list(Path(cfg.path.prepared.validation).glob('opt_*.h5'))
            sar_files = list(Path(cfg.path.prepared.validation).glob('sar_*.h5'))
            gen_files = list(Path(cfg.path.prepared.validation).glob('gen_*.h5'))
            
        opt_files.sort()
        sar_files.sort()
        gen_files.sort()
        
        self.files = list(zip(opt_files, sar_files, gen_files))
        random.shuffle(self.files)
        self.mode = mode
        opt_condition = cfg.exp.opt_condition
        sar_condition = cfg.exp.sar_condition
        
        opt_imgs_idx = cfg.site.original_data.opt.train.condition[opt_condition]
        sar_imgs_idx = cfg.site.original_data.sar.train.condition[sar_condition]
        
        #opt_combinations = list(product(opt_imgs_idx[0], opt_imgs_idx[1]))
        opt_combinations = possible_combinations(opt_imgs_idx)
        sar_combinations = possible_combinations(sar_imgs_idx)
        
        self.n_combinations = cfg.exp.train_params.repeat_batches * len(opt_combinations) * len(sar_combinations)
        self.combinations = cfg.exp.train_params.repeat_batches * list(product(opt_combinations, sar_combinations))
        
    def __len__(self):
        return len(self.files) * self.n_combinations
    
    def __getitem__(self, index):
        index_file = index // self.n_combinations
        comb_index = index % self.n_combinations
        opt_images_idx, sar_images_idx = self.combinations[comb_index]

        opt_patch_file, sar_patch_file, gen_patch_file = self.files[index_file]
        
        opt_data = h5py.File(opt_patch_file, 'r', rdcc_nbytes = 10*(1024**2))
        sar_data = h5py.File(sar_patch_file, 'r', rdcc_nbytes = 10*(1024**2))
        gen_data = h5py.File(gen_patch_file, 'r', rdcc_nbytes = 10*(1024**2))
    
        opt_patch = opt_data['opt'][()][opt_images_idx, :, :, :].astype(np.float32)
        sar_patch = sar_data['sar'][()][sar_images_idx, :, :, :].astype(np.float32)
        previous_patch = gen_data['previous'][()].astype(np.float32)
        label_patch = gen_data['label'][()].astype(np.int64)
        
        opt_patch = np.moveaxis(opt_patch, -1, -3)
        sar_patch = np.moveaxis(sar_patch, -1, -3)
        
        opt_patch = Image(opt_patch)
        sar_patch = Image(sar_patch)
        previous_patch = Image(previous_patch)
        label_patch = Mask(label_patch)
        
        if self.mode == 'train':
            if bool(random.getrandbits(1)):
                k = random.randint(0,3)
                opt_patch = torch.rot90(opt_patch, k=k, dims=(2, 3))
                sar_patch = torch.rot90(sar_patch, k=k, dims=(2, 3))
                previous_patch = torch.rot90(previous_patch, k=k, dims=(1, 2))
                label_patch = torch.rot90(label_patch, k=k, dims=(0, 1))
                
            elif bool(random.getrandbits(1)):
                opt_patch = hflip(opt_patch)
                sar_patch = hflip(sar_patch)
                previous_patch = hflip(previous_patch)
                label_patch = hflip(label_patch)
                
            elif bool(random.getrandbits(1)):
                opt_patch = vflip(opt_patch)
                sar_patch = vflip(sar_patch)
                previous_patch = vflip(previous_patch)
                label_patch = vflip(label_patch)
                
        return {
            'opt': opt_patch,
            'sar': sar_patch,
            'previous': previous_patch,
            #'cloud': torch.from_numpy(cloud_patch),
            }, label_patch
        
        
        
class PredDataset(Dataset):
    def __init__(self, cfg, img_combination, *args, **kwargs):
        super().__init__(*args, **kwargs)
        opt_imgs_idx, sar_imgs_idx = img_combination
        
        opt_img_files = [cfg.site.original_data.opt.test.imgs[img_idx] for img_idx in opt_imgs_idx]
        sar_img_files = [cfg.site.original_data.sar.test.imgs[img_idx] for img_idx in sar_imgs_idx]
        
        patch_size = cfg.general.patch_size
        pad_3d_width = (
            (patch_size, patch_size),
            (patch_size, patch_size),
            (0,0)
        )
        pad_2d_width = (
            (patch_size, patch_size),
            (patch_size, patch_size)
        )
        
        label = load_sb_image(cfg.path.label.test)
        #original_shape = label.shape
        
        label = np.pad(label, pad_2d_width)
        self.shape = label.shape
        label = label.flatten()
        
        previous = load_sb_image(cfg.path.prev_map.test)
        self.previous = np.pad(previous, pad_2d_width).flatten()

        self.opt_data = []
        for opt_img_file in tqdm(opt_img_files, desc='Reading Optical test data'):
            data_path = Path(cfg.path.opt) / opt_img_file
            data = load_opt_image(data_path).astype(np.float16)
            data = remove_outliers(data, cfg.general.outlier_significance)
            data = np.pad(data, pad_3d_width, mode = 'reflect')
            data = rearrange(data, 'h w c -> (h w) c')
            self.opt_data.append(data)
        
        self.sar_data = []
        for sar_img_file in tqdm(sar_img_files, desc='Reading SAR test data'):
            data_path = Path(cfg.path.sar) / sar_img_file
            data = load_SAR_image(data_path).astype(np.float16)
            data = remove_outliers(data, cfg.general.outlier_significance)
            data = np.pad(data, pad_3d_width, mode = 'reflect')
            data = rearrange(data, 'h w c -> (h w) c')
            self.sar_data.append(data)
            
        del data
        self.cfg = cfg
        
    #def set_discard_border(self, discard_border):
        discard_border = cfg.exp.pred_params.discard_border
        stride = self.cfg.general.patch_size - 2*discard_border
        indexes = np.arange(self.shape[0]* self.shape[1])
        indexes = rearrange(indexes, '(h w) -> h w', h = self.shape[0], w = self.shape[1])
        #indexes = view_as_windows(indexes, self.cfg.general.patch_size, int(self.cfg.general.patch_size*(1-overlap)))
        indexes = view_as_windows(indexes, self.cfg.general.patch_size, stride)
        self.indexes = rearrange(indexes, 'n m h w -> (n m) h w')
    
    def __len__(self):
        return self.indexes.shape[0]
    
    def __getitem__(self, index):
        patch_index = self.indexes[index]
        pack_patch = dict()
        opt_patch = []
        for data_i in self.opt_data:
            data_patch = data_i[patch_index]
            opt_patch.append(data_patch)
        if len(opt_patch) > 0:
            opt_patch = np.stack(opt_patch).astype(np.float32)
            opt_patch = np.moveaxis(opt_patch, -1, -3)
            pack_patch['opt'] = Image(opt_patch)
        
        sar_patch = []
        for data_i in self.sar_data:
            data_patch = data_i[patch_index]
            sar_patch.append(data_patch)
        if len(sar_patch) > 0:
            sar_patch = np.stack(sar_patch).astype(np.float32)
            sar_patch = np.moveaxis(sar_patch, -1, -3)
            pack_patch['sar'] = Image(sar_patch)
        
        previous_patch = self.previous[patch_index]
        previous_patch = np.expand_dims(previous_patch, axis=0).astype(np.float32)
        
        pack_patch['previous'] = Image(previous_patch)
        
        return pack_patch, patch_index

    @abstractmethod
    def test_combinations(cfg):
        opt_condition = cfg.exp.opt_condition
        sar_condition = cfg.exp.sar_condition
        
        opt_imgs_idx = cfg.site.original_data.opt.train.condition[opt_condition]
        sar_imgs_idx = cfg.site.original_data.sar.train.condition[sar_condition]
        
        opt_combinations = possible_combinations(opt_imgs_idx)
        sar_combinations = possible_combinations(sar_imgs_idx)

        return list(product(opt_combinations, sar_combinations))

    def generate_sample_images(self, img_combination):
        patch_size = self.cfg.general.patch_size
        crop_3d_width = (
            (patch_size, patch_size),
            (patch_size, patch_size),
            (0,0)
        )
        opt_comb, sar_comb = img_combination
        gen_opt_imgs = []
        gen_sar_imgs = []
        for opt_img in tqdm(self.opt_data, desc='Generating Optical Samples'):
            #file_path = path_dir / f'opt_{opt_i}.jpg'
            img = rearrange(opt_img[:, [2,1,0]], '(h w) c -> h w c', h = self.shape[0], w = self.shape[1])
            img = rescale(img, 0.1, channel_axis=2)
            img = crop(img, crop_3d_width)
            img = (10000 * img)
            img = 255 * img / 2000
            img = np.clip(img.astype(np.int32), 0, 255)
            #cv2.imwrite(str(file_path), img)
            gen_opt_imgs.append(img)
            
        for sar_img in tqdm(self.sar_data, desc='Generating SAR Samples'):
            #file_path_0 = path_dir / f'sar_{sar_i}_0.jpg'
            #file_path_1 = path_dir / f'sar_{sar_i}_1.jpg'
            img = rearrange(sar_img, '(h w) c -> h w c', h = self.shape[0], w = self.shape[1])
            img = rescale(img, 0.1, channel_axis=2)
            img = crop(img, crop_3d_width)
            img = 255 * img * np.array([3, 8])
            img = np.clip(img.astype(np.int32), 0, 255)
            #cv2.imwrite(str(file_path_0), img[:,:,0])
            #cv2.imwrite(str(file_path_1), img[:,:,1])
            gen_sar_imgs.append(img)
            
        return gen_opt_imgs, gen_sar_imgs