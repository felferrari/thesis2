from typing import Any, Literal, Sequence
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from src.utils.ops import load_sb_image
import numpy as np
from einops import rearrange
from skimage.util import crop

class PredictionCallback(BasePredictionWriter):
    def __init__(self, cfg) -> None:
        super().__init__(write_interval = 'batch')
        label = load_sb_image(cfg.path.label.test)
        
        patch_size = cfg.general.patch_size

        pad_width = (
            (patch_size, patch_size),
            (patch_size, patch_size)
        )
        self.shape = np.pad(label, pad_width=pad_width).shape
        
        self.patch_size = patch_size
        self.cfg = cfg
        self.reset_image()
        
        discard_border = cfg.exp.pred_params.discard_border
        
        self.crop_width_2d = (
            (discard_border, discard_border),
            (discard_border, discard_border)
        )
        
        self.crop_width_3d = (
            (discard_border, discard_border),
            (discard_border, discard_border),
            (0, 0)
        )
        
        ones_img = np.ones(shape=(patch_size, patch_size))
        self.ones_img = crop(ones_img, self.crop_width_2d)
        
        
        
    def reset_image(self):
        self.image_sum = np.zeros(shape = self.shape + (self.cfg.general.n_classes,), dtype=np.float16).reshape(-1, self.cfg.general.n_classes)
        self.image_count = np.zeros(shape = self.shape, dtype=np.int32).flatten()
        
    def write_on_batch_end(self, trainer: Trainer, pl_module: LightningModule, prediction: Any, batch_indices: Sequence[int] | None, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        _, patch_idx_b = batch
        patch_idx_b = patch_idx_b.detach().cpu().numpy()
        prediction = prediction.detach().cpu().numpy()
        prediction = rearrange(prediction, 'b c h w -> b h w c')
        
        for bi in range(patch_idx_b.shape[0]):
            patch_idx_i = patch_idx_b[bi]
            pred_i = prediction[bi]
            
            patch_idx_i = crop(patch_idx_i, crop_width=self.crop_width_2d)
            pred_i = crop(pred_i, crop_width=self.crop_width_3d)
            
            self.image_sum[patch_idx_i] = self.image_sum[patch_idx_i] + pred_i
            self.image_count[patch_idx_i] = self.image_count[patch_idx_i] + self.ones_img
            
            
    
    #def write_on_epoch_end(self, trainer: Trainer, pl_module: LightningModule, predictions: Sequence[Any], batch_indices: Sequence[Any]) -> None:
    def get_final_image(self):
        crop_width_3d = (
            (self.cfg.general.patch_size, self.cfg.general.patch_size),
            (self.cfg.general.patch_size, self.cfg.general.patch_size),
            (0, 0)
        )
        
        image_sum = rearrange(self.image_sum, '(h w) c -> h w c', h = self.shape[0], w = self.shape[1])
        image_count = rearrange(self.image_count, '(h w) -> h w 1', h = self.shape[0], w = self.shape[1])
        image_sum = crop(image_sum, crop_width_3d)
        image_count = crop(image_count, crop_width_3d)
        
        return image_sum / image_count
        
class AnalyzeCallback(PredictionCallback):
    def reset_image(self):
        self.image_sum = None
        self.image_count = None