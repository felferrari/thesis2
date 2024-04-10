from typing import Any, Literal, Sequence
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

class PredictionCallback(BasePredictionWriter):
    def __init__(self, write_interval: Literal['batch'] | Literal['epoch'] | Literal['batch_and_epoch'] = "batch") -> None:
        super().__init__(write_interval)
        
    def write_on_batch_end(self, trainer: Trainer, pl_module: LightningModule, prediction: Any, batch_indices: Sequence[int] | None, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return super().write_on_batch_end(trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx)
    
    def write_on_epoch_end(self, trainer: Trainer, pl_module: LightningModule, predictions: Sequence[Any], batch_indices: Sequence[Any]) -> None:
        return super().write_on_epoch_end(trainer, pl_module, predictions, batch_indices)