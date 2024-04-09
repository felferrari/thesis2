import hydra
from src.data.data_module import DataModule
from src.models.model_module import ModelModule
from lightning.pytorch.trainer.trainer import Trainer
        
@hydra.main(version_base=None, config_path='conf', config_name='config.yaml')
def train(cfg):
    data_module = DataModule(cfg)
    model_module = ModelModule(cfg)
    
    trainer = Trainer(
        accelerator='gpu',
        max_epochs = 10
    )
    
    trainer.fit(
        model=model_module,
        datamodule=data_module
    )
    
    
    
if __name__ == "__main__":
    train()