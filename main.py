from torch.utils.data import DataLoader
import wandb
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from src.dataloader import TextDataset
from src.model.model import Model
from src.train import train
from config.configs import Config

def main():
    config = Config.load_config('config\configs.yaml')

    wandb.init(
        mode="disabled"
    )
    
    dataloaders = {
        "train": DataLoader(
                        TextDataset(config=config,
                                    train=True),
                        batch_size=config.DataLoader.Batch_size,
                        shuffle=config.DataLoader.Shuffle
                    ),
        "val": DataLoader(
                        TextDataset(config=config,
                                    train=False),
                        batch_size=config.DataLoader.Batch_size,
                        shuffle=config.DataLoader.Shuffle
                    )
    }

    model = Model(
        config=config
    )

    model = train(
                config=config,
                model=model,
                dataloaders=dataloaders
                )   

if __name__ == "__main__":
    main()