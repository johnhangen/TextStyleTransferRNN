from dataclasses import dataclass
import yaml

@dataclass
class DataLoaderConfig:
  Path: str = r'data\tinyshakespeare.txt'
  Val_split: float = 0.05

@dataclass
class Config:
    DataLoader: DataLoaderConfig

    @staticmethod
    def load_config(config_path: str) -> 'Config':
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        DataLoader_config = DataLoaderConfig(**config_dict["DataLoader"])

        return Config(
            DataLoader=DataLoader_config,
        )