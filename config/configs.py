from dataclasses import dataclass
import yaml

@dataclass
class DataLoaderConfig:
  Path: str = r'data\Linux.txt'
  Letters = None
  N_letters: int = None
  Val_split: float = 0.05
  Seq_length: int = 100
  Batch_size: int = 1
  Shuffle:bool = True

@dataclass
class ModelConfig:
    model_type: str = "RNN"
    hidden_size: int = 256
    N_letters: int = None
    Path: str = 'model/Lunix.pt'
    Pretrain: bool = False

@dataclass
class TrainingConfig:
    epochs: int = 1
    length: int = 500
    temperature: float = 0.9

@dataclass
class OptimizerConfig:
    learning_rate: float = 0.1

@dataclass
class Config:
    DataLoader: DataLoaderConfig
    Model: ModelConfig
    Training: TrainingConfig
    Optimizer: OptimizerConfig

    @staticmethod
    def load_config(config_path: str) -> 'Config':
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        DataLoader_config = DataLoaderConfig(**config_dict["DataLoader"])
        Model_config = ModelConfig(**config_dict["Model"])
        Training_config = TrainingConfig(**config_dict["Train"])
        Optimizer_config = OptimizerConfig(**config_dict["Optimizer"])

        return Config(
            DataLoader=DataLoader_config,
            Model=Model_config,
            Training=Training_config,
            Optimizer=Optimizer_config
        )