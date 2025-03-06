from typing import Iterator
import torch.nn as nn
import torch
import wandb

from config.configs import Config
from .RNN import RNN
from .GRU import GRU
from .LSTM import LSTM
    
class Model:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.config.Model.model_type == "RNN":
            self.model = RNN(
                input_size=self.config.Model.N_letters,
                hidden_size=self.config.Model.hidden_size,
                output_size=self.config.Model.N_letters
            ).to(self.device)
        elif self.config.Model.model_type == "GRU":
            self.model = GRU(
                input_size=self.config.Model.N_letters,
                hidden_size=self.config.Model.hidden_size,
                output_size=self.config.Model.N_letters
            ).to(self.device)
        elif self.config.Model.model_type == "LSTM":
            self.model = LSTM(
                input_size=self.config.Model.N_letters,
                hidden_size=self.config.Model.hidden_size,
                output_size=self.config.Model.N_letters
            ).to(self.device)
        
        wandb.watch(self.model, log_freq=100)

    def forward(self, char, hidden):
        return self.model.forward(char, hidden)

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.model.parameters()
    
    def save(self) -> None:
        torch.save(self.model.state_dict(), self.config.Model.Path)

    def load(self) -> None:
        self.model.load_state_dict(torch.load(self.config.Model.Path, map_location=self.device, weights_only=True))

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()