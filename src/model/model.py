from typing import Iterator
import torch.nn as nn
import torch
import json
import wandb
import warnings

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
        torch.save(self.model.state_dict(), f"{self.config.Model.Path}.pt")

    def save_onnx(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", 
                message="Exporting a model to ONNX with a batch_size other than 1"
            )
            
            batch_size = 1
            seq_length = self.config.DataLoader.Seq_length
            input_size = self.config.Model.N_letters
            
            dummy_input = torch.randint(0, input_size, (1, seq_length), 
                                       device=self.device)
            
            hidden = self.model.init_hidden(batch_size=batch_size, device=self.device)
            output_path = f"{self.config.Model.Path}.onnx"
            torch.onnx.export(
                self.model,
                (dummy_input, hidden),
                output_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input', 'hidden'],
                output_names=['output', 'hidden_out'],
                dynamic_axes={
                    'input': {0: 'batch_size', 1: 'sequence_length'},
                    'hidden': {1: 'batch_size'},
                    'output': {0: 'batch_size', 1: 'sequence_length'},
                    'hidden_out': {1: 'batch_size'}
                }
            )

    def save_char_lookup(self) -> None:
        char_to_idx = {char: i for i, char in enumerate(self.config.DataLoader.Letters)}
        idx_to_char = {i: char for i, char in enumerate(self.config.DataLoader.Letters)}
        vocab_file = f"{self.config.Model.Path}_char_encoding.json"
        with open(vocab_file, 'w') as f:
            json.dump({
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char
            }, f, indent=2)
        
    def load(self) -> None:
        self.model.load_state_dict(torch.load(self.config.Model.Path, map_location=self.device, weights_only=True))

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()