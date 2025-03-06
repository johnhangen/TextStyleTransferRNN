from typing import Iterator
import torch.nn as nn
import torch
import wandb
import json
import os

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
        
    def export_to_onnx(self, output_dir="assets/model", char_set=None):
        """
        Export the model to ONNX format for web deployment.
        
        Args:
            output_dir: Directory to save the ONNX model and vocabulary
            char_set: Optional character set to use for encoding. If None, uses the config's character set.
        
        Returns:
            tuple: Paths to the exported ONNX model and character encoding JSON file
        """
        # Check for required dependencies
        try:
            import onnx
        except ImportError:
            print("Error: ONNX package is not installed!")
            print("Please install required packages with: pip install onnx onnxruntime")
            return None, None
            
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set model to evaluation mode
        self.eval()
        
        # Get character set from config if not provided
        if char_set is None:
            if hasattr(self.config.DataLoader, 'Letters') and self.config.DataLoader.Letters is not None:
                char_set = self.config.DataLoader.Letters
            else:
                # Fallback to a default character set if not available in config
                char_set = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?-_\"'()[] \n"
                print(f"Warning: No character set found in config. Using default character set.")
        
        # Create character encoding
        char_to_idx = {char: i for i, char in enumerate(char_set)}
        idx_to_char = {i: char for i, char in enumerate(char_set)}
        
        # Save character encoding
        vocab_file = os.path.join(output_dir, "char_encoding.json")
        with open(vocab_file, 'w') as f:
            json.dump({
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char
            }, f, indent=2)
        
        print(f"Character encoding saved to {vocab_file}")
        
        # Get model dimensions
        hidden_size = self.model.hidden_size
        num_layers = self.model.num_layers
        
        print(f"Model configuration: hidden_size={hidden_size}, num_layers={num_layers}")
        
        # Create dummy inputs for ONNX export
        dummy_input = torch.randint(0, len(char_set), (1, 1), device=self.device)
        dummy_hidden = self.model.init_hidden(1, self.device)
        
        # Print shape information for debugging
        print(f"Input shape: {dummy_input.shape}")
        print(f"Hidden shape: {dummy_hidden.shape}")
        
        # Export to ONNX
        onnx_path = os.path.join(output_dir, "model.onnx")
        
        try:
            torch.onnx.export(
                self.model,                  # model being run
                (dummy_input, dummy_hidden), # model input
                onnx_path,                   # where to save
                export_params=True,          # store the trained parameter weights
                opset_version=12,            # the ONNX version
                do_constant_folding=True,    # optimization
                input_names=['input', 'hidden'],   # model's input names
                output_names=['output', 'next_hidden'],  # model's output names
                dynamic_axes={
                    'input': {0: 'batch_size', 1: 'sequence_length'},
                    'hidden': {1: 'batch_size'},
                    'output': {0: 'batch_size', 1: 'sequence_length'},
                    'next_hidden': {1: 'batch_size'}
                }
            )
            print(f"Model exported successfully to {onnx_path}")
            
            # Verify the model
            try:
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                print("ONNX model verified successfully")
                
                # Save model metadata for JavaScript
                model_info_path = os.path.join(output_dir, "model_info.json")
                with open(model_info_path, 'w') as f:
                    json.dump({
                        'hidden_size': hidden_size,
                        'num_layers': num_layers,
                        'vocab_size': len(char_set)
                    }, f, indent=2)
                print(f"Model info saved to {model_info_path}")
                
            except Exception as e:
                print(f"Warning: ONNX model verification failed: {e}")
                
            return onnx_path, vocab_file
        except Exception as e:
            print(f"Error exporting model: {e}")
            return None, vocab_file
