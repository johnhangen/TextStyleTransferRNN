import torch
from typing import Union

from config.configs import Config
from src.model.model import Model


def generate_text(config: Config, model: Union[Model, torch.nn.Module], start_char: int):
    model.eval()
    generated_text = [start_char]
    hidden = model.model.init_hidden(1, model.device)
    
    char2idx = {ch: idx for idx, ch in enumerate(config.DataLoader.Letters)}
    idx2char = {idx: ch for ch, idx in char2idx.items()}
    
    char_tensor = torch.tensor([[start_char]], dtype=torch.long, device=model.device)

    for _ in range(config.Training.length):
        with torch.no_grad():
            output, hidden = model.forward(char_tensor, hidden)

        scaled_logits = output.squeeze(0) / config.Training.temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        
        next_char = torch.multinomial(probs[-1], num_samples=1).item()

        generated_text.append(next_char)
        char_tensor = torch.tensor([[next_char]], dtype=torch.long, device=model.device)

    return "".join([idx2char[idx] for idx in generated_text])
