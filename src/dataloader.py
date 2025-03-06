from torch.utils.data import Dataset
import torch

from config.configs import Config

class TextDataset(Dataset):

    def __init__(self, config: Config, train: bool = True) -> None:
        self.config = config
        self.train = train

        with open(self.config.DataLoader.Path, encoding='utf-8') as file:
            self._text = file.read(10_000)
        
        self.Letters = sorted(set(self._text))
        self.N_letters = len(self.Letters)
        self.config.DataLoader.Letters = self.Letters
        self.config.DataLoader.N_letters = self.N_letters
        self.config.Model.N_letters = self.N_letters

        index = len(self._text) - int(len(self._text) * self.config.DataLoader.Val_split)

        if train:
            self._text = self._text[:index]
        else:
            self._text = self._text[index:]

        self.char2idx = {ch: idx for idx, ch in enumerate(self.Letters)}
        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}
    
    @property
    def text(self) -> None:
        return self.config.DataLoader.Path
    
    @text.setter
    def text(self, path:str) -> None:
        with open(path, encoding='utf-8') as file:
            self._text = file.read()
        
        index = len(self._text) - int(len(self._text) * self.config.DataLoader.Val_split)

        if self.train:
            self._text = self._text[:index]
        else:
            self._text = self._text[index:]

        self.char2idx = {ch: idx for idx, ch in enumerate(self.Letters)}
        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}

    def __len__(self):
        return len(self._text) - self.config.DataLoader.Seq_length
    
    def __getitem__(self, index):
        input_text = self._text[index:index + self.config.DataLoader.Seq_length]
        target_text = self._text[index + 1:index + self.config.DataLoader.Seq_length + 1]

        input_tensor = torch.tensor([self.char2idx[ch] for ch in input_text], dtype=torch.long)
        target_tensor = torch.tensor([self.char2idx[ch] for ch in target_text], dtype=torch.long)
        
        return input_tensor, target_tensor