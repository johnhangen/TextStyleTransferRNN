import unicodedata
import string
import polars as pl
import torch
from sklearn.model_selection import train_test_split

from config.configs import Config

class DataLoader():

    def __init__(self, config: Config) -> None:
        self.config = config
        self.all_letters = string.ascii_letters + " .,;'-"
        self.n_letters = len(self.all_letters)

        self.df = self.ReadInFile()
        self.train_idx, self.val_idx = train_test_split(list(range(self.df.shape[0])), test_size=self.config.DataLoader.Val_split)

    def unicodeToAscii(self, line) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFD', line)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )  
    
    def lineToTensor(self, line) -> torch.tensor:
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][self.all_letters.find(letter)] = 1
        return tensor

    def ReadInFile(self) -> pl.DataFrame:
        rows = []
        with open(self.config.DataLoader.Path, encoding='utf-8') as file:
            for index, line in enumerate(file):
                line_converted = self.unicodeToAscii(line.strip())
                line_tensor = self.lineToTensor(line.strip())
                
                rows.append([index, line_converted, line_tensor])

        df = pl.DataFrame(
            rows, schema={
                "index": pl.Int64, 
                "line_text": pl.String, 
                "line_tensor":pl.Object
                }, orient="row")
        
        return df

    def __len__(self) -> int:
        return self.df.shape[0]