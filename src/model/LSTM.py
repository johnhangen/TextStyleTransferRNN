import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers: int = 1, dropout: float = 0.0) -> None:
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, char, hidden):
        char = self.embedding(char)
        output, hidden = self.rnn(char, hidden)
        output = self.linear(output)

        return output, hidden
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)