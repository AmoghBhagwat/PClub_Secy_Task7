import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()

        unique = len(dataset.get_unique_words())

        self.lstm_size = 128
        self.num_layers = 1
        
        self.embedding = nn.Embedding(num_embeddings=unique, embedding_dim=128)
        self.lstm = nn.LSTM(input_size=self.num_layers, hidden_size=self.lstm_size, num_layers=1)
        self.hidden = nn.Linear(self.lstm_size, 1024)
        self.relu = nn.ReLU()
        self.final = nn.Linear(1024, unique)

    def forward(self, x, previous):
        embed = self.embedding(x)
        output, state = self.lstm(embed, previous)
        output = self.hidden(output)
        output = self.relu(output)
        output = self.final(output)

        return output, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))