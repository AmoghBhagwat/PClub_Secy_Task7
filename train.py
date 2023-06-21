import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset

def train(dataset, model, batch_size, epochs, sequence_length):
    model.to("cuda")
    model.train()

    dataLoader = DataLoader(dataset, batch_size=batch_size)

    lossfn = nn.CrossEntropyLoss().to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        hidden, cell = model.init_state(sequence_length)
        cell = cell.to("cuda")
        hidden = hidden.to("cuda")

        for batch, (x, y) in enumerate(dataLoader):
            x = x.to("cuda")
            y = y.to("cuda")
            optimizer.zero_grad()

            y_pred, (hidden, cell) = model(x, (hidden, cell))
            loss = lossfn(y_pred.transpose(1, 2), y)

            hidden = hidden.detach()
            cell = cell.detach()

            loss.backward()
            optimizer.step()

dataset = Dataset(5)
model = Model(dataset).to("cuda")

train(dataset, model, 1024, 80, 5)
torch.save(model.state_dict(), 'model1.pt')
