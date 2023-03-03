import torch
import torch.nn as nn
from torch.functional import F
from torch.optim import Adam
import dataclasses
from tqdm import tqdm
import torch.utils.data as data_utils

NUM_EPOCHS = 100
LR = 6e-4
device = torch.device("cuda")

class MLP(nn.Module):
    def __init__(self, in_channels= 100, out_channels=10, num_hidden=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels,num_hidden)
        self.fc2 = nn.Linear(num_hidden,out_channels)

    def forward(self, x):
        x=torch.flatten(x, start_dim=1) # Flatten layer
        x=F.relu(self.fc1(x)) 
        x=F.softmax(self.fc2(x))
        return x

def train_nn(model, enocder_model, train_dataset, test_dataset, batch_size=32, num_epochs=NUM_EPOCHS, lr=LR, criterion=nn.CrossEntropyLoss()):
    device = 'cpu'
    TrainResult = {'train_losses': [], 'val_accs': [], 'train_accs': []}
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(num_epochs):
        total_train = 0
        model.train()
        train_accs = 0
        for datapoint in tqdm(train_loader):
            X, y = datapoint[0].to(device), datapoint[1].to(device)
            z_mu, z_sigma = enocder_model.encode(X)
            z_sample = z_mu + torch.randn_like(z_mu)*z_sigma
            optimizer.zero_grad()
            output = model(z_sample[:, enocder_model.coord:].to(device))
            loss = criterion(output, y)
            total_train += loss.item()
            train_accs += torch.sum(torch.argmax(output, dim=-1)==y).item()/output.size(dim=0)
            loss.backward()
            optimizer.step()
        
        total_train = total_train/len(train_loader)
        TrainResult['train_losses'].append(total_train)
        TrainResult['train_accs'].append(train_accs/len(train_loader))
                               
        model.eval()
        total_val = 0
        with torch.no_grad():
            for idx, datapoint in enumerate(test_loader):
                X, y = datapoint[0].to(device), datapoint[1].to(device)
                z_mu, z_sigma = enocder_model.encode(X)
                z_sample = z_mu + torch.randn_like(z_mu)*z_sigma
                optimizer.zero_grad()
                output = model(z_sample[:, enocder_model.coord:].to(device))
                total_val += torch.sum(torch.argmax(output, dim=-1)==y).item()/output.size(dim=0)
        TrainResult['val_accs'].append(total_val/len(test_loader))
        print("Epoch {}: Train Loss={} Train Acc={} Validation Acc={}%".format(epoch, TrainResult['train_losses'][-1], TrainResult['train_accs'][-1]*100, TrainResult['val_accs'][-1]*100))
    return TrainResult
