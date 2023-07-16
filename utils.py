import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from collections.abc import Iterable
from transformer_lens import HookedTransformerConfig

class MNIST_CNN(nn.Module):  
    
    def __init__(self, p=0.2):
        super(MNIST_CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 16, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(400, 200)
        self.fc2 = nn.Linear(200, 10)
        self.drop = nn.Dropout(p=p)

    def forward(self, X):
        X = self.pool(self.drop(F.relu(self.conv1(X))))
        X = self.pool(self.drop(F.relu(self.conv2(X))))
        X = X.view(-1, 400)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return X

    def get_neurons(self, X):
        X = self.pool(self.drop(F.relu(self.conv1(X))))
        X = self.pool(self.drop(F.relu(self.conv2(X))))
        X = X.view(-1, 400)
        X = F.relu(self.fc1(X))
        return X
    
    def get_neurons_pre(self, X):
        X = self.pool(self.drop(F.relu(self.conv1(X))))
        X = self.pool(self.drop(F.relu(self.conv2(X))))
        X = X.view(-1, 400)
        X = self.fc1(X)
        return X

    def get_features(self, X):
        X = self.pool(self.drop(F.relu(self.conv1(X))))
        X = self.pool(self.drop(F.relu(self.conv2(X))))
        X = X.view(-1, 400)
        return X

    def get_conv1(self, X):
        X = self.pool(self.drop(F.relu(self.conv1(X))))
        return X

    def get_conv2(self, X):
        X = self.pool(self.drop(F.relu(self.conv1(X))))
        X = self.pool(self.drop(F.relu(self.conv2(X))))
        return X
    
    
def get_max_data(batch:int, low:int, high:int, n_ctx:int, device:str='cpu'):
    batch = np.random.randint(low, high, size=(batch, n_ctx))
    batch[:, -1] = 0 
    batch = torch.from_numpy(batch).long().to(device)
    return batch

class MaxDataset(torch.utils.data.Dataset):
    def __init__(self, size:int, config: HookedTransformerConfig, device:str='cpu'):
        super(MaxDataset, self).__init__()
        self.size = size
        self.low = 1
        self.high = config.d_vocab - 1
        self.n_ctx = config.n_ctx
        self.device = device

        data = []
        for h in range(2, self.high):
            data.append(get_max_data(size//(self.high-2), self.low, h, self.n_ctx, device))
            if h == self.high-1:
                data.append(get_max_data(size%(self.high-2), self.low, h, self.n_ctx, device))
        self.data = torch.cat(data, dim=0)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

