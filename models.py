import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class MLP(nn.Module):

    def __init__(self, *layer_sizes):
        '''
        layer_sizes:: integers corresponding to layer sizes
               includes input and output dim
        TODO: add optional dropout, add typing for layer_sizes
        '''
        global device
        

        super(MLP, self).__init__()
        self.operations = nn.ModuleList([])
        for i in range(len(layer_sizes)-1):
            x, y = layer_sizes[i], layer_sizes[i+1]
            self.operations.append(nn.Linear(x, y))

        self.to(device)
        
    def forward(self, x):
        for i, f in enumerate(self.operations):
            x = f(x)
            if i < len(self.operations)-1:
                x = F.relu(x)
        return x

class Actor(nn.Module):

    def __init__(self, *layer_sizes):
        super(Actor, self).__init__()
        self.mlp = MLP(*layer_sizes)

    def forward(self, x):
        x = self.mlp(x)
        m, _ = x.max(dim=-1, keepdim=True)
        x = x - m
        x = F.softmax(x, dim=-1)

        return x

class Critic(nn.Module):

    def __init__(self, *layer_sizes):
        super(Critic, self).__init__()
        self.mlp = MLP(*layer_sizes)

    def forward(self, x):
        x = self.mlp(x)

        return x.squeeze(-1)