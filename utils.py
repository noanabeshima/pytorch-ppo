'''utils.py'''

from typing import Optional, List
from collections import namedtuple
from copy import deepcopy

import torch
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


Batch = namedtuple("Batch", "states policies actions rewards new_states")

class RewardNormalizer:
    def __init__(self, discount=.95, max_stored=1000):
        self.running_avg = 1
        self.running_avg_memory = [1]
        self.discount = discount
        self.max_stored = max_stored

    def transform_reward(self, reward):
        self.running_avg = self.discount * self.running_avg + reward
        self.running_avg_memory.append(deepcopy(self.running_avg))
        if len(self.running_avg_memory) > self.max_stored:
            self.running_avg_memory.pop(0)
        ram = torch.Tensor(self.running_avg_memory)
        reward = reward / (ram.std()+1)

        reward = reward.clamp(-10, 10)
        return float(reward)

def tensor(x):
    global device
    return torch.tensor(x).to(device).float()

def forward_sum(x: List[float], discount: Optional[float] = 1):
    x = deepcopy(x)
    for i in range(len(x)-2, -1, -1):
        x[i] = x[i] + discount * x[i+1]

    return x

def tensor_forward_sum(x: torch.Tensor, discount: Optional[float] = 1):
    for i in range(x.shape[-1]-2, -1, -1):
        x[:,i] = x[:,i] + discount * x[:, i+1]

    return x

def LD_to_DL(list_of_dicts):
        ''' list of dictionaries to dict of lists '''
        return {k:[d[k] for d in list_of_dicts] for k in list_of_dicts[0].keys()}

def episodes_to_batch(sample_episodes):
    '''(episode, timestep, type) List[List[Transition] -> (type, episode, timestep) Dict[torch.Tensor]''' 
    d = {'state':[], 'policy':[], 'action':[], 'reward':[], 'new_state':[]}
    for episode in sample_episodes:
        processed = LD_to_DL(episode)
        for k in d.keys():
            d[k].append(torch.stack(processed[k]))

    for k in d.keys():
        d[k] = torch.stack(d[k])

    return Batch(*d.values())