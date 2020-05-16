from typing import List, Optional

try:
    from typing import TypedDict
except:
    from typing_extensions import TypedDict

import torch
import numpy as np

from utils import tensor


class Transition(TypedDict):
    state: torch.Tensor
    policy: torch.Tensor
    action: int
    reward: float
    new_state: torch.Tensor

def zero_transition(t: Transition):
    result = Transition(
                state=tensor(torch.zeros(t['state'].shape)),
                policy=tensor(torch.ones(t['policy'].shape)/t['policy'].shape[0]),
                action=tensor(0).long(),
                reward=tensor(0.),
                new_state=tensor(torch.zeros(t['new_state'].shape)))

    return result

class ReplayMemory:
    
    def __init__(self, max_size):
        self.episodes = []
        self.episode_lengths = []
        self.max_size = max_size

    def sample(self, n_samples=1, sample_length=1):
        p = np.array(self.episode_lengths)/sum(self.episode_lengths)
        sample_idx = np.random.choice(len(self.episodes), size=n_samples, p=p)
        sample_episodes = [self.episodes[i] for i in sample_idx]

        zero_transitions = [zero_transition(sample_episodes[0][0]) for _ in range(sample_length)]
        
        sample_trajectories = []
        for ep in sample_episodes:
            j = np.random.choice(len(ep))
            end_of_sample = min(len(ep), j+sample_length)
            n_zero_transitions = max(0, j+sample_length-len(ep))
            trajectory = ep[j:end_of_sample]+zero_transitions[0:n_zero_transitions]
            sample_trajectories.append(trajectory)
            assert len(trajectory) == sample_length
        
        assert len(sample_trajectories) == n_samples
        
        return sample_trajectories

    def append(self, episode: List[Transition]):
        self.episodes.append(episode)
        self.episode_lengths.append(len(episode))

        while sum(self.episode_lengths) > self.max_size:
            self.episodes.pop(0)
            self.episode_lengths.pop(0)

    def extend(self, *episodes: List[List[Transition]]):
        for episode in episodes:
            self.append(episode)

    def __len__(self):
        return sum(self.episode_lengths)
