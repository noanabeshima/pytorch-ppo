''' train.py '''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import gym
import numpy as np

from models import Actor, Critic
from replaymemory import ReplayMemory, Transition
from utils import RewardNormalizer, tensor, episodes_to_batch, tensor_forward_sum

from copy import deepcopy



# Config
# ~~~~~~~~~~~~~~~~~~~~~
N_EPOCHS = 4
N_SAMPLES = 1000
SAMPLE_LENGTH = 15
memory_capacity = 2000
GAMMA = .997
LAMBDA = .95
EPSILON = .2
TARGET_DISCOUNT = .4
N_TIMESTEPS_PER_UPDATE = 300
# ~~~~~~~~~~~~~~~~~~


# Initialization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
env = gym.make('CartPole-v1')

replay_memory = ReplayMemory(memory_capacity)

policy_net = Actor(sum(env.observation_space.shape), 200, env.action_space.n)
value_net = Critic(sum(env.observation_space.shape), 200, 1)
target_value_net = Critic(sum(env.observation_space.shape), 200, 1) 
target_value_net.load_state_dict(value_net.state_dict())
target_value_net.eval()

params = list(policy_net.parameters()) + list(value_net.parameters())
optimizer = optim.SGD(params, lr=1e-3, momentum=.9, weight_decay=1e-6)

writer = SummaryWriter()

reward_normalizer = RewardNormalizer()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


global_t = 0
for ep in range(10000):

    # episode loop
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    trajectory = []
    ep_t = 0
    state = tensor(env.reset())
    ep_return = 0
    while True:
        with torch.no_grad():
            policy = policy_net(state)

        action = np.random.choice(env.action_space.n, p=policy.cpu().numpy())

        new_state, reward, done, _ = env.step(action)
        new_state = tensor(new_state)

        ep_return += reward
        # reward = reward_normalizer.transform_reward(reward)
        
        transition = Transition(state = state, policy = policy,
                                action = tensor(action).long(), reward = tensor(reward),
                                new_state = new_state)
        trajectory.append(transition)

        state = new_state

        if done:
            replay_memory.append(trajectory)
            if ep % 5 == 0:
                writer.add_scalar('episode_return/timestep', int(ep_return), int(global_t))
                writer.add_scalar('episode_return/episode', int(ep_return), int(ep))
            break

        global_t += 1
        ep_t += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if global_t % N_TIMESTEPS_PER_UPDATE == 0 and len(replay_memory) > N_SAMPLES:
            
            # training loop
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            for _ in range(N_EPOCHS):
                sample = replay_memory.sample(n_samples=N_SAMPLES, sample_length=SAMPLE_LENGTH)
                batch = episodes_to_batch(sample)

                values = value_net(batch.states)
                target_values = target_value_net(batch.states)
                shifted_values = torch.cat((target_values[:,1:], tensor(torch.zeros(target_values.shape[0], 1))), dim=-1)

                deltas = (-values + batch.rewards + GAMMA * shifted_values.detach())
                advantages = tensor_forward_sum(deltas, GAMMA * LAMBDA)

                value_net_loss = (advantages**2).mean()

                policies = policy_net(batch.states)
                entropy_loss = (policies*torch.log(policies)).mean()
                policies = policies.gather(dim=-1, index=batch.actions.unsqueeze(-1)).squeeze()
                old_policies = batch.policies.gather(dim=-1, index=batch.actions.unsqueeze(-1)).squeeze()

                ratios = policies / old_policies
                mask = (ratios < 1.4) * (ratios > .6)
                ratios = ratios*mask

                policy_net_loss = -torch.min(ratios*advantages, ratios.clamp(1 - EPSILON, 1 + EPSILON)*advantages.detach()).sum()/mask.sum()

                writer.add_scalar('Loss/Policy', policy_net_loss, global_t)
                writer.add_scalar('Loss/Value', value_net_loss, global_t)
                writer.add_scalar('Loss/entropy', -entropy_loss, global_t)

                loss = policy_net_loss + 50*value_net_loss +.01*entropy_loss

                optimizer.zero_grad()
                loss.backward()
                for p in params:
                    p.grad.data.clamp_(-1, 1)
                optimizer.step()


            vsd = value_net.state_dict()
            tsd = target_value_net.state_dict()
            for k in vsd:
                tsd[k] = (1-TARGET_DISCOUNT)*vsd[k]+TARGET_DISCOUNT*tsd[k]
            target_value_net.load_state_dict(tsd)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

env.close()