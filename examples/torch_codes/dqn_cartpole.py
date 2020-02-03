"""
This is a torch implementation of DQN Cartpole
"""
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from itertools import count

class Memory(object):
    def __init__(self, maxlen=10000):
        self.memory = deque(maxlen=maxlen)

    @property
    def size(self):
        return len(self.memory)

    def add_memory(self, tpl):
        self.memory.append(tpl)

    def get_memory(self, size):
        n_total = len(self.memory)
        inds = np.random.choice(np.arange(n_total), size)
        return [self.memory[ind] for ind in inds]



class NetCP(nn.Module):
    def __init__(self, env, batch_size):
        super(NetCP, self).__init__()
        obs_shape = env.observation_space.shape[0]
        act_n = env.action_space.n
        self.batch_size = batch_size
        self.dense1 = nn.Linear(obs_shape, 12)
        self.dense2 = nn.Linear(12, 12)
        self.dense3 = nn.Linear(12, act_n)

    def forward(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        x = torch.from_numpy(x).to(torch.float32).cuda()
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = F.relu(x)
        x = self.dense3(x)
        return x


class DQNAgent(object):
    def __init__(self, net, memory, gamma):
        self.net = net
        self.memory = memory
        self.gamma = gamma
        self.optimizer = optim.Adam(self.net.parameters())

    def greedy_action(self, state):
        output = self.net.forward(state).detach().numpy()
        return output.argmax()

    def eps_greedy_action(self, state, eps=0.95):
        output = self.net.forward(state).detach().cpu().numpy()[0]
        # probs_output = (1 - eps) * np.ones(output.shape) / (output.shape[1] - 1)
        # probs_output[(np.arange(output.shape[0]), output.argmax(axis=1))] = eps
        # return [np.random.choice(np.arange(output.shape[1]), p=p) for p in probs_output]\
        probs_output = (1 - eps) * np.ones(output.shape) / (output.shape[0] - 1)
        probs_output[output.argmax()] = eps
        return np.random.choice(np.arange(output.shape[0]), p=probs_output)

    def update(self, sampled_memory):
        states = np.concatenate([i[0][np.newaxis, :] for i in sampled_memory], axis=0)
        actions = np.array([i[1] for i in sampled_memory])[:, np.newaxis]
        next_states = np.concatenate([i[2][np.newaxis, :] for i in sampled_memory], axis=0)
        rewards = np.array([i[3] for i in sampled_memory])[:, np.newaxis]
        dones = np.array([i[4] for i in sampled_memory])[:, np.newaxis]

        pred_next_state_vals = self.net.forward(next_states).detach().cpu().numpy()
        pred_state_vals = self.net.forward(states)
        td_target = rewards + pred_next_state_vals.max(axis=1, keepdims=True) \
                    * (~dones).astype(float)
        td_error_tensor = torch.from_numpy(td_target).cuda() - \
                          torch.gather(pred_state_vals, 1, torch.from_numpy(actions).cuda())
        loss = torch.mean(td_error_tensor ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train():
    n_episodes = 1000
    batch_size = 32
    memory = Memory(10000)

    env = gym.make("CartPole-v0")
    net = NetCP(env, batch_size).cuda()
    dqn = DQNAgent(net, memory, 0.95)

    for i_episode in range(n_episodes):
        state = env.reset()
        total_rewards = 0
        for t in count():
            action = dqn.eps_greedy_action(state)
            next_state, reward, done, _ = env.step(action)
            total_rewards += reward
            memory.add_memory([state, action, next_state, reward, done])
            if memory.size >= batch_size:
                sampled_memory = memory.get_memory(batch_size)
                dqn.update(sampled_memory)

            if done:
                break
            state = next_state
        print("In episode {}, total rewards is {}".format(i_episode, total_rewards))


if __name__ == "__main__":
    train()
