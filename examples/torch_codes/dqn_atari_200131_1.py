"""
This is a torch implementation of DQN BreakoutDeterministic-v4
"""
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, defaultdict
from itertools import count


class Memory(object):
    def __init__(self, maxlen=100000):
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


class Net(nn.Module):
    def __init__(self, env):
        super(Net, self).__init__()
        obs_shape = env.observation_space.shape
        act_n = env.action_space.n
        self.conv1 = nn.Conv2d(obs_shape[2], 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.dense1 = nn.Linear(22528, 512)
        self.dense2 = nn.Linear(512, act_n)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x[np.newaxis, :]
        x = torch.from_numpy(x).permute(0, 3, 2, 1).to(torch.float32).cuda()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
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

    def eps_greedy_action(self, state, eps=0.9):
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

        self.optimizer.zero_grad()
        pred_next_state_vals = self.net.forward(next_states).detach().cpu().numpy()
        pred_state_vals = self.net.forward(states)
        td_target = rewards + pred_next_state_vals.max(axis=1, keepdims=True) \
                    * (~dones).astype(float)
        td_error_tensor = torch.from_numpy(td_target).cuda() - \
                          torch.gather(pred_state_vals, 1, torch.from_numpy(actions).cuda())
        loss = torch.mean(td_error_tensor ** 2)
        loss.backward()
        self.optimizer.step()


def train():
    res = []
    n_episodes = 20000
    batch_size = 64
    memory = Memory(10000)

    env = gym.make("BreakoutDeterministic-v4")
    net = Net(env).cuda()
    dqn = DQNAgent(net, memory, 0.9)

    for i_episode in range(n_episodes):
        state = env.reset()
        total_rewards = 0
        action_dict = defaultdict(int)
        for t in count():
            action = dqn.eps_greedy_action(state)
            action_dict[action] += 1
            next_state, reward, done, _ = env.step(action)
            total_rewards += reward
            memory.add_memory([state, action, next_state, reward, done])
            if memory.size >= batch_size:
                sampled_memory = memory.get_memory(batch_size)
                dqn.update(sampled_memory)

            if done:
                break
            state = next_state
        res.append([i_episode, total_rewards])
        print("In episode {}, total rewards is {}, {}".format(i_episode, total_rewards,
                                                              [[k, v] for k, v in action_dict.items()]))
    with open("result.csv", "w") as f:
        for i_episode, total_rewards in res:
            f.write("{},{}\n".format(i_episode, total_rewards))

if __name__ == "__main__":
    train()

"""
Under this setting, model will converges to action 0/1.

I guess this might be caused by:
1. No special treatment to 1
2. no frame skipping
3. normalization of inputs
4. rep buffer size: 1e6
"""
