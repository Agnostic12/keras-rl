"""
This is a torch implementation of DDQN BreakoutDeterministic-v4

This is based on dqn_atari_200131_4.py. I will revise it to DDQN

1. change batch size
2. revise to DDQN

WRONG REVISION: OOM Error!
"""
import os
from time import time
from copy import deepcopy

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque, defaultdict, OrderedDict
from itertools import count
from contextlib import contextmanager

@contextmanager
def timeit():
    t1 = time()
    yield
    print("Total time cost is {:.0f}".format(time() - t1))



class Memory(object):
    def __init__(self, maxlen=100000):
        self.state = np.zeros(maxlen).astype(object)
        self.action = np.zeros(maxlen).astype(object)
        self.next_state = np.zeros(maxlen).astype(object)
        self.reward = np.zeros(maxlen).astype(object)
        self.done = np.zeros(maxlen).astype(object)
        self.ind = 0
        self.maxlen = maxlen

    @property
    def size(self):
        return min(self.ind, self.maxlen)

    def add_memory(self, state, action, next_state, reward, done):
        curr_ind = self.ind % self.maxlen
        self.state[curr_ind] = state
        self.action[curr_ind] = action
        self.next_state[curr_ind] = next_state
        self.reward[curr_ind] = reward
        self.done[curr_ind] = done
        self.ind += 1


    def get_memory(self, size):
        n_total = self.ind if self.ind < self.maxlen - 1 else self.maxlen - 1
        inds = np.random.choice(np.arange(n_total), size)
        return [self.state[inds],
                self.action[inds],
                self.next_state[inds],
                self.reward[inds],
                self.done[inds]]


class Net(nn.Module):
    def __init__(self, env):
        super(Net, self).__init__()
        obs_shape = env.observation_space.shape
        act_n = env.action_space.n - 1
        self.conv1 = nn.Conv2d(obs_shape[2], 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.dense1 = nn.Linear(22528, 512)
        self.dense2 = nn.Linear(512, act_n)

    def forward(self, x):
        # normalization 0-256 -> 0-1
        x = x / 256
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
    def __init__(self, net, memory, gamma, soft_update_rate=0.99):
        self.net = net
        self.target_net = deepcopy(self.net)
        self.memory = memory
        self.gamma = gamma
        self.soft_update_rate = soft_update_rate
        self.optimizer = optim.Adam(self.net.parameters())

    def greedy_action(self, state):
        raw_action = self.net.forward(state).argmax().detach().numpy()
        return raw_action

    # don't use 1
    def eps_greedy_action(self, state, eps=0.9):
        output = self.net.forward(state).detach().cpu().numpy()[0]
        # probs_output = (1 - eps) * np.ones(output.shape) / (output.shape[1] - 1)
        # probs_output[(np.arange(output.shape[0]), output.argmax(axis=1))] = eps
        # return [np.random.choice(np.arange(output.shape[1]), p=p) for p in probs_output]\
        probs_output = (1 - eps) * np.ones(output.shape) / (output.shape[0] - 1)
        probs_output[output.argmax()] = eps
        raw_action = np.random.choice(np.arange(output.shape[0]), p=probs_output)
        return raw_action

    def update_target_net(self):
        new_state_dict = OrderedDict({key: (1 - self.soft_update_rate) * self.target_net.state_dict()[key] +
                                            self.soft_update_rate * self.net.state_dict()[key]
                                     for key in self.net.state_dict().keys()})
        self.target_net.load_state_dict(new_state_dict)

    def update(self, sampled_memory, counter=None, writer=None):
        # soft update target net
        self.update_target_net()
        states = torch.cat(tuple(sampled_memory[0]), axis=0)
        actions = torch.cat(tuple(sampled_memory[1]), axis=0)
        next_states = torch.cat(tuple(sampled_memory[2]), axis=0)
        rewards = torch.cat(tuple(sampled_memory[3]), axis=0)
        dones = torch.cat(tuple(sampled_memory[4]), axis=0)

        self.optimizer.zero_grad()
        pred_next_state_vals_net = self.net.forward(next_states)
        action_picked_next_state = pred_next_state_vals_net.argmax(dim=1, keepdims=True)
        pred_next_state_vals_target_net = self.target_net.forward(next_states)
        pred_next_state_val_target_net = torch.gather(pred_next_state_vals_target_net, 1, action_picked_next_state)
        pred_state_vals = self.net.forward(states)
        td_target = rewards + \
                    self.gamma * torch.max(pred_next_state_val_target_net, dim=1, keepdims=True).values * (~dones).to(torch.float32)
        td_error = td_target - torch.gather(pred_state_vals, 1, actions)
        loss = torch.mean(td_error ** 2)
        if writer:
            writer.add_scalar("loss", loss, counter)
        loss.backward()
        self.optimizer.step()


def train():
    res = []
    n_episodes = 20000
    train_every_n = 50
    frame_skipping = 2
    batch_size = 512
    memory = Memory(100000)
    exp_name = "200202_1"
    eps_start = 0.75
    eps_end = 0.95

    env = gym.make("BreakoutDeterministic-v4")
    net = Net(env).cuda()
    dqn = DQNAgent(net, memory, 0.9, 0.95)
    # to clear logs
    os.system('sudo rm -rf logs_{}'.format(exp_name))

    writer = SummaryWriter("logs_{}".format(exp_name))

    global_counter = 0

    for i_episode in range(n_episodes):
        t1 = time()
        state = env.reset()
        state_tensor = torch.from_numpy(state[np.newaxis, :]).cuda().permute(0, 3, 2, 1).to(torch.float32)
        next_state = None
        total_rewards = 0
        action_dict = defaultdict(int)
        for t in count():
            curr_eps = eps_start + (eps_end - eps_start) / n_episodes * i_episode
            raw_action = dqn.eps_greedy_action(state_tensor, eps=curr_eps)
            action_dict[raw_action] += 1
            action = raw_action if raw_action == 0 else raw_action + 1  # cuz we ignore action 1
            # frame skipping
            curr_rewards = 0
            for _ in range(frame_skipping):
                # env.render()
                next_state, reward, done, _ = env.step(action)
                # start new life
                if np.all(next_state == state):
                    next_state, reward, done, _ = env.step(1)
                curr_rewards += reward
                if done: break

            total_rewards += curr_rewards
            memory.add_memory(state_tensor,
                              torch.from_numpy(np.array([[raw_action]])).cuda(),
                              torch.from_numpy(next_state[np.newaxis, :]).cuda().permute(0, 3, 2, 1).to(torch.float32),
                              torch.from_numpy(np.array([[curr_rewards]])).cuda(),
                              torch.from_numpy(np.array([[done]])).cuda())

            if memory.size >= batch_size and global_counter % train_every_n == 0 :
                sampled_memory = memory.get_memory(batch_size)
                dqn.update(sampled_memory, counter=global_counter, writer=writer)

            if done:
                break
            state = next_state
            global_counter += 1
        res.append([i_episode, total_rewards])
        writer.add_scalar("total_rewards", total_rewards, i_episode)
        print("In episode {}, total rewards is {}, {}, time {:.2f}".format(
                                                              i_episode,
                                                              total_rewards,
                                                              [[i, action_dict[i]] for i in range(3)],
                                                              time() - t1))
    with open("result_{}.csv".format(exp_name), "w") as f:
        for i_episode, total_rewards in res:
            f.write("{},{}\n".format(i_episode, total_rewards))

    # save model
    torch.save(net.state_dict(), "model/{}_{}".format(exp_name, int(time.time())))

if __name__ == "__main__":
    with timeit():
        train()
