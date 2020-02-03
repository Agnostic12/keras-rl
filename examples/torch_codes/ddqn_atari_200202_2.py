"""
This is a torch implementation of DQN BreakoutDeterministic-v4

Revision compared to previous version:
1. do update each episode but not each line. Cuz we will not have too much new info
if we do update each iter. (DOESN'T WORK)
2. change batch size
3. add time module
4. update each 100 cnts
5. Change the order of normalization(/256) and .cuda()
6. add time module

Conclusion: With DDQN setting, it works better now..
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
        act_n = env.action_space.n - 1
        self.conv1 = nn.Conv2d(obs_shape[2], 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.dense1 = nn.Linear(22528, 512)
        self.dense2 = nn.Linear(512, act_n)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x[np.newaxis, :]

        # x = x / 256
        x = torch.from_numpy(x).cuda().permute(0, 3, 2, 1).to(torch.float32)
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
        output = self.net.forward(state).detach().numpy()
        raw_action = output.argmax()
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
        states = np.concatenate([i[0][np.newaxis, :] for i in sampled_memory], axis=0)
        actions = np.array([i[1] for i in sampled_memory])[:, np.newaxis]
        next_states = np.concatenate([i[2][np.newaxis, :] for i in sampled_memory], axis=0)
        rewards = np.array([i[3] for i in sampled_memory])[:, np.newaxis]
        dones = np.array([i[4] for i in sampled_memory])[:, np.newaxis]

        self.optimizer.zero_grad()
        pred_next_state_vals_net = self.net.forward(next_states)
        action_picked_next_state = pred_next_state_vals_net.argmax(dim=1, keepdims=True)
        pred_next_state_vals_target_net = self.target_net.forward(next_states)
        pred_next_state_val_target_net = torch.gather(pred_next_state_vals_target_net, 1, action_picked_next_state).detach()
        pred_state_vals = self.net.forward(states)
        td_target = torch.from_numpy(rewards).cuda() + \
                    self.gamma * pred_next_state_val_target_net * torch.from_numpy(~dones).cuda().to(torch.float32)
        td_error_tensor = td_target - \
                          torch.gather(pred_state_vals, 1, torch.from_numpy(actions).cuda())
        loss = torch.mean(td_error_tensor ** 2)
        if writer:
            writer.add_scalar("loss", loss, counter)
        loss.backward()
        self.optimizer.step()


def train():
    res = []
    n_episodes = 30000
    train_every_n = 50
    frame_skipping = 2
    batch_size = 256
    memory = Memory(100000)
    exp_name = "200202_2"
    eps_start = 0.6
    eps_end = 0.99

    env = gym.make("BreakoutDeterministic-v4")
    net = Net(env).cuda()
    dqn = DQNAgent(net, memory, 0.9)
    # to clear logs
    os.system('sudo rm -rf logs_{}'.format(exp_name))

    writer = SummaryWriter("logs_{}".format(exp_name))

    global_counter = 0

    for i_episode in range(n_episodes):
        t1 = time()
        state = env.reset()
        next_state = None
        total_rewards = 0
        action_dict = defaultdict(int)
        for t in count():
            curr_eps = eps_start + (eps_end - eps_start) / n_episodes * i_episode
            raw_action = dqn.eps_greedy_action(state, eps=curr_eps)
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
            memory.add_memory([state, raw_action, next_state, curr_rewards, done])
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
    torch.save(net.state_dict(), "model/{}_{}".format(exp_name, int(time())))

if __name__ == "__main__":
    with timeit():
        train()
