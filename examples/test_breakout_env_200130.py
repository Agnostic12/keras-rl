"""
This script is used to help you understand everything about breakout env in gym[atari]

To begin with, I will start from BreakoutDeterministic-v4.
"""

import numpy as np
import gym
import matplotlib.pyplot as plt

env_name = "BreakoutDeterministic-v4"
env = gym.make(env_name)
print(env.observation_space)
print(env.action_space)

# show game board
state = env.reset()

# check the functionality of different keys
while True:
    env.render()
    input0 = input()
    if input0 == "b":
        break
    if input0 == "":
        continue

    if int(input0) in list(range(env.action_space.n)):
        next_state, reward, done, _ = env.step(int(input0))
        print(reward, done)

# check for stop criterion
# for i in range(10000):
#     next_s, r, d, _ = env.step(1)
#     print(r, d)


# we can learn that:
# 0: stop
# 1: start the game (or if failed, we can't get a ball. There will be nothing here)
# 2: right
# 3: left

# we have 5 lives in total

# Besides, on click of the pad, we will only score if
# we break the bricks. And each brick will earn 1 point
