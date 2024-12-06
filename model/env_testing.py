import gymnasium as gym
import random
import numpy as np
import os
import pandas as pd
from gymnasium.wrappers import FlattenObservation
from gymnasium.spaces.utils import flatten_space

env = gym.make('gym_environment:gym_environment/SimpleBattery', predict=False, day_offset=0)
# env = GymEnv('gym_environment:gym_environment/SimpleBattery', predict=True, day_offset=0, device='cpu')

print(env.observation_space.is_np_flattenable)
print(env.observation_space.sample())
# wrapped_env = FlattenObservation(env)
print(env.observation_space.shape)

print(flatten_space(env.observation_space).shape)
print(env.observation_space.sample())

env = FlattenObservation(env)

observation, info = env.reset()

print("_______")
count = 0
length = 1440
for i in range(length):
    observation, reward, terminated, truncated, info = env.step(2)
    # print(observation)
    if i % 100000 == 0:
        print(i)
    if (i+1) % 1 == 0:
        imbalance_take_price = info['imb']['imbalance_take_price']
        imbalance_feed_price = info['imb']['imbalance_feed_price']
        medium_price = info['imb']['mid_price']
        calculated_take_price = observation[3]
        calculated_feed_price = observation[2]
        calc_medium_price = observation[5]
        print(observation[0])
        print(reward)
        if medium_price != calc_medium_price:
            # print("___")
            # print("reg_stat:{}".format(info['imb']['imbalance_regulation_state']))
            print(medium_price)
            print(calc_medium_price)
            print("Date:{} Time:{}".format(info['imb']['date'], info['imb']['time']))
            print(info['current_state']['hour_of_day'])
            # print("Info_feed:{} Info_take:{}".format(imbalance_feed_price, imbalance_take_price))
            # print("Obs_feed:{} Obs_take:{}".format(calculated_feed_price, calculated_take_price))
            count+=1
    # print("Observation:{} Reward:{}".format(observation, reward))

print("Error rate:{}".format(count/length))

# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html