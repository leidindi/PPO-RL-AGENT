import gymnasium
import random
import numpy as np
import os
import pandas as pd
from gymnasium.wrappers import FlattenObservation
from gymnasium.spaces.utils import flatten_space

env = gymnasium.make('gym_environment:gym_environment/SimpleBattery')
print(env.observation_space.is_np_flattenable)
print(env.observation_space.sample())
# wrapped_env = FlattenObservation(env)
print(env.observation_space.shape)

print(flatten_space(env.observation_space).shape)

env = FlattenObservation(env)

observation, info = env.reset()
for i in range(225):
    observation, reward, terminated, truncated, info = env.step(0)
    print("Info:{}".format(info))
    print("Observation:{} Reward:{}".format(observation, reward))
