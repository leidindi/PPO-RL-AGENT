import gymnasium
from gymnasium.wrappers import FlattenObservation
from gymnasium.spaces.utils import flatten_space

env = gymnasium.make('gym_environment:gym_environment/SimpleBattery')
print(env.observation_space.is_np_flattenable)
print(env.observation_space.sample())
# wrapped_env = FlattenObservation(env)
print(env.observation_space.shape)

print(flatten_space(env.observation_space).shape)

wrapper = FlattenObservation(env)