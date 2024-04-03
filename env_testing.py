import gymnasium

env = gymnasium.make('gym_examples:gym_examples/GridWorld')
print(env.reset())