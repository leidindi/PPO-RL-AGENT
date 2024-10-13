import gymnasium as gym
import random


""" Setting the custom environment to train the agent on """
env = gym.make('gym_environment:gym_environment/GridWorld', render_mode='human', max_episode_steps=300)
states = 2
actions = env.action_space.n



""" Visualising the custom environment """
episodes = 10
for episode in range(1, episodes +1):
    state = env.reset()
    terminated = False
    score = 0

    while not terminated:
        env.render()
        action = random.randint(0, 3)
        observation, reward, terminated, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
