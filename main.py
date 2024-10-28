import gymnasium as gym
import numpy as np
import ppo_torch
from ppo_torch import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env_name = "MountainCar-v0"
    env = gym.make(env_name)
    N = 64
    batch_size = 32
    n_epochs = 5
    alpha = 0.0001
    if env_name == "MountainCarContinuous-v0":
        n_actions = 100
    else:
        n_actions = env.action_space.n
    agent = Agent(n_actions=n_actions, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    #agent.load_models()
    agent.memory.clear_memory()
    n_games = 100
    agent.actor.train()
    agent.critic.train()
    figure_file = 'plots/'+env_name+'.png'

    best_score = -200
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        if isinstance(observation, tuple):
            observation = observation[0]
        done = False
        score = 0

        steps = 0
        while not done:
            steps += 1
            action, prob, val = agent.choose_action(observation)
            if env_name == "MountainCarContinuous-v0":
                action = np.array([(action-n_actions)/n_actions])
            observation_, reward, done, info, _ = env.step(action)
            n_steps += 1
            #print(observation_[0])
            reward += np.abs(observation_[1])

            if done:
                print("It managed to finish")
                reward += 500
            
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            if isinstance(observation_, tuple):
                observation_ = observation_[0]
            observation = observation_
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)


