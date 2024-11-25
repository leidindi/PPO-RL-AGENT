import gymnasium as gym
import numpy as np
import ppo_torch
from ppo_torch import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env_name = "MountainCar-v0"
    env = gym.make(env_name)
    N = 40
    batch_size = 4
    n_epochs = 15
    alpha = 0.001

    if env_name == "MountainCarContinuous-v0":
        n_actions = 100
    else:
        n_actions = env.action_space.n
    agent = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)
    agent.load_models()
    agent.memory.clear_memory()
    n_games = 1000
    agent.actor.train()
    agent.critic.train()
    figure_file = 'plots/'+env_name+'.png'

    best_score = -562
    score_history = []

    learn_iters = 0
    avg_score = 0

    for i in range(n_games):
        observation = env.reset()
        if isinstance(observation, tuple):
            observation = observation[0]
        done = False
        score = 0

        n_steps = 0
        highest = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            if env_name == "MountainCarContinuous-v0":
                action = np.array([(action-n_actions)/n_actions])
            next_observation, reward, done, truncated, _ = env.step(action)
            n_steps += 1
            
            if (np.abs(next_observation[0])*np.abs(next_observation[1])/(1.2*0.07))*2 > highest:
                highest = (np.abs(next_observation[0])*np.abs(next_observation[1])/(1.2*0.07))*2
                reward += highest

            if done:
                print("It managed to finish")
                reward += 500
            if truncated and 0:
                print(f'It did not manage to finish, got truncated at step {n_steps}')
                reward -= n_steps
                done = True
                
            
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            
            if n_steps % N == 0 or done:
                agent.learn()
                learn_iters += 1
            if isinstance(next_observation, tuple):
                next_observation = next_observation[0]
            observation = next_observation
            
        score_history.append(score)
        avg_score = np.mean(score_history[-2:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)


