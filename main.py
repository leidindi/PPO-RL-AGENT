import gymnasium as gym
import numpy as np
import ppo_torch
from ppo_torch import Agent
from utils import plot_learning_curve
import pickle 
import torch as torch

def sample_value_function(agent, episode, save_dir="value_data"):
    # Generate 1000x1000 grid of states
    x = np.linspace(-1.2, 0.6, 1000)
    y = np.linspace(-0.07, 0.07, 1000)
    xv, yv = np.meshgrid(x, y)
    
    # Flatten the grid for evaluation
    grid_states = np.stack([xv.flatten(), yv.flatten()], axis=1)
    
    # Convert to tensor
    grid_states_tensor = torch.tensor(grid_states, dtype=torch.float32, device=agent.critic.device)
    
    # Compute values using the critic
    with torch.no_grad():
        z = agent.critic.forward(grid_states_tensor).clone().detach()
    z = z.cpu().numpy().flatten()
    # Combine coordinates and values
    xyz = np.column_stack((grid_states, z))
    
    # Save the data
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"episode_{episode}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(xyz, f)
    print(f"Saved data for episode {episode} at {filename}")


if __name__ == '__main__':
    env_name = "MountainCar-v0"
    env = gym.make(env_name)
    N = 40
    batch_size = 4
    n_epochs = 15
    alpha = 0.1

    if env_name == "MountainCarContinuous-v0":
        n_actions = 100
    else:
        n_actions = env.action_space.n
    agent = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)
    #agent.load_models()
    agent.memory.clear_memory()
    n_games = 1000
    agent.actor.train()
    agent.critic.train()
    figure_file = 'plots/'+env_name+'.png'

    best_score = -10000
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
        truncated = False
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
        avg_score = np.mean(score_history[-1:])

        if avg_score > best_score:# and not truncated:
            best_score = avg_score
            agent.save_models()

        if i % 50 == 0 and i > 0:
            sample_value_function(agent,i)

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)


