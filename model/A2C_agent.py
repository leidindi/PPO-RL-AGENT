import multiprocessing.pool
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import time

# setup plots
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

def plot_durations(epoch_rewards, means, show_result=False):
    plt.figure(1)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(epoch_rewards)

    # Take 50 ep. Average and plot them
    if len(epoch_rewards) >= 50:
      mean = np.mean(epoch_rewards[-50:]) 
    else:
      mean = np.mean(epoch_rewards)

    means.append(mean)
    
    plt.plot(means)

    plt.pause(0.0001) # pause a bit to update plots
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True) 
        else:
            display.display(plt.gcf())

    return means

class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(ActorCritic, self).__init__()
        self.actor_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

        self.critic_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        policy = F.softmax(self.actor_layers(x), dim=-1)
        value = self.critic_layers(x)
        return policy, value
    
    def select_action(self, x):
        return torch.multinomial(self(x), 1).detach().numpy()
    
def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)  # Normalize returns
    return returns

def train(num_episodes, gamma):
    days = 1
    day_offset = 0
    charge_penatly_mwh = 0.0

    env = gym.make('gym_environment:gym_environment/SimpleBattery', days=days, predict=True, day_offset=day_offset, charge_penalty_mwh=charge_penatly_mwh)
    env = FlattenObservation(env)
    input_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    model = ActorCritic(input_size, input_size*2, num_actions)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    all_rewards = []
    means = []

    for episode in range(num_episodes):
        start_time = time.time()
        log_probs = []
        values = []
        rewards = []

        state, _ = env.reset()
        done = False

        while not done:
            state = torch.tensor(state, dtype=torch.float32)
            policy, value = model(state)
            dist = Categorical(policy)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            next_state, reward, done, _, _ = env.step(action.item())

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state

        all_rewards.append(sum(rewards))

        returns = compute_returns(rewards, gamma)

        actor_loss = 0
        critic_loss = 0
        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()
            actor_loss += -log_prob * advantage
            critic_loss += F.smooth_l1_loss(value, torch.tensor([R]))

        optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        optimizer.step()

        end_time = time.time()
        epoch_time = end_time - start_time

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {sum(rewards)}")
        print("Time until done: {}s".format(round(epoch_time * num_episodes - epoch_time * episode, 4)))
        means = plot_durations(all_rewards, means)

    env.close()
    return all_rewards

if __name__ == "__main__":
    num_episodes = 600
    gamma = 0.99

    all_rewards = train(num_episodes, gamma)