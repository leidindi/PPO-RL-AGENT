import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self, input_size, num_actions):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc_actor = nn.Linear(128, num_actions)
        self.fc_critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy = F.softmax(self.fc_actor(x), dim=-1)
        value = self.fc_critic(x)
        return policy, value

def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)  # Normalize returns
    return returns

def train(env_name, num_episodes, gamma):
    env = gym.make(env_name, max_episode_steps=600)
    input_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    model = ActorCritic(input_size, num_actions)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    all_rewards = []

    for episode in range(num_episodes):
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

            next_state, reward, done, truncated, _ = env.step(action.item())

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state
            if truncated:
                break

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

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {sum(rewards)}")

    env.close()
    return all_rewards

if __name__ == "__main__":
    env_name = 'CartPole-v1'
    num_episodes = 200
    gamma = 0.99

    all_rewards = train(env_name, num_episodes, gamma)

    plt.plot(all_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('A2C Training Progress')
    plt.show()


#https://www.youtube.com/watch?v=LawaN3BdI00
#https://www.youtube.com/watch?v=OcIx_TBu90Q&t=0s