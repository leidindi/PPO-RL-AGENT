import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import time
from ppo_torch import Agent

# Set up the environment

env_name = "MountainCar-v0"
env = gym.make(env_name  , render_mode="human") 

input_dim = env.observation_space.shape[0]  # Input size (4 for CartPole)
output_dim = env.action_space.n  # Number of possible actions (2 for CartPole)

N = 1
batch_size = 1
n_epochs = 10
alpha = 0.0003
n_games = 150

model = Agent(n_actions=env.action_space.n, input_dims=env.observation_space.shape,fc1_dims=24,fc2_dims=24)
model.load_models()

model.actor.eval()
model.critic.eval()
# Load the saved state dictionary (adjust the path to your saved checkpoint)
# Play the game and watch the agent
state = env.reset()
done = False

while not done:
    # Render the environment to see it in action
    env.render()

    # Convert the state to a tensor
    if isinstance(state, tuple):
            state = state[0]
    state_tensor = torch.tensor(state, dtype=torch.float32)
    # Add a batch dimension (if your model expects it)
    state_tensor = state_tensor.unsqueeze(0)

    # Get action from the trained model
    with torch.no_grad():
        dist = model.choose_action(state_tensor)
        action = dist[0]
        #print(action)
    # Take the action in the environment
    state, reward, done, info, _ = env.step(action)

    # Optional: Slow down the loop to better observe the game
    time.sleep(0.001)

# Close the environment after playing
env.close()
