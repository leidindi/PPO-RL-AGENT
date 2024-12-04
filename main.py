import gymnasium as gym
import numpy as np
import ppo_torch
from ppo_torch import Agent
from utils import plot_learning_curve
import pickle 
import torch as torch   
import os
from gym import spaces
import pandas as pd

class BatteryMarketEnv(gym.Env):
    def __init__(self, csv_path, autoencoder_model, device="cuda"):
        super(BatteryMarketEnv, self).__init__()
        
        # Load market data
        self.data = pd.read_csv(csv_path).to_numpy()  # Load as NumPy for easy indexing
        self.data = torch.tensor(self.data, dtype=torch.float32, device=device)  # Store on GPU

        # Environment constants
        self.battery_capacity = 2.0  # MWh
        self.charge_speed = 1.0  # MW (2 hours for full charge)
        self.cycle_cost = 80.0  # Cost per full charge/discharge cycle
        self.device = device

        # State variables
        self.battery_status = 0.0  # Initial battery charge level (in MWh)
        self.cash_balance = 0.0  # Initial cash balance
        self.current_step = 0  # Track the current time step

        # Autoencoder for market data compression
        self.autoencoder = autoencoder_model
        self.autoencoder.eval()  # Set model to evaluation mode
        
        # Observation space (compressed data + battery variables)
        # Compressed market data: 40 features + battery status + average price + cash balance
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(43,),  # 40 compressed features + 3 battery variables
            dtype=np.float32
        )

        # Action space: -1 (discharge), 0 (do nothing), 1 (charge)
        self.action_space = spaces.Discrete(3)

    def reset(self):
        # Reset environment state
        self.battery_status = 0.0
        self.cash_balance = 0.0
        self.current_step = 0
        
        # Get the initial observation
        return self._get_observation()

    def step(self, action):
        """
        Perform one time step in the environment.

        action:
            -1: Discharge battery
            0: Do nothing
            1: Charge battery
        """
        # Apply action
        if action == 1:  # Charge
            charge_amount = min(self.charge_speed / 60, self.battery_capacity - self.battery_status)  # MW to MWh
            self.battery_status += charge_amount
            self.cash_balance -= charge_amount * self._get_price()  # Deduct cost of charging
        elif action == -1:  # Discharge
            discharge_amount = min(self.charge_speed / 60, self.battery_status)  # MW to MWh
            self.battery_status -= discharge_amount
            self.cash_balance += discharge_amount * self._get_price()  # Add revenue from selling

        # Update the time step
        self.current_step += 1

        # Compute reward (cash balance change)
        reward = self.cash_balance

        # Check if episode is done
        done = self.current_step >= len(self.data) - 1

        # Get the next observation
        obs = self._get_observation()

        return obs, reward, done, {}

    def _get_price(self):
        """Get the current electricity market price."""
        return self.data[self.current_step, 0].item()  # Assuming the first column is the price

    def _get_observation(self):
        """Get the current state of the environment."""
        # Extract the last 6 hours (360 minutes) of data
        start_idx = max(0, self.current_step - 360)
        historical_data = self.data[start_idx:self.current_step, :]
        if len(historical_data) < 360:  # Pad if insufficient data
            padding = torch.zeros((360 - len(historical_data), self.data.shape[1]), device=self.device)
            historical_data = torch.cat([padding, historical_data], dim=0)
        
        # Compress historical data with the autoencoder
        compressed_data = self.autoencoder(historical_data.unsqueeze(0)).squeeze(0)  # Assuming autoencoder outputs 40 features

        # Create observation
        avg_price = self.data[start_idx:self.current_step, 0].mean().item() if self.current_step > 0 else 0.0
        obs = torch.cat([
            compressed_data,
            torch.tensor([self.battery_status, avg_price, self.cash_balance], device=self.device)
        ])

        return obs.cpu().numpy()  # Return as NumPy array for Gym compatibility

    def render(self, mode="human"):
        """Render the environment state."""
        print(f"Step: {self.current_step}, Battery: {self.battery_status:.2f} MWh, "
              f"Cash: {self.cash_balance:.2f}, Price: {self._get_price():.2f}")

    def close(self):
        """Clean up resources."""
        pass


def sample_value_function(agent, episode, save_dir="value_data"):
    # Generate 1000x1000 grid of states
    x = np.linspace(-1.2, 0.6, 100)
    y = np.linspace(-0.07, 0.07, 100)
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
    N = 10
    batch_size = 5
    n_epochs = 15
    alpha = 0.0004

    if env_name == "MountainCarContinuous-v0":
        n_actions = 100
    else:
        n_actions = env.action_space.n
    agent = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)
    #agent.load_models()
    agent.memory.clear_memory()
    n_games = 1000000
    agent.actor.train()
    agent.critic.train()
    figure_file = 'plots/'+env_name+'.png'

    best_score = -150
    score_history = []

    learn_iters = 0
    avg_score = 0
    done = False
    truncated = False

    for i in range(n_games):
        observation = env.reset()
        if isinstance(observation, tuple):
            observation = observation[0]

        score = 0
        n_steps = 0
        highest = 0
        done = False
        truncated = False

        while True:
            action, prob, val = agent.choose_action(observation)
            if env_name == "MountainCarContinuous-v0":
                action = np.array([(action-n_actions)/n_actions])
            next_observation, reward, done, truncated, _ = env.step(action)
            next_observation = np.round(next_observation, 2)
            n_steps += 1
            
            if (np.abs(next_observation[0])*np.abs(next_observation[1])/(1.2*0.07))*2 > highest:
                highest = (np.abs(next_observation[0])*np.abs(next_observation[1])/(1.2*0.07))*2
                reward += highest

            if done:
                print("It managed to finish")
                reward += 500
            if truncated and 0:
                print(f'It did not manage to finish, got truncated at step {n_steps}')
                
            
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            
            if n_steps % N == 0 or done:# or truncated:
                agent.learn()
                learn_iters += 1
            if isinstance(next_observation, tuple):
                next_observation = next_observation[0]
            observation = next_observation

            if done:# or truncated:
                break

            
        score_history.append(score)
        avg_score = np.mean(score_history[-1:])

        if avg_score > best_score:# and not truncated:
            best_score = avg_score
            agent.save_models()

        if i % 50 == 0:
            sample_value_function(agent,i)

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)


