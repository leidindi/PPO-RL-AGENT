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
from autoencoder import DenseAutoencoder
import autoencoder
from datetime import datetime
import copy

class BatteryMarketEnv(gym.Env):
    def __init__(self, csv_path = "final-imbalance-data-training.csv", autoencoder_model = None, device="cuda"):
        super(BatteryMarketEnv, self).__init__()
        self.device = device
        # Load market data
        self.csv_headers = pd.read_csv(csv_path, nrows=0).columns.tolist()
        self.csv_headers = self.csv_headers[7:]

        self.encoder_dimmension = 60*24*1 #( then elongated by 10 because of the interleaving for the encoders))
        # how long each training period is
        self.window_size = 60*24*8
        # constant for overlap between windows, circa 66% overlap if 60*24*6 and 60*24*2
        self.stride = 60*24*3
        self.batch_size = 4

        self.data = autoencoder.load_csv_for_autoencoder(csv_file = csv_path, feature_cols=self.csv_headers
                                                         , window_size = self.window_size, stride = self.stride
                                                         , batch_size = self.batch_size, device=self.device)
        # the first training window iteration is a wrapped list of [object], so it's 
        # unwrapped and thrown to the the device
        self.episode =  next(iter(self.data))[0].to(device)
        # flip axes and align memmory accordingly
        self.episode = self.episode.permute(0,2,1).contiguous()

        # Environment constants
        self.battery_capacity = 2.0  # MWh
        self.charge_speed = 1.0  # MW (2 hours until full charge)
        self.cycle_cost = 80.0  # Cost per full charge/discharge cycle

        # State variables
        self.battery_status = np.zeros((self.batch_size, 1)) + 1.0  # Initial battery charge level (in MWh)
        self.cash_balance = np.zeros((self.batch_size, 1)) + 1000.0  # Initial cash balance
        self.current_step = 0  # Track the current time step
        
        self.quarter_feed   = np.zeros((self.batch_size, 1))
        self.quarter_take   = np.zeros((self.batch_size, 1))
        self.quarter_mid    = np.zeros((self.batch_size, 1))
        self.quarter_state  = np.zeros((self.batch_size, 1))

        # Autoencoder for market data compression
        self.autoencoder = autoencoder_model
        for key, value in self.autoencoder.items():
            self.autoencoder[key] = value.eval()
        
        # we precalculate the encoded episode contexts, this initializes that
        self.encoded_episode = torch.zeros(
            (self.window_size - self.encoder_dimmension, self.encoder_dimmension * len(self.autoencoder)),
            device=self.device,
        )

        # Observation space (compressed data + battery variables)
        # Compressed market data: 16 features + battery status + average price + cash balance
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(16+64+3,), # 83 = 16 battery variables + 64 compressed features + battery status + average price + cash balance
            dtype=np.float32
        )
        # Action space: -1 (discharge), 0 (do nothing), 1 (charge)
        self.action_space = spaces.Discrete(3)

    def reset(self):
        # Reset environment state
        self.battery_status = 1.0
        self.cash_balance = 1000.0
        self.current_step = 0
        
        self.quarter_feed   = np.zeros((self.batch_size, 1))
        self.quarter_take   = np.zeros((self.batch_size, 1))
        self.quarter_mid    = np.zeros((self.batch_size, 1))
        self.quarter_state  = np.zeros((self.batch_size, 1))

        self.episode =  next(iter(self.data))[0].to(self.device)
        self.episode = self.episode.permute(0,2,1).contiguous()

        num_rows_to_process = self.window_size - self.encoder_dimmension
        num_columns = len(self.autoencoder)

        all_column_names = ["imbalance_feed_price",
                   "imbalance_take_price",
                   "mid_price",
                   "state"]

        self.encoded_episode = torch.zeros(
            (num_rows_to_process, self.encoder_dimmension * num_columns),
            device=self.device,
        )

        compressed_context = []

        for j, col_name in enumerate(self.csv_headers):
            if not(col_name in all_column_names):
                print(f'Column name:{col_name} was not recognized')
                continue
            # Get the data for the current column
            col_data = self.episode[:, j,:]

            # Transformation: Sliding window
            batch_transformed = []

            for batch in col_data:  # Iterate over each batch
                rows_transformed = [batch[i:i + self.encoder_dimmension] for i in range(num_rows_to_process)]
                rows_transformed = torch.stack(rows_transformed)
                
                batch_transformed.append(rows_transformed)
                
            # Final tensor: Shape [64, 15841, 1440]
            final_tensor = torch.stack(batch_transformed)
            del batch_transformed
            del rows_transformed
            # Squeeze along dim=1
            squeezed_tensor = final_tensor.view(-1,1440)
            del final_tensor
            # Define the batch size for processing parts of the tensor
            #print(squeezed_tensor.shape)
            batch_size = squeezed_tensor.shape[0]//1  # Change to a size that fits into memory

            # Split the tensor into smaller chunks (batches)
            results = []
            for i in range(0, squeezed_tensor.size(0), batch_size):
                # Slice the tensor to process a batch
                batch = copy.deepcopy(squeezed_tensor[i:i + batch_size])
                
                interleaved_batch = batch.repeat_interleave(10, dim=1)
                del batch
                # Process the batch (for example, pass through a function)
                # Replace `autoencoder.encode` with your function
                processed_batch = self.autoencoder[col_name].encode(interleaved_batch)
                del interleaved_batch
                # Optionally store or combine results if needed
                # (For example, append to a list or stack them back together)
                results.append(processed_batch)
                del processed_batch
            results = torch.stack(results).view(self.batch_size,-1,64)

            compressed_context.append(results)
            del results
        self.encoded_episode  = torch.stack(compressed_context, dim=3)
        shape = self.encoded_episode.shape
        self.encoded_episode = self.encoded_episode.view(shape[0],shape[1],-1)
        del compressed_context
        # Get the initial observation
        return self._get_observation()

    def step(self, action, charge_state):
        """
        Perform one time step in the environment.

        action:
            -1: Discharge battery
            0: Do nothing
            1: Charge battery
        """
        # Apply action
        battery_limit = 0.1
        if action == 1:  # Charge
            charge_amount = min(self.charge_speed / 60, self.battery_capacity - self.battery_status)  # MW to MWh
            if not (self.battery_status + charge_amount > self.battery_capacity*(1-battery_limit)):
                self.battery_status += charge_amount
                self.cash_balance -= charge_amount * self._get_price(charge_state, action)  # Deduct cost of charging
        elif action == -1:  # Discharge
            discharge_amount = min(self.charge_speed / 60, self.battery_status)  # MW to MWh
            if not (self.battery_status - discharge_amount < self.battery_capacity*battery_limit):
                self.battery_status -= discharge_amount
                self.cash_balance += discharge_amount * self._get_price(charge_state, action)  # Add revenue from selling

        # Update the time step
        self.current_step += 1

        # Compute reward (cash balance change)
        reward = self.cash_balance

        # Check if episode is done
        done = self.current_step >= len(self.data) - 1

        # Get the next observation
        obs = self._get_observation()

        return obs, reward, done, {}

    def _get_price(self, reg_state, action):
        """Get the current electricity market price."""
        if action == 0:
            # standing by, unchanged
            return 0
        
        if reg_state == 0:
            # no regulation, mid-price used
            column_index = next((i for i, column_name in enumerate(self.csv_headers) if column_name == "mid_price"), -1)
            if column_index == -1:
                # the column name was not found
                raise ValueError
            step_index = self.encoder_dimmension + self.current_step

            if action == 1:
                # you are buying
                return -1 * self.episode[column_index, step_index]
            elif action == -1:
                # you are selling
                return self.episode[column_index, step_index]
        elif reg_state == -1:
            # down regulation
            # get lowest take price this 15 min block
            low_take_price = min(self.quarter_take) if self.quarter_take else None
            
            if pd.isnull(low_take_price):
                raise ValueError

            if action == 1:
                # you are buying
                return -1 * low_take_price
            elif action == -1:
                # you are selling
                return low_take_price
        elif reg_state == 1:
            # up regulation
            # get highest feed price this 15 min block
            high_feed_price = max(self.quarter_feed) if self.quarter_feed else None
            
            if pd.isnull(high_feed_price):
                raise ValueError

            if action == 1:
                # you are buying
                return -1 * high_feed_price
            elif action == -1:
                # you are selling
                return high_feed_price
        elif reg_state == 2:
            # upwards and downwards regulation in same block
            column_index = next((i for i, column_name in enumerate(self.csv_headers) if column_name == "mid_price"), -1)
            if column_index == -1:
                # the column name was not found
                raise ValueError
            step_index = self.encoder_dimmension + self.current_step

            mid_price = self.episode[column_index, step_index]
            high_feed_price = max(self.quarter_feed) if self.quarter_feed else None
            low_take_price = min(self.quarter_take) if self.quarter_take else None
            
            if action == 1:
                # you are buying
                if high_feed_price >= mid_price:
                    return -1 * high_feed_price
                else:
                    return -1 * mid_price
            if action == -1:
                # you are selling
                if low_take_price <= mid_price:
                    return low_take_price
                else:
                    return mid_price

        else:
            # The previous states are only permissable, the control flow should
            # never come here
            raise ValueError

    def _get_observation(self):
        """Get the current state of the environment."""

        historical_data = self.episode[:,:,self.encoder_dimmension+self.current_step]        
        # Compress historical data with the autoencoder
         
        current_context = self.encoded_episode[:,self.current_step,:]  # Assuming autoencoder outputs 40 features
        

        if self.current_step % 15 == 0:
            # a new 15 minute block
            self.quarter_feed = np.zeros((self.batch_size, 15))
            self.quarter_take = np.zeros((self.batch_size, 15))
            self.quarter_mid = np.zeros((self.batch_size, 15))
            self.quarter_state = np.zeros((self.batch_size, 15))
        
        # store step, taken straight from each column respectively
        self.quarter_state[:,self.current_step % 15]  = historical_data[:,-1].cpu()
        self.quarter_feed[:,self.current_step % 15]   = historical_data[:,-2].cpu()
        self.quarter_take[:,self.current_step % 15]   = historical_data[:,-3].cpu()
        self.quarter_mid[:,self.current_step % 15]    = historical_data[:,-7].cpu()

        # Create observation
        #avg_price = self.data[start_idx:self.current_step, 0].mean().item() if self.current_step > 0 else 0.0
        #avg_price = self.average_price
        obs = torch.cat([
            current_context,
            torch.tensor([self.battery_status, self.cash_balance], device=self.device),
            torch.tensor(historical_data[-1,:], device=self.device),
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
    loaded_autoencoder = None
    combined_autoencoder = {"imbalance_feed_price":None,
                   "imbalance_take_price":None,
                   "mid_price":None,
                   "state":None,
                   "times":None,
                   }
    with open("autoencoder-Dense-imbalance_feed_price-2024-12-16.pkl", "rb") as file:
        combined_autoencoder["imbalance_feed_price"] = pickle.load(file)

    with open("autoencoder-Dense-imbalance_take_price-2024-12-16.pkl", "rb") as file:
        combined_autoencoder["imbalance_take_price"] = pickle.load(file)

    with open("autoencoder-Dense-mid_price-2024-12-15.pkl", "rb") as file:
        combined_autoencoder["mid_price"] = pickle.load(file)

    with open("autoencoder-Dense-state-2024-12-16.pkl", "rb") as file:
        combined_autoencoder["state"] = pickle.load(file)

    with open("autoencoder-Dense-times-2024-12-16.pkl", "rb") as file:
        combined_autoencoder["times"] = pickle.load(file)

    env = BatteryMarketEnv(csv_path="final-imbalance-data-training.csv",autoencoder_model=combined_autoencoder)
    env_name = "Custom"
    N = 10
    batch_size = 4
    n_epochs = 15
    alpha = 0.0004

    if env_name == "MountainCarContinuous-v0":
        n_actions = 100
    elif env_name == "Custom":
        n_actions = env.action_space.n
    else:
        # the same as previous elif, but this is 
        # here in case I want to customize for custom envs
        n_actions = env.action_space.n
    agent = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)
    #agent.load_models()
    agent.memory.clear_memory()

    n_games = 10000
    agent.actor.train()
    agent.critic.train()
    figure_file = 'plots/' + env_name + '-' + str(datetime.now().strftime("%Y-%m-%d"))+'.png'

    best_score = -100000
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


