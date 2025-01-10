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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
        self.current_context = None

        # Environment constants
        self.battery_capacity = 2.0  # MWh
        self.charge_speed = 1.0  # MW (2 hours until full charge)
        self.cycle_cost = 80.0  # Cost per full charge/discharge cycle
        self.battery_limit = 0.1

        # State variables
        self.battery_status = np.zeros(self.batch_size) + 1.0  # Initial battery charge level (in MWh)
        self.cash_balance = np.zeros(self.batch_size) + 1000.0  # Initial cash balance
        self.current_step = 0  # Track the current time step
        
        self.quarter_charge    = np.zeros(self.batch_size)
        self.quarter_discharge = np.zeros(self.batch_size)
        
        self.quarter_feed   = np.zeros(self.batch_size)
        self.quarter_take   = np.zeros(self.batch_size)
        self.quarter_mid    = np.zeros(self.batch_size)
        self.quarter_state  = np.zeros(self.batch_size)

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
            shape=(263,), # This value will need to change as features are introduced
            dtype=np.float32
        )
        # Action space: -1 (discharge), 0 (do nothing), 1 (charge)
        self.action_space = spaces.Discrete(3)

    def reset(self):
        # Reset environment state
        self.battery_status = np.ones(self.batch_size)
        self.cash_balance = np.zeros(self.batch_size) + 1000.0
        self.current_step = 0
        
        self.quarter_charge    = np.zeros(self.batch_size)
        self.quarter_discharge = np.zeros(self.batch_size)
        
        self.quarter_feed   = np.zeros(self.batch_size)
        self.quarter_take   = np.zeros(self.batch_size)
        self.quarter_mid    = np.zeros(self.batch_size)
        self.quarter_state  = np.zeros(self.batch_size)

        # need to unwrap the iteration from the dataloader
        self.episode =  next(iter(self.data))[0].to(self.device)
        self.episode = self.episode.permute(0,2,1).contiguous()

        # we process the encoded historical context every 15 minute/ every block
        num_rows_to_process = (self.window_size - self.encoder_dimmension)//15

        all_column_names = ["imbalance_feed_price",
                   "imbalance_take_price",
                   "mid_price",
                   "state"]
        
        num_columns = len(all_column_names)

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
                # we only look at historical context every 15 minutes, each quarter hour
                # block uses same context
                rows_transformed = [batch[i*15:i*15 + self.encoder_dimmension] for i in range(num_rows_to_process)]
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
            batch_size = squeezed_tensor.shape[0]//1  # divide into a size that fits into memory if needed

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

    def step(self, actions):
        """
        Perform one time step in the batch environments.
        
        Parameters:
        actions: numpy array of shape (batch_size,)
            -1: Discharge battery
            0: Do nothing
            1: Charge battery
        
        Returns:
        tuple: (observations, rewards, dones, infos)
        """
        # Mapping default action range from : 
        #   0->-1, 1->0, 2->1, 
        #   it needs to be done here for gymnasium compatability
        actions = actions - 1 
        # charge_states: numpy array of shape (batch_size,)
        # The charge states for each environment in the batch
        charge_states = self.quarter_state
        if torch.is_tensor(actions):
            actions = actions.cpu().numpy()
        if torch.is_tensor(charge_states):
            charge_states = charge_states.cpu().numpy()
        
        # Initialize arrays
        rewards = np.zeros(self.batch_size, dtype=np.float32)
        charge_changes = np.zeros(self.batch_size, dtype=np.float32)
        
        # Calculate potential charge changes for all environments
        max_charge_change = np.minimum(self.charge_speed / 60.0, self.battery_capacity * (1 - self.battery_limit) - self.battery_status)
        
        # Ensure no negative charge changes
        charge_changes = np.maximum(max_charge_change, 0)
        
        # Handle charging (action == 1)
        charging_mask = (actions == 1)
        if np.any(charging_mask):
            self.quarter_charge[charging_mask] += charge_changes[charging_mask]
        
        # Handle discharging (action == -1)
        discharging_mask = (actions == -1)
        if np.any(discharging_mask):
            self.quarter_discharge[discharging_mask] += charge_changes[discharging_mask]
        
        # Apply cycle costs for active batteries (charging or discharging)
        active_mask = (actions != 0)
        if np.any(active_mask):
            cycle_costs = -(self.charge_speed / 60.0) * self.cycle_cost
            rewards[active_mask] = cycle_costs
            self.cash_balance[active_mask] += cycle_costs
        
        # Handle end of 15-minute period
        if self.current_step % 15 == 0:
            # Get final charge states for the period
            final_states = charge_states
            
            # Calculate rewards for the quarter
            charging_prices = self._get_price(final_states, 1)
            discharging_prices = self._get_price(final_states, -1)
            
            quarter_rewards = (-self.quarter_charge * charging_prices + self.quarter_discharge * discharging_prices)
            
            # Update cash balances and rewards
            self.cash_balance += quarter_rewards
            rewards += quarter_rewards
            
            # Reset quarter accumulators
            self.quarter_charge.fill(0)
            self.quarter_discharge.fill(0)
        
        # Update time step (same for all environments)
        self.current_step += 1
        
        # Check if episodes are done
        dones = np.full( self.batch_size, self.current_step >= self.window_size - self.encoder_dimmension - 1, dtype=bool)
        
        # Get the next observations (already batched)
        obs = self._get_observation()
        
        # Create truncated array (required for Gymnasium)
        truncated = np.zeros(self.batch_size, dtype=bool)
        
        # Additional info dictionary
        infos = {
            'quarter_charge': self.quarter_charge.copy(),
            'quarter_discharge': self.quarter_discharge.copy(),
            'battery_status': self.battery_status.copy(),
            'cash_balance': self.cash_balance.copy()
        }
        
        # Handle Gymnasium's step API requirements
        if self.batch_size == 1:
            return (
                obs[0] if isinstance(obs, np.ndarray) else obs,
                float(rewards[0]),
                bool(dones[0]),
                bool(truncated[0]),
                {k: v[0] if isinstance(v, np.ndarray) else v for k, v in infos.items()}
            )
        else:
            return obs, rewards, dones, truncated, infos

    def _get_price(self, reg_states, actions):
        """
        Get the current electricity market prices for a batch of states and actions.
        
        Parameters:
        reg_states: numpy array or torch tensor of shape (batch_size,)
            Regulation states for each environment
            0: no regulation (use mid price)
            -1: down regulation (use take price)
            1: up regulation (use feed price)
            2: both up and down regulation (compare with mid price)
        actions: numpy array or torch tensor of shape (batch_size,)
            -1: Discharge/Selling
            0: Do nothing
            1: Charge/Buying
        
        Returns:
            numpy array of shape (batch_size,) with prices for each environment
        """
        if isinstance(actions, int):
            # Create a new array with the same shape as reg_states
            actions = np.full_like(reg_states, actions)

        # Convert inputs to numpy if they're torch tensors
        if torch.is_tensor(reg_states):
            reg_states = reg_states.cpu().numpy()
        if torch.is_tensor(actions):
            actions = actions.cpu().numpy()
            
        # Initialize prices array
        prices = np.zeros_like(reg_states, dtype=np.float32)
        
        # Mask for active transactions (action != 0)
        active_mask = (actions != 0)
        
        if not np.any(active_mask):
            return prices  # Early return if all rows are action 0
        
        # Create masks for different regulation states
        no_reg_mask = (reg_states == 0) & active_mask
        down_reg_mask = (reg_states == -1) & active_mask
        up_reg_mask = (reg_states == 1) & active_mask
        both_reg_mask = (reg_states == 2) & active_mask
        
        # Create masks for buying/selling actions
        buying_mask = (actions == 1)
        selling_mask = (actions == -1)
        
        # Handle no regulation state (state 0)
        if no_reg_mask.any():

            check_mask = no_reg_mask & buying_mask
            if check_mask.any():
                prices[no_reg_mask & buying_mask] = -1 * self.quarter_mid

            check_mask = no_reg_mask & selling_mask
            if check_mask.any():
                prices[no_reg_mask & selling_mask] = self.quarter_mid
        
        # Handle down regulation state (state -1)
        if down_reg_mask.any():

            check_mask = down_reg_mask & buying_mask
            if check_mask.any():
                prices[down_reg_mask & buying_mask] = -1 * self.quarter_take

            check_mask = down_reg_mask & selling_mask
            if check_mask.any():
                prices[down_reg_mask & selling_mask] = self.quarter_take
        
        # Handle up regulation state (state 1)
        if up_reg_mask.any():
            
            check_mask = up_reg_mask & buying_mask
            if check_mask.any():
                prices[up_reg_mask & buying_mask] = -1 * self.quarter_feed
            
            check_mask = up_reg_mask & selling_mask
            if check_mask.any():
                prices[up_reg_mask & selling_mask] = self.quarter_feed
        
        # Handle both regulations state (state 2)
        if both_reg_mask.any():
            # For buying in state 2
            both_reg_buying = both_reg_mask & buying_mask
            if both_reg_buying.any():
                prices[both_reg_buying] = np.where(
                    self.quarter_feed >= self.quarter_mid,
                    -1 * self.quarter_feed,
                    -1 * self.quarter_mid
                )[both_reg_buying]
            
            # For selling in state 2
            both_reg_selling = both_reg_mask & selling_mask
            if both_reg_selling.any():
                prices[both_reg_selling] = np.where(
                    self.quarter_take <= self.quarter_mid,
                    self.quarter_take,
                    self.quarter_mid
                )[both_reg_selling]
        
        # Check for invalid regulation states
        invalid_mask = ~(no_reg_mask | down_reg_mask | up_reg_mask | both_reg_mask) & active_mask
        if invalid_mask.any():
            raise ValueError(f"Invalid regulation states detected: {reg_states[invalid_mask]}")
        
        return prices

    def _get_observation(self):
        """Get the current state of the environment."""

        historical_data = self.episode[:,:,self.encoder_dimmension+self.current_step]        
        # Compress historical data with the autoencoder
         
          # Assuming autoencoder outputs 40 features
        

        if self.current_step % 15 == 0:
            # a new 15 minute block
            self.current_context = self.encoded_episode[:,self.current_step//15,:]
            self.quarter_feed = np.zeros(self.batch_size)
            self.quarter_take = np.zeros(self.batch_size)
            self.quarter_mid = np.zeros(self.batch_size)
            self.quarter_state = np.zeros(self.batch_size)            
        
            self.quarter_charge    = np.zeros(self.batch_size)
            self.quarter_discharge = np.zeros(self.batch_size)
        
        # store step, taken straight from each column respectively

        # THE INDEXES WILL BE WRONG IF THE UNDERLYING DATA COLUMN ORDER IS CHANGED
        # update the min feed, confusing as it may be it is the lower low_take_price column
        self.quarter_feed = np.minimum(self.quarter_feed, historical_data[:,-8].cpu().numpy())
        # update the maximum take, confusing as it may be it is the highest high_feed_price column
        self.quarter_take = np.maximum(self.quarter_take, historical_data[:,-9].cpu().numpy())
        # update the mid price, taken at face value
        self.quarter_mid = historical_data[:,-7].cpu().numpy()

        # logic for infering state during quarter hour
        mask_unchangeable = self.quarter_state == 2
    
        # Create masks for each condition
        mask_both_active = (self.quarter_feed != 0.0) & (self.quarter_take != 0.0)
        mask_take_only = (self.quarter_feed == 0.0) & (self.quarter_take != 0.0)
        mask_feed_only = (self.quarter_feed != 0.0) & (self.quarter_take == 0.0)
        mask_none_active = (self.quarter_feed == 0.0) & (self.quarter_take == 0.0)
        
        # Create new state array
        new_state = np.where(mask_both_active, 2,
                        np.where(mask_take_only, -1,
                            np.where(mask_feed_only, 1,
                                np.where(mask_none_active, 0,
                                    self.quarter_state)
                                )
                            )
                        )
        
        # Keep original value where state was 2
        self.quarter_state = np.where(mask_unchangeable, self.quarter_state, new_state)

        # Create observation
        #avg_price = self.data[start_idx:self.current_step, 0].mean().item() if self.current_step > 0 else 0.0
        #avg_price = self.average_price
        obs = torch.cat([
            self.current_context,
            torch.tensor(self.battery_status, device=self.device).view(4,1),
            torch.tensor(self.cash_balance, device=self.device).view(4,1),
            torch.tensor(self.quarter_charge, device=self.device).view(4,1),
            torch.tensor(self.quarter_discharge, device=self.device).view(4,1),
            torch.tensor(self.quarter_feed, device=self.device).view(4,1),
            torch.tensor(self.quarter_take, device=self.device).view(4,1),
            torch.tensor(self.quarter_mid, device=self.device).view(4,1)
        ], dim=1)

        return obs.detach().cpu().numpy() # Return as NumPy array for Gym compatibility

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

    
    device_used = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'The device being used is {device_used}')

    env = BatteryMarketEnv(csv_path="final-imbalance-data-training.csv",autoencoder_model=combined_autoencoder,device=device_used)
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
    agent = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape, device=device_used)
    #agent.load_models()
    agent.memory.clear_memory()

    n_games = 1000
    agent.actor.train()
    agent.critic.train()
    figure_file = 'plots/' + env_name + '-' + str(datetime.now().strftime("%Y-%m-%d"))+'.png'

    best_score = 1000
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
            #next_observation = np.round(next_observation, 2)
            n_steps += 1
            
            #if (np.abs(next_observation[0])*np.abs(next_observation[1])/(1.2*0.07))*2 > highest:
            #    highest = (np.abs(next_observation[0])*np.abs(next_observation[1])/(1.2*0.07))*2
            #    reward += highest

            #if done:
            #    print("It managed to finish")
            #    reward += 500
            #if truncated and 0:
            #    print(f'It did not manage to finish, got truncated at step {n_steps}')
                
            
            score += reward
            for index in range(batch_size):
                action_to_save = action[index].unsqueeze(0)
                prob_to_save = prob[index].unsqueeze(0)
                val_to_save = val[index]
                reward_to_save = torch.tensor(reward[index], device=device_used).unsqueeze(0)
                done_to_save = torch.tensor(done[index], device=device_used).unsqueeze(0)

                agent.remember(observation[index], action_to_save, prob_to_save, val_to_save, reward_to_save, done_to_save)
            
            if n_steps % N == 0 or done.any():# or truncated:
                agent.learn()
                learn_iters += 1
            if isinstance(next_observation, tuple):
                next_observation = next_observation[0]
            observation = next_observation

            if done.any():# or truncated:
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


