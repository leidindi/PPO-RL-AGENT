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
    def __init__(self, csv_path = "final-imbalance-data-training.csv", autoencoder_model = None, batch_size = 4, device="cuda"):
        super(BatteryMarketEnv, self).__init__()
        self.device = device
        # Load market data
        self.csv_headers = pd.read_csv(csv_path, nrows=0).columns.tolist()
        # take out the index and the timestamp
        self.csv_headers = self.csv_headers[2:]

        self.encoder_dimmension = 60*24*1 #( then elongated by 10 because of the interleaving for the encoders))
        # how long each training period is
        self.window_size = 60*24*2
        # constant for overlap between windows, circa 66% overlap if 60*24*6 and 60*24*2
        self.stride = self.window_size//3
        self.batch_size = batch_size

        self.data = autoencoder.load_csv_for_autoencoder(csv_file = csv_path, feature_cols=self.csv_headers
                                                         , window_size = self.window_size, stride = self.stride
                                                         , batch_size = self.batch_size)
        # the first training window iteration is a wrapped list of [object], so it's 
        # unwrapped and thrown to the the device
        self.episode =  next(iter(self.data))[0].to(device)
        # flip axes and align memmory accordingly
        #self.episode = self.episode.permute(0,2,1).contiguous()
        self.current_context = None

        # Environment constants
        self.battery_capacity = 2.0  # MWh
        self.charge_speed = 1.0  # MW (2 hours until full charge)
        self.cycle_cost = 80.0  # Cost per full charge/discharge cycle
        self.battery_limit = 0.1

        # State variables
        self.battery_status  = torch.zeros(self.batch_size, device=self.device) + 1.0  # Initial battery charge level (in MWh)
        self.cash_balance  = torch.zeros(self.batch_size, device=self.device)  # Initial cash balance
        self.current_step = 0  # Track the current time step
        
        self.quarter_charge     = torch.zeros(self.batch_size, device=self.device)
        self.quarter_discharge  = torch.zeros(self.batch_size, device=self.device)
        
        self.quarter_feed    = torch.zeros(self.batch_size, device=self.device)
        self.quarter_take    = torch.zeros(self.batch_size, device=self.device)
        self.quarter_mid     = torch.zeros(self.batch_size, device=self.device)
        self.quarter_state  = torch.zeros(self.batch_size, device=self.device)

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
            shape=(264,), # This value will need to change as features are introduced
            dtype=np.float32
        )
        # Action space: -1 (discharge), 0 (do nothing), 1 (charge)
        self.action_space = spaces.Discrete(3)

    def reset(self):
        # Reset environment state
        self.battery_status = torch.zeros(self.batch_size, device=self.device) + 1 # All ones
        self.cash_balance = torch.zeros(self.batch_size, device=self.device)   # All zeros
        self.current_step = torch.tensor(0, device=self.device)               # Single value, scalar tensor

        self.quarter_charge = torch.zeros(self.batch_size, device=self.device)    # All zeros
        self.quarter_discharge = torch.zeros(self.batch_size, device=self.device) # All zeros

        self.quarter_feed = torch.zeros(self.batch_size, device=self.device)  # All zeros
        self.quarter_take = torch.zeros(self.batch_size, device=self.device)  # All zeros
        self.quarter_mid = torch.zeros(self.batch_size, device=self.device)   # All zeros
        self.quarter_state = torch.zeros(self.batch_size, device=self.device) # All zeros

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
                #print(f'Column name:{col_name} was not recognized')
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
            shape = final_tensor.shape
            del batch_transformed
            del rows_transformed
            # Squeeze along dim=1
            squeezed_tensor = final_tensor.view(-1, shape[2])
            del final_tensor
            # Define the batch size for processing parts of the tensor
            #print(squeezed_tensor.shape)
            batch_size = squeezed_tensor.shape[0]//1  # divide into a size that fits into memmory if needed

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
        self.current_context = self.encoded_episode[:,0]
        del compressed_context
        # Get the initial observation
        return self._get_observation()

    def step(self, actions):
        """
        Perform one time step in the batch environments using PyTorch tensors.
        
        Parameters:
        actions: torch.Tensor of shape (batch_size,)
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
        # The charge states for each environment in the batch
        charge_states = self.quarter_state.to(self.device)
        
        # Initialize arrays
        rewards = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        charge_changes = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        
        # Calculate potential charge changes for all environments
        max_charge_change = torch.minimum(
        torch.tensor(self.charge_speed / 60.0, device=self.device),
        self.battery_capacity * (1 - self.battery_limit) - self.battery_status)
        # Ensure no negative charge changes
        charge_changes = torch.maximum(max_charge_change, torch.tensor(0.0, device=self.device))
    
        
        # Handle charging (action == 1)
        charging_mask = (actions == 1)
        if charging_mask.any():
            self.quarter_charge[charging_mask] += charge_changes[charging_mask]
        
        # Handle discharging (action == -1)
        discharging_mask = (actions == -1)
        if discharging_mask.any():
            self.quarter_discharge[discharging_mask] += charge_changes[discharging_mask]
        
        # Apply cycle costs for active batteries (charging or discharging)
        active_mask = (actions != 0)
        if active_mask.any():
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
            self.quarter_charge.zero_()
            self.quarter_discharge.zero_()
        
        # Update time step (same for all environments)
        self.current_step += 1
        
        # debugger hook to check if episodes are finishing correctly
        if self.encoder_dimmension + self.current_step >= self.window_size - 1:
            pass
        done_condition = self.encoder_dimmension + self.current_step >= self.window_size - 1
        dones = torch.full((self.batch_size,), done_condition, 
                        dtype=torch.bool, device=self.device)
        
        # Get the next observations (already batched)
        obs = self._get_observation()
        
        # Create truncated array (required for Gymnasium)
        truncated = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        # Additional info dictionary
        infos = {
        'quarter_charge': self.quarter_charge.clone().detach(),
        'quarter_discharge': self.quarter_discharge.clone().detach(),
        'battery_status': self.battery_status.clone().detach(),
        'cash_balance': self.cash_balance.clone().detach()
        }
        
        # Handle Gymnasium's step API requirements
        if self.batch_size == 1:
            return (
                obs[0] if isinstance(obs, torch.Tensor) else obs,
                float(rewards[0].item()),
                bool(dones[0].item()),
                bool(truncated[0].item()),
                {k: v[0] if isinstance(v, torch.Tensor) else v for k, v in infos.items()}
            )
        else:
            return obs, rewards, dones, truncated, infos

    def _get_price(self, reg_states, actions):
        """
        Get the current electricity market prices for a batch of states and actions.
        
        Parameters:
        reg_states: torch.Tensor of shape (batch_size,)
            Regulation states for each environment
            0: no regulation (use mid price)
            -1: down regulation (use take price)
            1: up regulation (use feed price)
            2: both up and down regulation (compare with mid price)
        actions: torch.Tensor of shape (batch_size,) or int
            -1: Discharge/Selling
            0: Do nothing
            1: Charge/Buying
        
        Returns:
            torch.Tensor of shape (batch_size,) with prices for each environment
        """
        if not torch.is_tensor(reg_states):
            raise TypeError("reg_states in the get_price function is not a tensor, but should be")

        # Handle scalar action input
        if isinstance(actions, int):
            actions = torch.full_like(reg_states, actions).to(self.device)
        elif isinstance(actions, (float, np.integer, np.floating)):
            actions = torch.full_like(reg_states, float(actions)).to(self.device)

        reg_states = reg_states.to(self.device) 
        # Initialize prices array
        prices = torch.zeros_like(reg_states, dtype=torch.float32, device=self.device)
        
        # Mask for active transactions (action != 0)
        active_mask = (actions != 0)
        
        # Early return if all rows are action 0
        if not active_mask.any():
            return prices  
        
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
                prices[check_mask] = -1 * self.quarter_mid[check_mask]

            check_mask = no_reg_mask & selling_mask
            if check_mask.any():
                prices[check_mask] = self.quarter_mid[check_mask]
        
        # Handle down regulation state (state -1)
        if down_reg_mask.any():

            check_mask = down_reg_mask & buying_mask
            if check_mask.any():
                prices[check_mask] = -1 * self.quarter_take[check_mask]

            check_mask = down_reg_mask & selling_mask
            if check_mask.any():
                prices[check_mask] = self.quarter_take[check_mask]
        
        # Handle up regulation state (state 1)
        if up_reg_mask.any():
            
            check_mask = up_reg_mask & buying_mask
            if check_mask.any():
                prices[check_mask] = -1 * self.quarter_feed[check_mask]
            
            check_mask = up_reg_mask & selling_mask
            if check_mask.any():
                prices[check_mask] = self.quarter_feed[check_mask]
        
        # Handle both regulations state (state 2)
        if both_reg_mask.any():
            # For buying in state 2
            check_mask = both_reg_mask & buying_mask
            if check_mask.any():
                prices[check_mask] = torch.where(
                    self.quarter_feed >= self.quarter_mid,
                    -1 * self.quarter_feed,
                    -1 * self.quarter_mid
                )[check_mask]
            
            # For selling in state 2
            check_mask = both_reg_mask & selling_mask
            if check_mask.any():
                prices[check_mask] = torch.where(
                    self.quarter_take <= self.quarter_mid,
                    self.quarter_take,
                    self.quarter_mid
                )[check_mask]
        
        # Check for invalid regulation states
        bogus_mask = ~(no_reg_mask | down_reg_mask | up_reg_mask | both_reg_mask) & active_mask
        if bogus_mask.any():
            raise ValueError(f"Invalid regulation states detected: {reg_states[bogus_mask]}")
        
        return prices

    def _get_observation(self):
        """Get the current state of the environment."""

        historical_data = self.episode[:,:,self.encoder_dimmension+self.current_step]

        if self.current_step % 15 == 0:
            # a new 15 minute block
            self.current_context = self.encoded_episode[:,self.current_step//15,:]
            self.quarter_feed       = torch.zeros(self.batch_size, device=self.device)
            self.quarter_take       = torch.zeros(self.batch_size, device=self.device)
            self.quarter_mid        = torch.zeros(self.batch_size, device=self.device)
            self.quarter_state      = torch.zeros(self.batch_size, device=self.device)
            self.quarter_charge     = torch.zeros(self.batch_size, device=self.device)
            self.quarter_discharge  = torch.zeros(self.batch_size, device=self.device)
        
        # THE INDEXES WILL BE WRONG IF THE UNDERLYING DATA COLUMN ORDER IS CHANGED
        # Update price values using tensor operations
        # Note: Using historical_data directly as tensor, assuming it's already on correct device

        # the quarter imbalance feed price is the lowest take_price
        self.quarter_feed = torch.minimum(self.quarter_feed, historical_data[:,-8])
        
        # the quarter imbalance take price is the highest feed_price
        self.quarter_take = torch.maximum(self.quarter_take, historical_data[:,-9])

        # mid is taken at face value
        self.quarter_mid = historical_data[:,-7]

        # we also have to ensure that we don't have a look-ahead bias
        # so if we pass on historical data as an observation we need to remove highest/lowest of 
        # a quarter hour that has not passed yet

        # logic for infering state during quarter hour
        mask_unchangeable = self.quarter_state == 2
        
        # Create masks for each condition (these are already batch_size shaped)
        mask_both_active = (self.quarter_feed != 0.0) & (self.quarter_take != 0.0)
        mask_take_only = (self.quarter_feed == 0.0) & (self.quarter_take != 0.0)
        mask_feed_only = (self.quarter_feed != 0.0) & (self.quarter_take == 0.0)
        mask_none_active = (self.quarter_feed == 0.0) & (self.quarter_take == 0.0)
        
        # Create batched state values instead of scalars
        state_both = torch.full((self.batch_size,), 2., device=self.device)
        state_take = torch.full((self.batch_size,), -1., device=self.device)
        state_feed = torch.full((self.batch_size,), 1., device=self.device)
        state_none = torch.full((self.batch_size,), 0., device=self.device)
        
        # Create new state tensor with proper batching
        new_state = torch.where(mask_both_active, state_both,
            torch.where(mask_take_only, state_take,
                torch.where(mask_feed_only, state_feed,
                    torch.where(mask_none_active, state_none, 
                        self.quarter_state
                    )
                )
            )
        )
        
        # Update quarter_state maintaining unchangeable states
        if self.quarter_state.all() == 0:
            self.quarter_state = new_state
        else:
            self.quarter_state = torch.where(mask_unchangeable, self.quarter_state, new_state)

        # Create observation
        #avg_price = self.data[start_idx:self.current_step, 0].mean().item() if self.current_step > 0 else 0.0
        #avg_price = self.average_price
        obs = torch.cat([
            # autoencoder context
            self.current_context,
            
            # entire game variables
            self.battery_status.clone().detach().view(self.batch_size,1),
            self.cash_balance.clone().detach().view(self.batch_size,1),

            # quarter dependent variable
            # for charge            
            self.quarter_charge.clone().detach().view(self.batch_size,1),
            self.quarter_discharge.clone().detach().view(self.batch_size,1),

            # for current prices            
            self.quarter_feed.clone().detach().view(self.batch_size,1),
            self.quarter_take.clone().detach().view(self.batch_size,1),
            self.quarter_mid.clone().detach().view(self.batch_size,1),

            # for current state
            self.quarter_state.clone().detach().view(self.batch_size,1)
            
            #torch.tensor(self.quarter_take, device=self.device).view(self.batch_size,1),
            #torch.tensor(self.quarter_take, device=self.device).view(self.batch_size,1),
            #torch.tensor(self.quarter_take, device=self.device).view(self.batch_size,1),
            #torch.tensor(self.quarter_take, device=self.device).view(self.batch_size,1),
        ], dim=1)
        
        if (self.battery_status/self.battery_capacity > 1).any() or (self.battery_status < 0).any():
            raise ValueError

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

class DualProgress:
    def __init__(self, total_scenarios, steps_per_scenario, bar_length=40):
        """
        Initialize the dual progress bar system.
        
        Args:
            total_scenarios (int): Total number of scenarios to run
            steps_per_scenario (int): Number of steps in each scenario
            bar_length (int): Length of each progress bar visualization
        """
        self.total_scenarios = total_scenarios
        self.steps_per_scenario = steps_per_scenario
        self.bar_length = bar_length
        self.last_scenario_progress = -1
        self.last_overall_progress = -1
        
    def create_bar(self, current, total, bar_length):
        """Helper function to create a progress bar string."""
        progress = int((current / total) * 100)
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        return bar, progress
        
    def update(self, current_scenario, current_step, metrics=None):
        """
        Update both progress bars.
        
        Args:
            current_scenario (int): Current scenario number (0-based)
            current_step (int): Current step within the current scenario
            metrics (dict): Optional dictionary of metrics to display
        """
        # Calculate progress for current scenario
        scenario_bar, scenario_progress = self.create_bar(
            current_step, 
            self.steps_per_scenario, 
            self.bar_length
        )
        
        # Calculate overall progress
        total_steps = self.total_scenarios * self.steps_per_scenario
        current_total_steps = current_scenario * self.steps_per_scenario + current_step
        overall_bar, overall_progress = self.create_bar(
            current_total_steps, 
            total_steps, 
            self.bar_length
        )
        
        # Only update if either progress has changed
        if (scenario_progress != self.last_scenario_progress or 
            overall_progress != self.last_overall_progress):
            
            self.last_scenario_progress = scenario_progress
            self.last_overall_progress = overall_progress
            
            # Create the progress strings
            scenario_str = f'Scenario {current_scenario + 1}/{self.total_scenarios} '
            scenario_str += f'|{scenario_bar}| {scenario_progress}% '
            scenario_str += f'({current_step}/{self.steps_per_scenario})'
            
            overall_str = f'Overall Progress |{overall_bar}| {overall_progress}% '
            overall_str += f'({current_total_steps}/{total_steps})'
            
            # Add metrics if provided
            metrics_str = ''
            if metrics:
                metrics_str = ' | ' + ' | '.join(f'{k}: {v:.2f}' for k, v in metrics.items())
            
            # Print both progress bars (with carriage return for in-place updates)
            print(f'\r{scenario_str}\n{overall_str}{metrics_str}', end='\033[K\033[F', flush=True)
            
            # Print newline if training is complete
            if (current_scenario == self.total_scenarios - 1 and 
                current_step == self.steps_per_scenario):
                print('\n\nTraining Complete!')

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

    batch_size = 35
    env = BatteryMarketEnv(csv_path="final-imbalance-data-training.csv",autoencoder_model=combined_autoencoder,batch_size=batch_size, device=device_used)
    env_name = "Custom"
    n_epochs = 1
    alpha = 0.0001

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
    
    N = 100
    n_games = 10
    agent.actor.train()
    agent.critic.train()
    figure_file = 'plots/' + env_name + '-' + str(datetime.now().strftime("%Y-%m-%d"))+'.png'

    best_score = 0
    score_history = []

    learn_iters = 0
    avg_score = 0
    done = False
    truncated = False

    progress = DualProgress(n_games, env.window_size - env.encoder_dimmension + 1)

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
                reward_to_save = reward[index].clone().detach().unsqueeze(0)
                done_to_save = done[index].clone().detach().unsqueeze(0)

                agent.remember(observation[index], action_to_save, prob_to_save, val_to_save, reward_to_save, done_to_save)
            
            if n_steps % N == 0 or done.any():# or truncated:
                agent.learn()
                learn_iters += 1
            if isinstance(next_observation, tuple):
                next_observation = next_observation[0]
            observation = next_observation

            metrics = {
                'latest reward': torch.mean(reward)
                }
            
            # Update progress bars
            progress.update(i, env.current_step, metrics)

            if done.any():# or truncated:
                break
        
        score_history.extend(score)
        avg_score = torch.mean(score)

        if avg_score > best_score:# and not truncated:
            best_score = avg_score
            agent.save_models()

        # the sample value function does not apply to >2 dimmensional state spaces
        #if i % 50 == 0:
            #sample_value_function(agent,i)

        print('\nepisode', i, 'latest score %.1f' % score_history[-1], 'avg batch scores %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters, '\n')
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)


