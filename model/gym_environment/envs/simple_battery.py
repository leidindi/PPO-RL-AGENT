import numpy as np
import os
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

# Load data
cur_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(cur_path, "../../data/df_imb.csv")
imb = pd.read_csv(data_path)
imb_sub = imb[['dates', 'min_price', 'mid_price', 'max_price', 'imbalance_take_price', 'imbalance_feed_price', 'imbalance_regulation_state']]

class SimpleBatteryEnv(gym.Env):
    metadata = {"render_modes": ['human', 'rgb_array']}

    def __init__(self, days=1, seed=None):
        super().reset(seed=seed)
        super(SimpleBatteryEnv, self).__init__()

        # Battery Capacity in kWh
        self.capacity = 6000.0

        # Charge rate in kilo Watt per hour = kWh
        self.charge_rate = 600.0

        # kilo Watt hour per minute
        self.charge_per_minute = self.charge_rate / 60.0

        # Imbalance data               
        self.imb = imb_sub

        # Max energy price
        energy_take_max = self.imb['imbalance_take_price'].max()
        energy_feed_max = self.imb['imbalance_feed_price'].max()

        # Minimum energy price
        energy_take_min = self.imb['imbalance_take_price'].min()
        energy_feed_min = self.imb['imbalance_feed_price'].min()

        # Observation sapce is the opservations made / information
        # Battery charge is in kilo Wh
        # Energy price is in euro per mega watt hour
        self.observation_space = spaces.Dict (
            spaces = {
                "battery_charge": spaces.Box(low=self.capacity * 0.1, high=self.capacity * 0.9, shape=(1,), dtype=float),
                "energy_take_price": spaces.Box(low=energy_take_min, high=energy_take_max, shape=(1,), dtype=float),
                "energy_feed_price": spaces.Box(low=energy_feed_min, high=energy_feed_max, shape=(1,), dtype=float),
            }
        )

        self._action_to_charge = {
            0 : self.charge_per_minute,
            1 : -self.charge_per_minute,
            2 : 0
        }

        self.start_day = 0
        self.count = 0
        self.minutes_per_day = 1440
        self.days = days
        self.max_count = self.minutes_per_day * self.days

        self.best_feed_price = 0
        self.best_take_price = 0

        # Action space encoded by Discrete numbers
        # "charge" "discharge" "nothing"
        self.action_space = spaces.Discrete(3)  


    # This method will generate the observations from the environment space
    def _get_obs(self):
        i = self.start_minute + self.count
        reg_state = self.imb['imbalance_regulation_state'][i]

        if self.count % 15 == 0:
            self.best_feed_price = float('-inf')
            self.best_take_price = float('inf')

        min_price = self.imb['min_price'][i]
        mid_price = self.imb['mid_price'][i]
        max_price = self.imb['max_price'][i]

        if reg_state == 0:
            self.current_state['battery_charge'] = np.array([self._battery_charge], dtype=float)
            self.current_state['energy_take_price'] = np.array([mid_price], dtype=float)
            self.current_state['energy_feed_price'] = np.array([mid_price], dtype=float)

            return self.current_state
        elif reg_state == -1:
            if np.isnan(min_price) and self.best_take_price == float('inf'):
                self.best_take_price = min(mid_price, self.best_take_price)
            elif not np.isnan(min_price):
                self.best_take_price = min(min_price, self.best_take_price)
            self.best_feed_price = self.best_take_price

        elif reg_state == 1:
            if np.isnan(max_price) and self.best_feed_price == float('-inf'):
                self.best_feed_price = max(mid_price, self.best_feed_price)
            elif not np.isnan(max_price):
                self.best_feed_price = max(max_price, self.best_feed_price)

            self.best_take_price = self.best_feed_price
        elif reg_state == 2:
            if np.isnan(max_price):
                self.best_feed_price = max(mid_price, self.best_feed_price)
            else:
                self.best_feed_price = max(max_price, self.best_feed_price)
            if np.isnan(min_price):
                self.best_take_price = min(mid_price, mid_price, self.best_take_price)
            else:
                self.best_take_price = min(min_price, mid_price, self.best_take_price)

        self.current_state['battery_charge'] = np.array([self._battery_charge], dtype=float)
        self.current_state['energy_take_price'] = np.array([self.best_take_price], dtype=float)
        self.current_state['energy_feed_price'] = np.array([self.best_feed_price], dtype=float)
        return self.current_state
    
    # This method will generate the manhattan distance as extra information
    # Individual reward term should be defined here
    def _get_info(self):
        return {'imb': self.imb.iloc[self.start_minute+self.count]}
        
    # how many kW charged in the last minute
    def _get_charged_minute(self, charge_per_minute):
        if self._battery_charge + charge_per_minute > self.capacity * 0.9:
            return (self.capacity * 0.9) - self._battery_charge
        elif self._battery_charge + charge_per_minute < self.capacity * 0.1:
            return - (self._battery_charge - (self.capacity * 0.1))
        else:
            return charge_per_minute

    def _get_reward(self, charge_per_minute):
        i = self.start_minute + self.count
        if charge_per_minute > 0:
            price_per_kW = self.imb['imbalance_take_price'][i] / 1000.0
        else:
            price_per_kW = self.imb['imbalance_feed_price'][i] / 1000.0
        reward_per_minute = -(charge_per_minute * price_per_kW)
        return reward_per_minute
    
    """
    Reset is called to create a new episode

    Seed should be set in a Research setting for replication

    Needs to include tuple of initial observation and other info.

    Both _get_obs and _get_info can be used for that
    """
    def reset(self, seed=None, options=None):
        # Setting the seed of the np_random
        super().reset(seed=seed)

        self.count=0

        # random start minute
        # self.start_minute = self.np_random.integers(0, self.imb.shape[0] - (self.days * self.minutes_per_day))
        # Make sure that this is always at the start of a 15 minute block otherwise the reward function will break
        self.start_minute = 1305

        # Init the state
        self.current_state = self.observation_space.sample()
        self._battery_charge = self.capacity * 0.5
        # self._battery_charge = self.current_state['battery_charge'][0]
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    

    """
    Step contains the Logic of the environment
    Accepts an action and computes the environment after applying the action

    Return a tuple of (observation, reward, terminated, truncated, info)

    Check if the state is terminal (set done accordingly)

    This case of sparse Binary reward computing it is trivial

    Use _get_obs and _get_info to gather observation and information
    """

    def step(self, action):
        # Determine charge amount
        charge_per_minute = self._get_charged_minute(self._action_to_charge[action])
        self._battery_charge = self._battery_charge + charge_per_minute

        # Check if episode is done
        terminated = self.count >= self.max_count
        truncated = self.count >= self.max_count

        observation = self._get_obs()
        reward = self._get_reward(charge_per_minute)  # Minute price
        info = self._get_info()

        self.count+=1
        return observation, reward, terminated, truncated, info
