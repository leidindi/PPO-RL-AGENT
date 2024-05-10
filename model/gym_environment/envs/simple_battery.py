import numpy as np
import os
import pandas as pd
import math

import gymnasium as gym
from gymnasium import spaces

# Load data
cur_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(cur_path, "../../data/backend/2023/train_imb.csv")
imb = pd.read_csv(data_path)


class SimpleBatteryEnv(gym.Env):
    metadata = {"render_modes": ['human', 'rgb_array']}

    def __init__(self, days=1, predict=False, day_offset=0, charge_penalty_mwh=8.0, seed=None):
        super().reset(seed=seed)
        super(SimpleBatteryEnv, self).__init__()

        # TODO: max amount of cycles per year\
        # aangehouden in baseline is 350
        # mag naar 500

        # Battery Capacity in kWh
        self.capacity = 2000.0

        # Charge rate in kilo Watt per hour = kWh
        self.charge_rate = 1000.0

        # Euro per mwh
        self.charge_penalty_mwh = charge_penalty_mwh
        self.charge_penalty_kwh = self.charge_penalty_mwh / 1000.0

        # kilo Watt hour per minute
        self.charge_per_minute = self.charge_rate / 60.0

        self.battery_change = 0

        # Imbalance data               
        self.imb = imb

        # Max energy price
        energy_take_max = self.imb['imbalance_take_price'].max()
        energy_feed_max = self.imb['imbalance_feed_price'].max()
        mid_price_max = self.imb['mid_price'].max()

        # Minimum energy price
        energy_take_min = self.imb['imbalance_take_price'].min()
        energy_feed_min = self.imb['imbalance_feed_price'].min()
        mid_price_min = self.imb['mid_price'].min()


        # Observation sapce is the opservations made / information
        # Battery charge is in kilo Wh
        # Energy price is in euro per mega watt hour
        # TODO: Regulation state moet er uit, wordt anders berekend
        self.observation_space = spaces.Dict (
            spaces = {
                "battery_charge": spaces.Box(low=self.capacity * 0.1, high=self.capacity * 0.9, shape=(1,), dtype=float),
                "energy_take_price": spaces.Box(low=energy_take_min, high=energy_take_max, shape=(1,), dtype=float),
                "energy_feed_price": spaces.Box(low=energy_feed_min, high=energy_feed_max, shape=(1,), dtype=float),
                "mid_price": spaces.Box(low=mid_price_min, high=mid_price_max, shape=(1,), dtype=float),
                # "regulation_state": spaces.Discrete(4, start=-1),
                "month": spaces.Box(low=1, high=12, shape=(1,), dtype=int),
                "day_of_week": spaces.Box(low=0, high=6, shape=(1,), dtype=int),
                "hour_of_day": spaces.Box(low=0, high=23, shape=(1,), dtype=int),
            }
        )

        # Charge, Discharge, Nothing
        self._action_to_charge = {
            0 : self.charge_per_minute,
            1 : -self.charge_per_minute,
            2 : 0
        }

        self.count = 0
        self.minutes_per_day = 1440
        self.days = days
        self.day_offset = day_offset
        self.start_offset = self.day_offset * self.minutes_per_day
        self.predict = predict

        self.max_count = self.minutes_per_day * self.days

        self.best_feed_price = 0
        self.best_take_price = 0
        self.check_mid = True

        # Action space encoded by Discrete numbers
        # "charge" "discharge" "nothing"
        self.action_space = spaces.Discrete(3)  


    # This method will generate the observations from the environment space
    def _get_obs(self):
        i = self.start_minute + self.count
        reg_state = self.imb['imbalance_regulation_state'][i]
        row = self.imb.iloc[i]

        if self.count % 15 == 0:
            self.best_feed_price = float('-inf')
            self.best_take_price = float('inf')
            self.check_mid = True

        low_take_price = row['low_take_price']
        mid_price = row['mid_price']
        high_feed_price = row['high_feed_price']


        if reg_state == 0:
            self.current_state['battery_charge'] = np.array([self._battery_charge], dtype=float)
            self.current_state['energy_take_price'] = np.array([mid_price], dtype=float)
            self.current_state['energy_feed_price'] = np.array([mid_price], dtype=float)
            self.current_state['mid_price'] = np.array([mid_price], dtype=float)
            self.current_state['month'] = np.array([row['month']], dtype=int)
            self.current_state['day_of_week'] = np.array([row['day_of_week']], dtype=int)
            self.current_state['hour_of_day'] = np.array([row['hour_of_day']], dtype=int)

            return self.current_state
        elif reg_state == -1:
            if pd.isnull(low_take_price) and self.check_mid:
                self.best_take_price = min(mid_price, self.best_take_price)
            elif not pd.isnull(low_take_price):
                if self.check_mid:
                    self.best_take_price = float('inf')
                self.best_take_price = min(low_take_price, self.best_take_price)
                self.check_mid = False
            self.best_feed_price = self.best_take_price

        elif reg_state == 1:
            if pd.isnull(high_feed_price) and self.check_mid:
                self.best_feed_price = max(mid_price, self.best_feed_price)
            elif not pd.isnull(high_feed_price):
                if self.check_mid:
                    self.best_feed_price = float('-inf')
                self.best_feed_price = max(high_feed_price, self.best_feed_price)
                self.check_mid = False
            self.best_take_price = self.best_feed_price

        elif reg_state == 2:
            if np.isnan(high_feed_price):
                self.best_feed_price = max(mid_price, self.best_feed_price)
            else:
                self.best_feed_price = max(high_feed_price, mid_price, self.best_feed_price)
            if np.isnan(low_take_price):
                self.best_take_price = min(mid_price, self.best_take_price)
            else:
                self.best_take_price = min(low_take_price, mid_price, self.best_take_price)

        self.current_state['battery_charge'] = np.array([self._battery_charge], dtype=float)
        self.current_state['energy_take_price'] = np.array([self.best_take_price], dtype=float)
        self.current_state['energy_feed_price'] = np.array([self.best_feed_price], dtype=float)
        self.current_state['mid_price'] = np.array([mid_price], dtype=float)
        # self.current_state['regulation_state'] = np.int64(reg_state)
        self.current_state['month'] = np.array([row['month']], dtype=int)
        self.current_state['day_of_week'] = np.array([row['day_of_week']], dtype=int)
        self.current_state['hour_of_day'] = np.array([row['hour_of_day']], dtype=int)

        return self.current_state
    
    # This method will generate the manhattan distance as extra information
    # Individual reward term should be defined here
    def _get_info(self):
        charged = False
        if self.battery_change != 0:
            charged = True
        return {'imb': self.imb.iloc[self.count+self.start_minute], "current_state": self.current_state, "charged": charged}
        
    # how many kW charged in the last minute
    def _get_charged_minute(self, charge_per_minute):
        if self._battery_charge + charge_per_minute > self.capacity * 0.9:
            return (self.capacity * 0.9) - self._battery_charge
        elif self._battery_charge + charge_per_minute < self.capacity * 0.1:
            return - (self._battery_charge - (self.capacity * 0.1))
        else:
            return charge_per_minute

    def _get_reward(self, charge_per_minute):
        take_price = self.current_state['energy_take_price'][0]
        feed_price = self.current_state['energy_feed_price'][0]
        mid_price = self.current_state['mid_price'][0]

        if charge_per_minute > 0:
            price_per_kW = (take_price-mid_price) / 1000.0
        elif charge_per_minute < 0:
            price_per_kW = (feed_price-mid_price) / 1000.0
        else:
            price_per_kW = 0

        reward_per_minute = -(charge_per_minute * price_per_kW) - (self.charge_penalty_kwh * np.abs(self.charge_per_minute))

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
        if self.predict:
            self.start_minute = self.start_offset
        else:
        # # Training
            self.start_minute = self.np_random.integers(0, math.floor((self.imb.shape[0] - (self.days*self.minutes_per_day))/15)) * 15
            assert self.start_minute % 15 == 0

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
        self.battery_change = self._get_charged_minute(self._action_to_charge[action])
        self._battery_charge = self._battery_charge + self.battery_change

        # Check if episode is done
        terminated = self.count >= self.max_count
        truncated = self.count >= self.max_count

        observation = self._get_obs()
        reward = self._get_reward(self.battery_change)  # Minute price
        info = self._get_info()

        self.count+=1
        return observation, reward, terminated, truncated, info
