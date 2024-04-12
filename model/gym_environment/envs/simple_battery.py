import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class SimpleBatteryEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self):

        # Observation sapce is the opservations made / information
        # available to the agent
        # In this case it is location encoded as {0, ..., `size`}^2

        self.observation_space = spaces.Dict (
            spaces = {
                "battery_state": spaces.Box(low=10.0, high=90.0, shape=(1,), dtype=float),
                "energy_price": spaces.Box(low=-2000.0, high=2000.0, shape=(1,), dtype=float),
            }
        )

        # Action space encoded by Discrete numbers
        # "charge" "discharge" "nothing"
        self.action_space = spaces.Discrete(3)


    # This method will generate the observations from the environment space
    def _get_obs(self):
        self.current_state['agent'] = np.array(self._agent_location, dtype=int)
        self.current_state['target'] = np.array(self._target_location, dtype=int)
        self.current_state['bonus'] = np.array(self._bonus_location, dtype=int)
        self.current_state['collected'] = 1 if self.collected else 0
        return self.current_state
    
    # This method will generate the manhattan distance as extra information
    # Individual reward term should be defined here
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1,
            )
        }
    
    def _get_reward(self):

        return 1
    
    """
    Reset is called to create a new episode

    Seed should be set in a Research setting for replication

    Needs to include tuple of initial observation and other info.

    Both _get_obs and _get_info can be used for that
    """
    def reset(self, seed=None, options=None):
        # Setting the seed of the np_random
        super().reset(seed=seed)

        # Init the state
        self.current_state = self.observation_space.sample()
        
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
        # Check if episode is done
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = self.count > self.max_count
        reward = self._get_reward() if terminated else -0.1  # Binary Sparse Rewards
        observation = self._get_obs()
        info = self._get_info()

            

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
