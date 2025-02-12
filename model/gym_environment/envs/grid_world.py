import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size # Size of Square grid
        self.window_size = 512 # PyGame window Size

        # Observation sapce is the opservations made / information
        # available to the agent
        # In this case it is location encoded as {0, ..., `size`}^2

        self.observation_space = spaces.Dict (
            spaces = {
                "agent": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=int),
                "bonus": spaces.Box(low=-1, high=size - 1, shape=(2,), dtype=int),
                "collected": spaces.Discrete(2)
            }
        )

        # Action space encoded by Discrete numbers
        # "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        Map the actions to a dictionary for the direction to walk in if 
        the action is taken
        """

        self._action_to_direction = {
            0: np.array([1,0]),
            1: np.array([0,1]),
            2: np.array([-1,0]),
            3: np.array([0,-1]),
        }

        self.count = 0.0
        self.max_count = 100.0
        self.collected = False
        self.first_time = True

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        human-rendering will confine self.window to a refernce to
        the window that is drawn on.
        In order to ensure the correct frametime `self.clock` is used in human-mnode
        They are set to `None` until human-render is done once
        """

        self.window = None
        self.clock= None

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
    
    def _get_reward(self, terminated):
        if np.array_equal(self._agent_location, self._bonus_location) and self.first_time:
            reward = 1
            self.first_time = False
        elif terminated:
            reward = 1
        else:
            reward = -0.1
        return reward
    
    """
    Reset is called to create a new episode

    Seed should be set in a Research setting for replication

    Needs to include tuple of initial observation and other info.

    Both _get_obs and _get_info can be used for that
    """
    def reset(self, seed=None, options=None):
        # Setting the seed of the np_random
        super().reset(seed=seed)
        self.count=0.0
        self.collected=False
        self.first_time=True
        # Init the state
        self.current_state = self.observation_space.sample()

        # Random position for the agent at the start
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Setting the target location randomly that does not overlap with the Agents location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        self._bonus_location = self._target_location
        while np.array_equal(self._bonus_location, self._target_location):
            self._bonus_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
        
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
        # Map the action to the direction dict.
        direction = self._action_to_direction[action]
        # Use 'np.clip' to not leave the grid 
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size-1
        )

        self.count+=1
        
        # Check if reward is collected
        if np.array_equal(self._agent_location, self._bonus_location):
            self.collected = True

        # Check if episode is done
        terminated = np.array_equal(self._agent_location, self._target_location) and self.collected
        truncated = self.count > self.max_count
        reward = self._get_reward(terminated) # Binary Sparse Rewards
        observation = self._get_obs()
        info = self._get_info()

            

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    
    # Using PyGame to render the game. This is not needed for the final project
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )

        color = (255,0, 0) if self.collected else (255, 165, 0)
        # Drawing the target
        pygame.draw.rect(
            canvas,
            color,
            pygame.Rect(
                (pix_square_size * self._target_location),
                (pix_square_size, pix_square_size)
            ),
        )

        # Drawing the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            tuple(e + pix_square_size/2 for e in (pix_square_size * self._agent_location)),
            pix_square_size / 3,
        )

        # Drawing the bonus
        if not self.collected:
            pygame.draw.circle(
                canvas,
                (0, 255, 0),
                tuple(e + pix_square_size/2 for e in (pix_square_size * self._bonus_location)),
                pix_square_size / 3,
            )

        # Adding gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width = 3,
            )
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width = 3,
            )
        if self.render_mode == "human":
            # Copy the canvas to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # Set the framerate by adding a delay to the frame rendering
            self.clock.tick(self.metadata["render_fps"])
        else: # `rgb_array`
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes = (1,0,2)
            )
        

    # Close any open resources like pygame / files accessed by the environment
    def close(self):
        if self.window is not None:
            pygame.dispaly.quit()
            pygame.quit()