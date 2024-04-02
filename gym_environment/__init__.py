from gymnasium.envs.registration import register
import gymnasium

register(
    id="gym_examples/GridWorld",
    entry_point="gym_examples.envs:GridWorldEnv",
    max_episode_steps=300
)