from gymnasium.envs.registration import register
import gymnasium

register(
    id="gym_environment/GridWorld",
    entry_point="gym_environment.envs:GridWorldEnv",
    max_episode_steps=300
)

register(
    id="gym_environment/SimpleBattery",
    entry_point="gym_environment.envs:SimpleBatteryEnv",
    max_episode_steps=3000000000
)