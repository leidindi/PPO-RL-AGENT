import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import gymnasium as gym

class A2C(nn.Module):
    """
    (Synchronous) Advantage Actor-Critic agent class

    Args:
        n_features: The number of features of the input state.
        n_actions: The number of actions the agent can take.
        device: The device to run the computations on (running on a GPU might be quicker for larger Neural Nets,
                for this code CPU is totally fine).
        critic_lr: The learning rate for the critic network (should usually be larger than the actor_lr).
        actor_lr: The learning rate for the actor network.
        n_envs: The number of environments that run in parallel (on multiple CPUs) to collect experiences.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        device: torch.device,
        critic_lr: float,
        actor_lr: float,
        n_envs: int,
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()
        self.device = device
        self.n_envs = n_envs

        critic_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # estimate V(s)
        ]

        actor_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(
                32, n_actions
            ),  # estimate action logits (will be fed into a softmax later)
        ]

        # define actor and critic networks
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        self.actor = nn.Sequential(*actor_layers).to(self.device)

        # define optimizers for actor and critic
        self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=actor_lr)

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the networks.

        Args:
            x: A batched vector of states.

        Returns:
            state_values: A tensor with the state values, with shape [n_envs,].
            action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
        """
        x = torch.Tensor(x).to(self.device)
        state_values = self.critic(x)  # shape: [n_envs,]
        action_logits_vec = self.actor(x)  # shape: [n_envs, n_actions]
        return (state_values, action_logits_vec)

    def select_action(
        self, x: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of the chosen actions and the log-probs of those actions.

        Args:
            x: A batched vector of states.

        Returns:
            actions: A tensor with the actions, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
            state_values: A tensor with the state values, with shape [n_steps_per_update, n_envs].
        """
        state_values, action_logits = self.forward(x)
        action_pd = torch.distributions.Categorical(
            logits=action_logits
        )  # implicitly uses softmax
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()
        return (actions, action_log_probs, state_values, entropy)

    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lam: float,
        ent_coef: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
        using Generalized Advantage Estimation (GAE) to compute the advantages (https://arxiv.org/abs/1506.02438).

        Args:
            rewards: A tensor with the rewards for each time step in the episode, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions taken at each time step in the episode, with shape [n_steps_per_update, n_envs].
            value_preds: A tensor with the state value predictions for each time step in the episode, with shape [n_steps_per_update, n_envs].
            masks: A tensor with the masks for each time step in the episode, with shape [n_steps_per_update, n_envs].
            gamma: The discount factor.
            lam: The GAE hyperparameter. (lam=1 corresponds to Monte-Carlo sampling with high variance and no bias,
                                          and lam=0 corresponds to normal TD-Learning that has a low variance but is biased
                                          because the estimates are generated by a Neural Net).
            device: The device to run the computations on (e.g. CPU or GPU).

        Returns:
            critic_loss: The critic loss for the minibatch.
            actor_loss: The actor loss for the minibatch.
        """
        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device=device)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration
        actor_loss = (
            -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
        )
        return (critic_loss, actor_loss)

    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        """
        Updates the parameters of the actor and critic networks.

        Args:
            critic_loss: The critic loss.
            actor_loss: The actor loss.
        """
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()


# TODO: Different A2C implementation: https://github.com/lcswillems/rl-starter-files/blob/master/model.py


if __name__ == "__main__":
    # Setup of the Environments
    n_envs = 10
    n_updates = 1000
    n_steps_per_update = 128
    randomize_domain = True

    # Setup of the A2C agent
    gamma = 0.999
    lam = 0.95
    ent_coef = 0.01
    actor_lr = 0.001
    critic_lr = 0.005

    # Important to note is that the actor has a slower learning rate so that
    # value targets become more stationariy and are easier to estimate for the critic


    # Envionrment setup
    # Setting up multiple Environments with random parameter initialization
    # Using Normal distribution with standard parameterization and arbitrary SD (scale)
    # Because 3 is a low number of Environments this will likely create overfitting
    # Either using more random environments or remake the environments after x sampling phases
    # To generate a new set of "random" environments
    # TODO: Set a seed for the random state for reproducability


    if randomize_domain:
        envs = gym.vector.AsyncVectorEnv(
            [
                lambda: gym.make(
                    "LunarLander-v2",
                    gravity=np.clip(
                        np.random.normal(loc=-10.0, scale=1.0), a_min=-11.99, a_max=-0.01
                    ),
                    enable_wind=np.random.choice([True, False]),
                    wind_power=np.clip(
                        np.random.normal(loc=15.0, scale=1.0), a_min=0.01, a_max=19.99
                    ),
                    turbulence_power=np.clip(
                        np.random.normal(loc=1.5, scale=0.5), a_min=0.01, a_max=1.99
                    ),
                    max_episode_steps=600,
                )
                for i in range(n_envs)
            ]
        )
    else:
        envs = gym.vector.make("LunarLander-v2", num_envs=n_envs, max_episode_steps = 600)


    obs_shape = envs.single_observation_space.shape[0]
    action_shape = envs.single_action_space.n

    # Device for trainig
    use_cuda = False
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Agent init
    agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs)

    """ 
    Training the A2C Agent

    Using the RecordEpisodeStatistics wrapper in order to retrieve information and saving results from each epoch for plotting
    """

    envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=n_envs * n_updates)

    critic_losses = []
    actor_losses = []
    entropies = []

    # tqdm for training progress bar
    for sample_phase in tqdm(range(n_updates)):
        # no need to reset the envs, just continue playing
        # until the episode is over and reset automatically

        # reset the experiment lists of an episode
        ep_value_preds = torch.zeros(n_steps_per_update, n_envs, device=device)
        ep_rewards = torch.zeros(n_steps_per_update, n_envs, device=device)
        ep_action_log_probs = torch.zeros(n_steps_per_update, n_envs, device=device)
        masks = torch.zeros(n_steps_per_update, n_envs, device=device)

        # get initial state of all environments at the start
        if sample_phase == 0:
            states, info = envs_wrapper.reset(seed=42)
        
        # play n steps in parallel environment to collect data
        for step in range(n_steps_per_update):
            # Select action using the state as input for the agent
            actions, action_log_probs, state_value_preds, entropy = agent.select_action(
                states
            )

            # Perform the action and get a new state & rewards for the action
            states, rewards, terminated, truncated, infos = envs_wrapper.step(
                actions.cpu().numpy()
            )

            ep_value_preds[step] = torch.squeeze(state_value_preds)
            ep_rewards[step] = torch.tensor(rewards, device=device)
            ep_action_log_probs[step] = action_log_probs

            # add a mask for return calc later
            # for each env the mask is 1 if the episode is ongoing and 0 if terminated
            masks[step] = torch.tensor([not term for term in terminated])

            # calculate the losses for the actor and the critic
            critic_loss, actor_loss = agent.get_losses(
                ep_rewards,
                ep_action_log_probs,
                ep_value_preds,
                entropy,
                masks,
                gamma,
                lam,
                ent_coef,
                device,
            )

            # update the cator and the critic networkds
            agent.update_parameters(critic_loss, actor_loss)

            # log the losses and entropy
            critic_losses.append(critic_loss.detach().cpu().numpy())
            actor_losses.append(actor_loss.detach().cpu().numpy())
            entropies.append(entropy.detach().cpu().numpy())

    """ Plotting the training results """

    # %matplotlib inline

    rolling_length = 20
    fig, axs = plt.subplots(nrows=2, ncolcs=2, figsize=(12, 5))
    fig.subtitle(
        f"Training plots for {agent.__class__.__name__} in the LunarLander-v2 environment \n \
            (n_envs)={n_envs}, n_steps_per_update={n_steps_per_update}, randomize_domain={randomize_domain})"
    )

    # episode return
    axs[0][0].set_title("Episode Returns")
    episode_returns_moving_average = (
        np.convolve(
            np.array(envs_wrapper.return_queue).flatten(),
            np.ones(rolling_length),
            mode="valid",
        )
        / rolling_length
    )
    axs[0][0].plot(
        np.arange(len(episode_returns_moving_average)) / n_envs,
        episode_returns_moving_average,
    )
    axs[0][0].set_xlabel("Number of episodes")

    # entropy
    axs[1][0].set_title("Entropy")
    entropy_moving_average = (
        np.convolve(np.array(entropies), np.ones(rolling_length), mode="valid")
        / rolling_length
    )
    axs[1][0].plot(entropy_moving_average)
    axs[1][0].set_xlabel("Number of updates")


    # critic loss
    axs[0][1].set_title("Critic Loss")
    critic_losses_moving_average = (
        np.convolve(
            np.array(critic_losses).flatten(), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[0][1].plot(critic_losses_moving_average)
    axs[0][1].set_xlabel("Number of updates")


    # actor loss
    axs[1][1].set_title("Actor Loss")
    actor_losses_moving_average = (
        np.convolve(np.array(actor_losses).flatten(), np.ones(rolling_length), mode="valid")
        / rolling_length
    )
    axs[1][1].plot(actor_losses_moving_average)
    axs[1][1].set_xlabel("Number of updates")

    plt.tight_layout()
    plt.show()