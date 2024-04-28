#https://www.youtube.com/watch?v=OcIx_TBu90Q

import gymnasium as gym
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from gymnasium.wrappers import FlattenObservation

class SharedAdam(T.optim.Adam):
	def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
		super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

		for group in self.param_groups:
			for p in group['params']:
				state = self.state[p]
				state['step'] = 0
				state['exp_avg'] = T.zeros_like(p.data)
				state['exp_avg_sq'] = T.zeros_like(p.data)

				state['exp_avg'].share_memory_()
				state['exp_avg_sq'].share_memory_()

class ActorCritic(nn.Module):
	def __init__(self, input_dims, n_actions, gamma=0.99):
		super(ActorCritic, self).__init__()

		self.gamma = gamma

		self.conv_layers = nn.Sequential(
			nn.Conv1d(in_channels=input_dims[0], out_channels=32, kernel_size=1),
			nn.ReLU(),
			nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1),
			nn.ReLU(),
			nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
			nn.ReLU(),
		)

		# Calculate the shape of the output from the convolutional layers
		conv_output_size = self.get_conv_output_size(input_dims)
        
		self.lstm = nn.LSTM(input_size=conv_output_size, hidden_size=64, batch_first=True)

		self.fc_layers = nn.Sequential(
			nn.Linear(64, 128),
			nn.ReLU(),
			nn.Linear(128, n_actions)
		)

		self.v_layers = nn.Sequential(
			nn.Linear(64, 128),
			nn.ReLU(),
			nn.Linear(128, 1)
		)

		# self.pi1 = nn.Linear(*input_dims, 128)  
		# self.v1 = nn.Linear(*input_dims, 128)
		# self.pi = nn.Linear(128, n_actions)
		# self.v = nn.Linear(128, 1)

		self.rewards = []
		self.actions = []
		self.states = []

	def remember(self, state, action, reward):
		self.states.append(state)
		self.actions.append(action)
		self.rewards.append(reward)

	def clear_memory(self):
		self.states = []
		self.actions = []
		self.rewards = []

	def forward(self, state):
		# pi1 = F.relu(self.pi1(state))
		# v1 = F.relu(self.v1(state))

		# pi = self.pi(pi1.squeeze(1))
		# v = self.v(v1.squeeze(1))
		state = state.unsqueeze(2)
	
		# for layer in self.conv_layers:
		# 	state = layer(state)
		# 	print(state.size())
	
		pi = self.conv_layers(state)
		pi = pi.view(pi.size(0), -1)  # Flatten the output for the RNN
		pi, _ = self.lstm(pi.unsqueeze(1))  # Add a time dimension (batch_size, seq_len, input_size)
		pi = pi.squeeze(1)  # Remove the time dimension
		pi = self.fc_layers(pi)

		v = self.conv_layers(state)
		v = v.view(v.size(0), -1)  # Flatten the output for the RNN
		v, _ = self.lstm(v.unsqueeze(1))  # Add a time dimension (batch_size, seq_len, input_size)
		v = v.squeeze(1)  # Remove the time dimension
		v = self.v_layers(v)

		return pi, v
		# return state
  
	def calc_R(self, done):
		states = T.tensor(self.states, dtype=T.float32)
		_, v = self.forward(states)

		# Self the reard to 0 if a terminal state is reached
		R = v[-1]*(1-int(done))

		# Calculate the batch returns in reverse
		batch_return = []
		for reward in self.rewards[::-1]:
			R = reward + self.gamma*R
			batch_return.append(R)

		# Reverse the list a gain to make sure the states are in the correct order again
		batch_return.reverse()
		batch_return = T.tensor(batch_return, dtype=T.float32)

		return batch_return
  
	def calc_loss(self, done):
		states = T.tensor(self.states, dtype=T.float32)
		actions = T.tensor(self.actions, dtype=T.float32)

		returns = self.calc_R(done)

		pi, values = self.forward(states)
		values = values.squeeze()

		critic_loss = (returns - values)**2

		probs = T.softmax(pi, dim=1)
		dist = Categorical(probs)
		log_probs = dist.log_prob(actions)
		actor_loss = -log_probs*(returns-values)

		total_loss = (critic_loss + actor_loss).mean()
		return total_loss

	def choose_action(self, observation):
		state = T.tensor([observation], dtype=T.float32)
		pi, _ = self.forward(state)
		probs = T.softmax(pi, dim=1)
		dist = Categorical(probs)
		action = dist.sample().numpy()[0]

		return action
	
	def get_conv_output_size(self, shape):
		batch_size = 1
		input = T.autograd.Variable(T.rand(batch_size, *shape))
		output_feat = self.conv_layers(input)
		conv_output_size = output_feat.data.view(batch_size, -1).size(1)

		return conv_output_size

  
class Agent(mp.Process):
	def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, gamma, lr, name, global_ep_idx, env_id):

		super(Agent, self).__init__()
		self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
		self.global_actor_critic = global_actor_critic
		self.name = 'w%02i' % name
		self.episode_idx = global_ep_idx
		# self.env = gym.make(env_id)
		self.env = FlattenObservation(gym.make('gym_environment:gym_environment/SimpleBattery', days=1, predict=True, day_offset=0, charge_penalty_mwh=8.0))
		self.optimizer = optimizer

	def run(self):
		t_step = 1
		while self.episode_idx.value < N_GAMES:
			done = False
			observation, _ = self.env.reset()
			score = 0
			self.local_actor_critic.clear_memory()
			while not done:
				action = self.local_actor_critic.choose_action(observation)
				observation_, reward, done, truncated, info = self.env.step(action) # Where observation_ = next_state
				score += reward
				self.local_actor_critic.remember(observation, action, reward)
				if t_step % T_MAX == 0 or done:
					loss = self.local_actor_critic.calc_loss(done)
					self.optimizer.zero_grad()
					loss.backward()
					for local_param, global_param in zip(self.local_actor_critic.parameters(), self.global_actor_critic.parameters()):
						global_param._grad = local_param.grad
					self.optimizer.step()
					self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
					self.local_actor_critic.clear_memory()
				t_step += 1
				observation = observation_
			with self.episode_idx.get_lock():
				self.episode_idx.value += 1
			print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)

N_GAMES = 600
T_MAX = 15

if __name__ == '__main__':
	lr = 1e-4
	env_id = 'CartPole-v1'
	env = FlattenObservation(gym.make('gym_environment:gym_environment/SimpleBattery'))
	n_actions = env.action_space.n
	input_dims = np.array(env.observation_space.sample()).reshape((-1,1)).shape
	global_actor_critic = ActorCritic(input_dims, n_actions)
	for layer in global_actor_critic.children():
		if hasattr(layer, 'out_features'):
			print(layer.out_features)
	global_actor_critic.share_memory()
	optim = SharedAdam(global_actor_critic.parameters(), lr=lr, betas=(0.92, 0.999))

	global_ep = mp.Value('i', 0)

	workers = [Agent(global_actor_critic, 
						optim, 
						input_dims,
						n_actions,
						gamma=0.99,
						lr=lr,
						name=i,
						global_ep_idx=global_ep,
						env_id=env_id) for i in range(mp.cpu_count() - 2)]
	# range(mp.cpu_count() - 2)
	
	[w.start() for w in workers]
	[w.join() for w in workers]