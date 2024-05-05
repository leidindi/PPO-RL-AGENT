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

		# self.conv_layers = nn.Sequential(
		# 	nn.Conv1d(in_channels=input_dims[1], out_channels=32, kernel_size=1),
		# 	nn.ReLU(),
		# 	nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1),
		# 	nn.ReLU(),
		# 	nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
		# 	nn.ReLU(),
		# )

		# https://stackoverflow.com/questions/58949680/using-lstms-to-predict-from-single-element-sequence
		# https://stackoverflow.com/questions/61960385/using-lstm-stateful-for-passing-context-b-w-batches-may-be-some-error-in-contex
		# https://stackoverflow.com/questions/58949680/using-lstms-to-predict-from-single-element-sequence
  
		# Calculate the shape of the output from the convolutional layers
		# conv_output_size = self.get_conv_output_size(input_dims)
        
		# self.lstm = nn.LSTM(input_size=conv_output_size, hidden_size=64, batch_first=True)

		# self.fc_layers = nn.Sequential(
		# 	nn.Linear(64, 128),
		# 	nn.ReLU(),
		# 	nn.Linear(128, n_actions)
		# )

		# self.v_layers = nn.Sequential(
		# 	nn.Linear(64, 128),
		# 	nn.ReLU(),
		# 	nn.Linear(128, 1)
		# )

		# self.pi1 = nn.Linear(input_dims[0], 128)  
		# self.pi2 = nn.Linear(128, 256)  
		# self.pi3 = nn.Linear(256, 256)  
		# self.pi4 = nn.Linear(256, 128)  
		# self.pi = nn.Linear(128, n_actions)

		# self.v1 = nn.Linear(input_dims[0], 128)
		# self.v2 = nn.Linear(128, 128)
		# self.v = nn.Linear(128, 1)
	
		""" Using the layers from: https://github.com/dgriff777/rl_a3c_pytorch/blob/master/model.py"""

		self.hidden_size = 64
		# self.covn1 = nn.Conv1d(input_dims[0], 32, 5, stride=1, padding=2)
		# self.conv2 = nn.Conv1d(32, 32, 5, stride=1, padding=1)
		# self.conv3 = nn.Conv1d(32, 64, 4, stride=1, padding=1)
		# self.conv4 = nn.Conv1d(64, 64, 3, stride=1, padding=1)
		self.lin1 = nn.Linear(input_dims, 32)
		self.lin2 = nn.Linear(32, 32)
		self.lin3 = nn.Linear(32, 64)
		self.lin4 = nn.Linear(64, 64)

		self.lstm = nn.LSTMCell(64, self.hidden_size)
		self.critic_lin = nn.Linear(self.hidden_size, 1)
		self.actor_lin = nn.Linear(self.hidden_size, n_actions)
		
		# Might need zero weight initialization

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

	def forward(self, state, hx, cx):
		# pi1 = F.relu(self.pi1(state))
		# pi2 = F.relu(self.pi2(pi1))
		# pi3 = F.relu(self.pi3(pi2))
		# pi4 = F.relu(self.pi4(pi3))
		# v1 = F.relu(self.v1(state))
		# v2 = F.relu(self.v2(v1))

		# pi = self.pi(pi4.squeeze(1))
		# v = self.v(v2.squeeze(1))
		# return pi, v
		# state = state.unsqueeze(2)
	
		# for layer in self.conv_layers:
		# 	state = layer(state)
		# 	print(state.size())
	
		# pi = self.conv_layers(state)
		# pi = pi.view(pi.size(0), -1)  # Flatten the output for the RNN
		# pi, _ = self.lstm(pi.unsqueeze(1))  # Add a time dimension (batch_size, seq_len, input_size)
		# pi = pi.squeeze(1)  # Remove the time dimension
		# pi = self.fc_layers(pi)

		# v = self.conv_layers(state)
		# v = v.view(v.size(0), -1)  # Flatten the output for the RNN
		# v, _ = self.lstm(v.unsqueeze(1))  # Add a time dimension (batch_size, seq_len, input_size)
		# v = v.squeeze(1)  # Remove the time dimension
		# v = self.v_layers(v)

		# return state
		
		""" Using the layers from: https://github.com/dgriff777/rl_a3c_pytorch/blob/master/model.py"""

		# Apply relu and max pooling to cnn layers
		# x = F.relu(F.max_pool2d(self.covn1(state), 2, 2))
		# x = F.relu(F.max_pool2d(self.covn2(x), 2, 2))
		# x = F.relu(F.max_pool2d(self.covn3(x), 2, 2))
		# x = F.relu(F.max_pool2d(self.covn4(x), 2, 2))
		x = F.relu(self.lin1(state))
		x = F.relu(self.lin2(x))
		x = F.relu(self.lin3(x))
		x = F.relu(self.lin4(x))

		# print(x.shape)
		# x = x.view(x.size(0), -1) # Flatten the output for the RNN
		# print(x.shape)

		# May need to init hx and cx to zero for the initial state
		if hx is not None and cx is not None:
			hx, cx = self.lstm(x, (hx, cx))
		else:
			hx, cx = self.lstm(x)

		x = hx

		return self.actor_lin(x), self.critic_lin(x), hx, cx
  
	def calc_R(self, done, v):
		# Self the reard to 0 if a terminal state is reached
		R = v[-1]*(1-int(done))

		print(v.shape)
		# Calculate the batch returns in reverse
		batch_return = []
		for reward in self.rewards[::-1]:
			R = reward + self.gamma*R
			batch_return.append(R)

		# Reverse the list a gain to make sure the states are in the correct order again
		batch_return.reverse()
		batch_return = T.tensor(batch_return, dtype=T.float32)

		return batch_return
  
	def calc_loss(self, done, hx, cx):
		states = T.tensor(self.states, dtype=T.float32)
		actions = T.tensor(self.actions, dtype=T.float32)

		pi, values, hx, cx = self.forward(states, hx, cx)

		returns = self.calc_R(done, values)

		values = values.squeeze()
		critic_loss = (returns - values)**2

		probs = T.softmax(pi, dim=1)
		dist = Categorical(probs)
		log_probs = dist.log_prob(actions)
		actor_loss = -log_probs*(returns-values)

		total_loss = (critic_loss + actor_loss).mean()
		return total_loss, hx, cx

	def choose_action(self, observation, hx, cx):
		state = T.tensor(observation, dtype=T.float32).unsqueeze(0)
		pi, _, _, _ = self.forward(state, hx, cx)
		probs = T.softmax(pi, dim=1)
		dist = Categorical(probs)
		action = dist.sample().numpy()[0]

		return action
	
	def get_conv_output_size(self, shape):
		batch_size = 1
		inp = T.autograd.Variable(T.rand(batch_size, shape[1], shape[0]))
		output_feat = self.conv_layers(inp)
		print(output_feat.shape)
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
		self.env = FlattenObservation(gym.make('gym_environment:gym_environment/SimpleBattery', days=1, predict=True, day_offset=0, charge_penalty_mwh=0.0))
		self.optimizer = optimizer

		# Might need to init
		self.hx = None
		self.cx = None

	def run(self):
		t_step = 1
		while self.episode_idx.value < N_GAMES:
			done = False
			observation, _ = self.env.reset()
			score = 0
			self.local_actor_critic.clear_memory()
			while not done:
				action = self.local_actor_critic.choose_action(observation, None, None)
				observation_, reward, done, truncated, info = self.env.step(action) # Where observation_ = next_state
				score += reward
				self.local_actor_critic.remember(observation, action, reward)
				if t_step % T_MAX == 0 or done:
					loss, self.hx, self.cx = self.local_actor_critic.calc_loss(done, self.hx, self.cx)
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
			print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score, 'loss ', loss.item())

N_GAMES = 100
T_MAX = 15

if __name__ == '__main__':
	lr = 1e-4
	env_id = 'CartPole-v1'
	env = FlattenObservation(gym.make('gym_environment:gym_environment/SimpleBattery'))
	n_actions = env.action_space.n
	input_dims = env.observation_space.shape
	# test_env = gym.make('SpaceInvaders-v4')
	# print(test_env.observation_space.shape)
	# print(test_env.observation_space.sample())
	global_actor_critic = ActorCritic(input_dims[0], n_actions)
	global_actor_critic.share_memory()
	optim = SharedAdam(global_actor_critic.parameters(), lr=lr, betas=(0.92, 0.999))

	global_ep = mp.Value('i', 0)

	workers = [Agent(global_actor_critic, 
						optim, 
						input_dims[0],
						n_actions,
						gamma=0.99,
						lr=lr,
						name=i,
						global_ep_idx=global_ep,
						env_id=env_id) for i in range(1)]
	# range(mp.cpu_count() - 2)
	
	[w.start() for w in workers]
	[w.join() for w in workers]