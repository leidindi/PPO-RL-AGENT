#https://www.youtube.com/watch?v=OcIx_TBu90Q

import gymnasium as gym
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import json
import itertools
from gymnasium.wrappers import FlattenObservation

class SharedAdam(T.optim.Adam):
	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-3, weight_decay=0):
		super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

		for group in self.param_groups:
			for p in group['params']:
				state = self.state[p]
				state['step'] = 0
				state['exp_avg'] = T.zeros_like(p.data)
				state['exp_avg_sq'] = T.zeros_like(p.data)

				state['exp_avg'].share_memory_()
				state['exp_avg_sq'].share_memory_()

def ensure_shared_grads(local_model, global_model):
    for param, global_param in zip(local_model.parameters(),global_model.parameters()):
        if global_param.grad is not None:
            return
        global_param.grad = param.grad 

class ActorCritic(nn.Module):
	def __init__(self, input_dims, n_actions, hidden_size, gamma=0.99):
		super(ActorCritic, self).__init__()

		self.gamma = gamma

		# https://stackoverflow.com/questions/58949680/using-lstms-to-predict-from-single-element-sequence
		# https://stackoverflow.com/questions/61960385/using-lstm-stateful-for-passing-context-b-w-batches-may-be-some-error-in-contex
		# https://stackoverflow.com/questions/58949680/using-lstms-to-predict-from-single-element-sequence
  
	
		""" Using the layers from: https://github.com/dgriff777/rl_a3c_pytorch/blob/master/model.py"""

		self.hidden_size = hidden_size

		self.lin1 = nn.Linear(input_dims, 64) # from 64 to 128
		self.lin2 = nn.Linear(64, 64)
		self.lin3 = nn.Linear(64, 128)
		self.lin4 = nn.Linear(128, 128)

		self.lstm = nn.LSTMCell(128, self.hidden_size)
		self.critic_lin = nn.Linear(self.hidden_size, 1)
		self.actor_lin = nn.Linear(self.hidden_size, n_actions)

	def save_hyperparameters(self, hyper_params, name):
		cur_path = os.path.dirname(os.path.realpath(__file__))
		file_path = os.path.join(cur_path, './models/A3C/hyper_params/{}.json'.format(name))
		with open(file_path, "w") as json_file:
			json.dump(hyper_params, json_file, indent=4)
	
	def save_model(self, episodes, hyper_params, score):
		cur_path = os.path.dirname(os.path.realpath(__file__))
		name = "A3C_{}score_{}ep_{}days{}_{}pen".format(score, episodes, hyper_params['days'], hyper_params['day_offset'], hyper_params['charge_pen'])
		data_path = os.path.join(cur_path, './models/A3C/{}.pt'.format(name))
		T.save(self.state_dict(), data_path)
		self.save_hyperparameters(hyper_params, name)

	def forward(self, state, hx, cx):
		
		""" Using the layers from: https://github.com/dgriff777/rl_a3c_pytorch/blob/master/model.py"""

		# Apply relu and max pooling to cnn layers
		x = F.relu(self.lin1(state))
		x = F.relu(self.lin2(x))
		x = F.relu(self.lin3(x))
		x = F.relu(self.lin4(x))

		x = x.view(x.size(0), -1) # Flatten the output for the RNN

		hx, cx = self.lstm(x)

		x = hx

		return self.actor_lin(x), self.critic_lin(x), (hx, cx)
	


class Agent(mp.Process):
	def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, name, global_ep_idx, score_avg, high_score, hyper_params, stop_event):

		super(Agent, self).__init__()
		# Setting seed
		T.manual_seed(SEED + name)
		T.cuda.manual_seed(SEED + name)

		self.stop_event = stop_event
		
		self.hyper_params = hyper_params
		self.hidden_size = self.hyper_params['hidden_size']
		self.gamma = self.hyper_params['gamma']
		self.gamma_coef = self.hyper_params['gamma_coef']
		self.ent_coef = self.hyper_params['ent_coef']


		self.local_actor_critic = ActorCritic(input_dims, n_actions, self.hidden_size, self.gamma)
		self.global_actor_critic = global_actor_critic
		self.name = 'w%02i' % name
		self.episode_idx = global_ep_idx
		self.score_avg = score_avg
		self.high_score = high_score
		# self.env = gym.make(env_id)
		self.env = FlattenObservation(gym.make('gym_environment:gym_environment/SimpleBattery', days=self.hyper_params['days'], predict=True
																				 , day_offset=self.hyper_params['day_offset'], charge_penalty_mwh=self.hyper_params['charge_pen']))
		self.optimizer = optimizer


	def run(self):
			try:
				while self.episode_idx.value < N_GAMES and not self.stop_event.is_set():
					done = False
					self.hx = T.zeros(1, self.hidden_size)
					self.cx = T.zeros(1, self.hidden_size)

					state, _ = self.env.reset()
					state = T.tensor(state, dtype=T.float32)
					self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
					score = 0

					values = []
					log_probs = []
					entropies = []
					rewards = []

					while not done:
						# TODO: """ Add T_STEP updates for longer time series"""
						# t_step = 0
						# while t_step > T_STEP:
						pi, v, (self.hx, self.cx) = self.local_actor_critic(state.unsqueeze(0), self.hx, self.cx)

						prob = F.softmax(pi, dim=1)
						log_prob = F.log_softmax(pi, dim=1)
						entropy =-(log_prob * prob).sum(1, keepdim=True)
						entropies.append(entropy)

						action = prob.multinomial(num_samples=1).detach()
						log_prob = log_prob.gather(1, action)

						next_state, reward, done, truncated, info = self.env.step(action.item())
						score += reward

						state = T.tensor(next_state, dtype=T.float32)
						values.append(v)
						log_probs.append(log_prob)
						rewards.append(reward)

					R = T.zeros(1, 1)
					values.append(R)

					actor_loss = 0
					critic_loss = 0
					gae = T.zeros(1, 1)
					for i in reversed(range(len(rewards))):
						R = rewards[i] + self.gamma*R
						advantage = R - values[i]
						critic_loss = critic_loss + 0.5 * advantage.pow(2)

						# Generalized Advantage Estimation
						delta_t = rewards[i] + self.gamma * values[i + 1] - values[i]
						gae = gae * self.gamma * self.gamma_coef + delta_t

						actor_loss = actor_loss - log_probs[i] * gae.detach() - self.ent_coef * entropies[i]

					self.optimizer.zero_grad()
					# loss = (critic_loss +  actor_loss).mean()
					loss = (0.5 * critic_loss +  actor_loss)
					loss.backward()

					ensure_shared_grads(self.local_actor_critic, self.global_actor_critic)
					self.optimizer.step()

					
					# Outside of T_STEP
					with self.episode_idx.get_lock():
						self.episode_idx.value += 1
					with self.score_avg.get_lock():
						for i in range(len(self.score_avg) - 1):
							self.score_avg[i] = self.score_avg[i  + 1]
						self.score_avg[-1] = score
					
					last_10_avg = round(np.mean(self.score_avg[:]))
					if last_10_avg > self.high_score.value:
						with self.high_score.get_lock():
							self.high_score.value = last_10_avg
						self.global_actor_critic.save_model(self.episode_idx.value, self.hyper_params, round(last_10_avg))
					print(self.name, '\t episode ', self.episode_idx.value, '\t reward %.1f' % score, '\t loss %.1f' % loss.item(), '\t average %.1f' % last_10_avg)
			except KeyboardInterrupt:
				print("KeyboardInterrupt exception is caught")

N_GAMES = 500
SEED = 1234
T_STEP = 1440 * 1

def save_hyper_dict(hyper_dict):
	with open('hyper_dict.json', 'w') as file:
		for json_obj in hyper_dict:
			json_str = json.dumps(json_obj)
			file.write(json_str +"\n")

def load_hyper_dict():
	with open("hyper_dict.json") as file:
		json_list = [json.loads(line) for line in file]
	return json_list


def create_permutations(lr_range, gamma_range, gamma_coef_range, ent_coef_range, hidden_size_range, charge_pen_range, days_range, days_offset_range, T_STEP_range):
	permutations = list(itertools.product(lr_range, gamma_range, gamma_coef_range, 
																			 ent_coef_range, hidden_size_range, charge_pen_range, 
																			 days_range, days_offset_range, T_STEP_range))
	hyper_dicts = []
	for perm in permutations:
		hyper_dict = {
		'lr' : perm[0], 
		'gamma' : perm[1],
		'gamma_coef' : perm[2], 
		'ent_coef' : perm[3], 
		'hidden_size' : perm[4],
		'charge_pen' : perm[5],
		'days' : perm[6], 
		'day_offset' : perm[7], 
		'T_STEP': perm[8], 
		}
		hyper_dicts.append(hyper_dict)

	return hyper_dicts
 

if __name__ == '__main__':
	T.manual_seed(SEED)
	T.cuda.manual_seed(SEED)
	np.random.seed(SEED)
	
	env = FlattenObservation(gym.make('gym_environment:gym_environment/SimpleBattery'))
	n_actions = env.action_space.n
	input_dims = env.observation_space.shape


	# Hyperparamters
	hyper_params = {
		'lr' : 1e-5, # learning rate (-4)
		'gamma' : 0.8, # future rewards (0.8) -> this means the loss will be higher (no loss = nothing to optimize)
		'gamma_coef' : 0.9, # lambda(tau 0.8) -> affects the actor_loss
		'ent_coef' : 0.02, # exploration (0.02) -> affects the actor loss
		'hidden_size' : 256, # LSTM Cells (128 1 day) (256 7 days)
		'charge_pen' : 0.0, # cycles reduction
		'days' : 1, # lengths of training data
		'day_offset' : 120, # offset in the training data
		'T_STEP': T_STEP, # after how many steps to do one optimizer step
	}
 
	tune = False
	if tune:

		new_params = True
		if new_params:
			hyper_dicts = create_permutations(lr_range=[1e-5, 1e-6],
												gamma_range=[0.7, 0.8, 0.9, 0.99],
												gamma_coef_range=[0.8, 0.9],
												ent_coef_range=[0.2, 0.1, 0.02, 0.01],
												hidden_size_range=[128, 256],
												charge_pen_range=[0.0],
												days_range=[7],
												days_offset_range=[120],
												T_STEP_range=[T_STEP]
												)
		else:
			hyper_dicts = load_hyper_dict()

		checked_params = []

		try:
			for hyper_params in hyper_dicts:
				print("Tuning for {} hyper paramters".format(len(hyper_dicts)))
				print("Now training with: {}".format(hyper_params))
				global_actor_critic = ActorCritic(input_dims[0], n_actions, hyper_params['hidden_size'])
				global_actor_critic.share_memory()
				optim = SharedAdam(global_actor_critic.parameters(), lr=hyper_params['lr'], betas=(0.92, 0.999))

				score_avg = mp.Array('d', range(10))
				global_ep = mp.Value('i', 0)
				high_score = mp.Value('i', 3000)
				stop_event = mp.Event()
				start_time = time.time()

				# total_workers = mp.cpu_count() - 8
				total_workers = 8
				workers = []


				for i in range(total_workers):
					w = Agent(global_actor_critic, optim, input_dims[0], n_actions, name=i, global_ep_idx=global_ep, score_avg=score_avg, 
								high_score=high_score, hyper_params=hyper_params, stop_event=stop_event)
					w.start()
					workers.append(w)
					time.sleep(0.001)

				print("All workers are active...")

				# Check of average reward change
				reward_changing = True
				last_averages = None
				last_index = None
				last_20 = []
				while reward_changing and global_ep.value < N_GAMES:
					# Ensure an episode is completed by the workers
					if last_index != global_ep.value:
						# After how many episodes to check for training amount
						if global_ep.value >= 100:
							if global_ep.value % 10 == 0:
								last_20 = last_20 + score_avg[:]
							if global_ep.value % 20 == 0:
								if np.std(last_20) <= 15:
									stop_event.set()
									reward_changing = False
								last_20 = []
						last_index = global_ep.value
					time.sleep(0.1)

				for w in workers:
					time.sleep(0.001)
					w.join()

				checked_params.append(hyper_params)
				new_dict_list = [x for x in hyper_dicts if x not in checked_params]
				save_hyper_dict(new_dict_list)
				print("Model is not training anymore, trying new hyper_params")
				print("Saving model...")
				global_actor_critic.save_model(global_ep.value, hyper_params, round(np.mean(score_avg[:])))
					
		except KeyboardInterrupt:
			global_actor_critic.save_model(global_ep.value, hyper_params, round(np.mean(score_avg[:])))
			end_time = time.time()
			total_time = end_time - start_time
			episodes_s = round(global_ep.value / total_time, 2)
			print('Total Training time: {} minutes with {} workers EP/s {}'.format(round(total_time/60, 2), total_workers, episodes_s))
	else:
		try:
			print("No tuning")
			print("Now training with: {}".format(hyper_params))
			global_actor_critic = ActorCritic(input_dims[0], n_actions, hyper_params['hidden_size'])
			global_actor_critic.share_memory()
			optim = SharedAdam(global_actor_critic.parameters(), lr=hyper_params['lr'], betas=(0.92, 0.999))

			score_avg = mp.Array('d', range(10))
			global_ep = mp.Value('i', 0)
			high_score = mp.Value('i', 3000)
			stop_event = mp.Event()
			start_time = time.time()

			# total_workers = mp.cpu_count() - 8
			total_workers = 8
			workers = []


			for i in range(total_workers):
				w = Agent(global_actor_critic, optim, input_dims[0], n_actions, name=i, global_ep_idx=global_ep, score_avg=score_avg, 
							high_score=high_score, hyper_params=hyper_params, stop_event=stop_event)
				w.start()
				workers.append(w)
				time.sleep(0.001)

			print("All workers are active...")

			for w in workers:
				time.sleep(0.001)
				w.join()
		except:
			global_actor_critic.save_model(global_ep.value, hyper_params, round(np.mean(score_avg[:])))
			end_time = time.time()
			total_time = end_time - start_time
			episodes_s = round(global_ep.value / total_time, 2)
			print('Total Training time: {} minutes with {} workers EP/s {}'.format(round(total_time/60, 2), total_workers, episodes_s))
			