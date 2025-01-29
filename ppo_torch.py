import os
import numpy as np
import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

def assure_tensor_type(input_collection, device):
    if not torch.is_tensor(input_collection):
        return torch.tensor(np.array([input_collection]), dtype=torch.float32, device =device)
    else:
        return input_collection

def add_to_tensor(prev_collection, new_data):
    if prev_collection.numel() > 0:
        return torch.cat((prev_collection, new_data))        
    else:
        return new_data

class PPOMemory:
    def __init__(self, batch_size, learning_length, device, input_dims):
        self.device = device
        self.batch_size = batch_size
        self.learning_length = learning_length
        self.num_features = input_dims

        self.states = torch.zeros((self.batch_size, self.learning_length, self.num_features), device=self.device) 
        self.probs = torch.zeros((self.batch_size, self.learning_length), device=self.device)  
        self.vals = torch.zeros((self.batch_size, self.learning_length), device=self.device)  
        self.actions = torch.zeros((self.batch_size, self.learning_length), device=self.device)  
        self.rewards = torch.zeros((self.batch_size, self.learning_length), device=self.device) 
        self.dones = torch.zeros((self.batch_size, self.learning_length), device=self.device) 

        self.batch_size = batch_size
        self.memory_counter = 0

    def generate_batches(self, learning_batch_size = 256):
        batch_start = np.arange(0, self.batch_size * self.learning_length, learning_batch_size)
        indices = np.arange(self.batch_size * self.learning_length, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+learning_batch_size] for i in batch_start]
        return self.states, self.actions, self.probs, self.vals, self.rewards, self.dones, batches



    def store_memory(self, state, action, probs, vals, reward, done):
        # Ensure all inputs are tensors
        self.states[:,self.memory_counter,:] = state
        self.actions[:,self.memory_counter] = action
        self.probs[:,self.memory_counter] = probs
        self.vals[:,self.memory_counter] = vals
        self.rewards[:,self.memory_counter] = reward
        self.dones[:,self.memory_counter] =  done

        self.memory_counter += 1


    def clear_memory(self):
        self.memory_counter = 0
        self.states = torch.zeros((self.batch_size, self.learning_length, self.num_features), device=self.device) 
        self.probs = torch.zeros((self.batch_size, self.learning_length), device=self.device)  
        self.vals = torch.zeros((self.batch_size, self.learning_length), device=self.device)  
        self.actions = torch.zeros((self.batch_size, self.learning_length), device=self.device)  
        self.rewards = torch.zeros((self.batch_size, self.learning_length), device=self.device) 
        self.dones = torch.zeros((self.batch_size, self.learning_length), device=self.device) 

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='tmp\\ppo', device="cuda"):
        super(ActorNetwork, self).__init__()
        self.device = device
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.LayerNorm(fc1_dims),  # Batch normalization after the first linear layer
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Dropout with 50% probability
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),  # Batch normalization after the second linear layer
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Dropout with 50% probability
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay = (alpha*3)/10)
        self.to(self.device)

    def forward(self, state):
        assert state.shape != torch.Size([0]), "Error: torch tensor has an empty shape!"              
        dist = self.actor(state)
        
        dist = Categorical(dist)
        
        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp/ppo', device="cuda"):
        super(CriticNetwork, self).__init__()
        self.device = device
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.LayerNorm(fc1_dims),  # Batch normalization after the first linear layer
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),  # Dropout with 50% probability
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),  # Batch normalization after the second linear layer
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),  # Dropout with 50% probability
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay = (alpha*3)/10)
        self.to(self.device)

    def forward(self, state):
        assert state.shape != torch.Size([0]), "Error: torche tensor has an empty shape!"
        value = self.critic(state)

        return value.squeeze()

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99999, alpha=0.0003, gae_lambda=0.99999,
            policy_clip=0.15, batch_size=64, learning_length = 1, n_epochs=10, fc1_dims=256, fc2_dims=256, device="cuda"):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.device = device
        self.batch_size = batch_size
        if isinstance(input_dims, tuple):
            self.input_dims = input_dims[0]
        else:
            self.input_dims = input_dims

        self.actor = ActorNetwork(n_actions, input_dims, alpha,fc1_dims=fc1_dims,fc2_dims=fc2_dims, device=self.device)
        self.critic = CriticNetwork(input_dims, alpha,fc1_dims=fc1_dims,fc2_dims=fc2_dims, device=self.device)
        self.memory = PPOMemory(batch_size, learning_length, device=self.device, input_dims=self.input_dims)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        print('... model has been saved ...')


    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        print('... model has been loaded ...')

    def choose_action(self, observation, epsilon = 0.0, debug = False):
        #debug state output
        if debug:
            print(f'observation before fix {observation}')
            print(f'observation before fix {type(observation)}')

        if isinstance(observation, tuple):
            observation = observation[0]
        
        if debug:
            print(f'observation after fix {observation}')   

        assert observation.shape != torch.Size([0]), "Error: torch tensor has an empty shape!"
        dist = self.actor(observation)
        value = self.critic(observation)
        action = dist.sample()
        probs = dist.log_prob(action)

        if np.random.rand() > epsilon:
            #print('Random action taken')
            action = torch.randint(0, 3, value.shape, device=value.device).squeeze()
            #probs = torch.zeros_like(probs, device=value.device) + 1/3

        #probs = torch.squeeze(probs).item()
        #action = torch.squeeze(action).item()
        #value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            # Process all data at once instead of generating batches multiple times
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.generate_batches(learning_batch_size=512)  # Increased batch size

            dones_arr = dones_arr.int()
            
            # Compute all values at once using vectorized operations
            with torch.no_grad():  # Prevent unnecessary gradient computation
                next_vals = torch.cat([vals_arr[:, 1:], 
                                    torch.zeros((self.batch_size, 1), device=self.device)], dim=1)
                deltas_arr = reward_arr + self.gamma * next_vals * (1 - dones_arr) - vals_arr

                # Vectorized GAE computation
                advantages_reversed = []
                gae = torch.zeros(self.batch_size, device=self.device)
                
                for delta, done in zip(deltas_arr.transpose(0, 1).flip(0), 
                                    dones_arr.transpose(0, 1).flip(0)):
                    gae = delta + self.gamma * self.gae_lambda * (1 - done) * gae
                    advantages_reversed.append(gae)
                
                advantage_arr = torch.stack(advantages_reversed[::-1], dim=1)

            # Reshape all tensors at once
            state_arr = state_arr.reshape(-1, self.input_dims)
            old_prob_arr = old_prob_arr.reshape(-1)
            action_arr = action_arr.reshape(-1)
            vals_arr = vals_arr[:, :-1].reshape(-1)
            advantage_arr = advantage_arr.reshape(-1)
            
            # Normalize advantages for better training stability
            advantage_arr = (advantage_arr - advantage_arr.mean()) / (advantage_arr.std() + 1e-8)

            for batch in batches:
                # Get batch data - no need for detach() and clone() since we're not modifying these
                states = state_arr[batch]
                old_probs = old_prob_arr[batch]
                actions = action_arr[batch]
                values = vals_arr[batch]
                advantage = advantage_arr[batch]

                # Zero gradients once for both networks
                self.actor.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                self.critic.optimizer.zero_grad(set_to_none=True)

                # Forward passes
                dist = self.actor(states)
                critic_value = self.critic(states)
                
                # Compute actor loss
                new_probs = dist.log_prob(actions)
                prob_ratio = (new_probs - old_probs).exp()
                
                # Vectorized loss computation
                weighted_probs = advantage * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 
                                                1-self.policy_clip, 
                                                1+self.policy_clip) * advantage
                
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                # Compute critic loss
                returns = advantage + values
                critic_loss = 0.5 * ((returns - critic_value) ** 2).mean()
                
                # Combined loss and backprop
                total_loss = actor_loss + critic_loss
                total_loss.backward()
                
                # Optional: Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                
                # Update both networks
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

