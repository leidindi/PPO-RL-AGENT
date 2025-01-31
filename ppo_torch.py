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

    def generate_batches(self, learning_batch_size=256):
        # Calculate total size of the data
        total_size = self.batch_size * self.learning_length
        
        # Create indices and shuffle them
        indices = np.arange(total_size, dtype=np.int64)
        np.random.shuffle(indices)
        
        # Calculate number of complete batches
        n_batches = total_size // learning_batch_size
        
        # Create batches of the correct size
        batches = []
        for i in range(n_batches):
            start_idx = i * learning_batch_size
            batch_indices = indices[start_idx:start_idx + learning_batch_size]
            batches.append(batch_indices)
        
        # Handle any remaining data (optional)
        if total_size % learning_batch_size != 0:
            remaining_indices = indices[n_batches * learning_batch_size:]
            if len(remaining_indices) > 0:
                batches.append(remaining_indices)
        
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
        # Enable anomaly detection for debugging
        torch.autograd.set_detect_anomaly(True)
        
        for _ in range(self.n_epochs):
            # Process all data at once instead of generating batches multiple times
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.generate_batches(learning_batch_size=256)

            dones_arr = dones_arr.int()
            
            # Create a separate tensor for next values
            next_vals = torch.cat([vals_arr[:, 1:], 
                                torch.zeros((self.batch_size, 1), device=self.device)], dim=1)
            
            # Compute advantages using fresh tensors
            with torch.no_grad():
                # Compute deltas without modifying original tensors
                deltas = reward_arr + self.gamma * next_vals * (1 - dones_arr) - vals_arr
                
                # GAE computation with explicit new tensor creation
                advantages = torch.zeros_like(deltas, device=self.device)
                last_gae = torch.zeros(self.batch_size, device=self.device)
                
                for t in reversed(range(deltas.size(1))):
                    # Create new tensors for each computation
                    last_gae = deltas[:, t] + \
                            self.gamma * self.gae_lambda * \
                            (1 - dones_arr[:, t]) * last_gae
                    advantages[:, t] = last_gae.clone()
                
                # Normalize advantages with new tensor creation
                advantage_mean = advantages.mean()
                advantage_std = advantages.std() + 1e-8
                normalized_advantages = (advantages - advantage_mean) / advantage_std
                
                # Create fresh tensors for all reshaping operations
                states = state_arr.reshape(-1, self.input_dims).clone()
                old_probs = old_prob_arr.reshape(-1).clone()
                actions = action_arr.reshape(-1).clone()
                values = vals_arr.reshape(-1).clone()
                advantages = normalized_advantages.reshape(-1).clone()

            for batch in batches:
                # Create fresh batch tensors
                batch_states = states[batch]
                batch_old_probs = old_probs[batch]
                batch_actions = actions[batch]
                try:
                    batch_values = values[batch]
                except:
                    pass
                batch_advantages = advantages[batch]

                # Zero gradients
                self.actor.optimizer.zero_grad(set_to_none=True)
                self.critic.optimizer.zero_grad(set_to_none=True)

                # Forward passes
                dist = self.actor(batch_states)
                critic_value = self.critic(batch_states)
                
                # Compute losses with explicit new tensor creation
                new_probs = dist.log_prob(batch_actions)
                prob_ratio = (new_probs - batch_old_probs).exp()
                
                # Create separate tensors for each operation
                weighted_probs = batch_advantages * prob_ratio
                clipped_ratio = torch.clamp(prob_ratio, 
                                        1-self.policy_clip, 
                                        1+self.policy_clip)
                weighted_clipped_probs = batch_advantages * clipped_ratio
                
                # Compute final losses
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = batch_advantages + batch_values
                critic_loss = 0.5 * ((returns - critic_value) ** 2).mean()
                
                # Compute total loss and backward pass
                total_loss = actor_loss + critic_loss
                
                try:
                    total_loss.backward()
                except RuntimeError as e:
                    print(f"Error during backward pass: {e}")
                    print(f"Actor loss: {actor_loss}")
                    print(f"Critic loss: {critic_loss}")
                    print(f"Advantage shape: {batch_advantages.shape}")
                    print(f"Prob ratio shape: {prob_ratio.shape}")
                    raise e
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                
                # Update networks
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        # Disable anomaly detection after training
        torch.autograd.set_detect_anomaly(False)
        self.memory.clear_memory()

