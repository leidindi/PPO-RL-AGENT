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
    try:
        if prev_collection.numel() > 0:
            return torch.cat((prev_collection, new_data))        
        else:
            return new_data
    except:
        pass

class PPOMemory:
    def __init__(self, batch_size, device):
        self.device = device
        self.states = torch.tensor([], device=self.device) 
        self.probs = torch.tensor([], device=self.device) 
        self.vals = torch.tensor([], device=self.device) 
        self.actions = torch.tensor([], device=self.device) 
        self.rewards = torch.tensor([], device=self.device) 
        self.dones = torch.tensor([], device=self.device)
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return self.states, self.actions, self.probs, self.vals, self.rewards, self.dones, batches



    def store_memory(self, state, action, probs, vals, reward, done):
        # Ensure all inputs are tensors
        state   = assure_tensor_type(state, self.device)
        action  = assure_tensor_type(action,self.device)
        probs   = assure_tensor_type(probs, self.device)
        vals    = assure_tensor_type(vals,  self.device)
        reward  = assure_tensor_type(reward,self.device)
        done    = assure_tensor_type(done,  self.device)

        # Append using torch.cat if empty object variable
        

        self.states     = add_to_tensor(self.states,    state)
        self.actions    = add_to_tensor(self.actions,   action)
        self.probs      = add_to_tensor(self.probs,     probs)
        self.vals       = add_to_tensor(self.vals,      vals)
        self.rewards    = add_to_tensor(self.rewards,   reward)
        self.dones      = add_to_tensor(self.dones,     done)


    def clear_memory(self):
        self.states = torch.tensor(np.array([]), device=self.device) 
        self.probs = torch.tensor(np.array([]), device=self.device) 
        self.vals = torch.tensor(np.array([]), device=self.device) 
        self.actions = torch.tensor(np.array([]), device=self.device) 
        self.rewards = torch.tensor(np.array([]), device=self.device) 
        self.dones = torch.tensor(np.array([]), device=self.device)

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
        assert state.shape != torch.Size([0]), "Error: torchhe tensor has an empty shape!"
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=1, alpha=0.0003, gae_lambda=1,
            policy_clip=0.15, batch_size=64, n_epochs=10, fc1_dims=256, fc2_dims=256, device="cuda"):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.device = device

        self.actor = ActorNetwork(n_actions, input_dims, alpha,fc1_dims=fc1_dims,fc2_dims=fc2_dims, device=self.device)
        self.critic = CriticNetwork(input_dims, alpha,fc1_dims=fc1_dims,fc2_dims=fc2_dims, device=self.device)
        self.memory = PPOMemory(batch_size, device=self.device)
       
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

    def choose_action(self, observation, debug = False):
        #debug state output
        if debug:
            print(f'observation before fix {observation}')
            print(f'observation before fix {type(observation)}')

        if isinstance(observation, tuple):
            observation = observation[0]
        
        if debug:
            print(f'observation after fix {observation}')   

        observation = torch.tensor(np.array(observation), dtype=torch.float32)
        state = observation.to(self.actor.device)
        assert state.shape != torch.Size([0]), "Error: torch tensor has an empty shape!"
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = dist.log_prob(action)

        #probs = torch.squeeze(probs).item()
        #action = torch.squeeze(action).item()
        #value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):

            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            rewards = reward_arr
            dones = dones_arr.int()

            print_shapes = False
            if print_shapes:
                print(state_arr.shape)
                print(action_arr.shape)
                print(old_prob_arr.shape)
                print(vals_arr.shape)
                print(reward_arr.shape)
                print(dones_arr.shape)

            values = torch.cat([values, torch.zeros(1, device=self.device)])

            
            # Compute deltas including the immediate reward
            deltas = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]

            advantage = torch.zeros_like(deltas, device=self.device)
            
            # ---------------------------------------------------------------
            # old nested loop solution
            #for t in range(len(reward_arr)-1):
            #    discount = 1
            #    a_t = 0
            #    for k in range(t, len(reward_arr)-1):
            #
            #        done_switch = 1-int(dones_arr[k])
            #        rewards_diff = self.gamma*values[k+1]*done_switch - values[k]
            #        all_rewards = reward_arr[k] + rewards_diff
            #        a_t += discount*all_rewards
            #        discount *= self.gamma*self.gae_lambda
            #    advantage[t] = a_t
            #advantage = torch.tensor(advantage).to(self.actor.device)
            # ---------------------------------------------------------------

            # ---------------------------------------------------------------
            # new reverse cumulative sum with discount factors
            gae = 0
            for t in reversed(range(len(deltas))):
                gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantage[t] = gae
            # --------------------------------------------------------------

            values = values.clone().detach().to(self.actor.device)
            for batch in batches:
                states = state_arr[batch].clone().detach().to(self.actor.device)
                old_probs = old_prob_arr[batch].clone().detach().to(self.actor.device)
                actions = action_arr[batch].clone().detach().to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                #prob_ratio = new_probs.exp() / old_probs.exp()
                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio

                if torch.isnan(advantage).any() or torch.isinf(advantage).any():
                        print("Invalid values found in advantage tensor")
                if torch.isnan(prob_ratio).any() or torch.isinf(prob_ratio).any():
                    print("Invalid values found in prob_ratio tensor")
                try:
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                except:
                    print(advantage.cpu().numpy())
                    pass
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

        self.memory.clear_memory()               


