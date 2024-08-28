import torch
import numpy as np
import gymnasium as gym
import time 
import torch.nn as nn
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt

BATCH_SIZE = 128
EPSILON = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.998
SYNC_RATE = 5
OPTIMIZE_RATE = 5





class DQ_Network:
    def __init__(self,n_actions,n_states,hidden_dim):
        super(DQ_Network,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(n_states,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU()
            
        )
     
        self.fc2 = nn.Linear(hidden_dim,n_actions)

    def forward(self,x):
        x = self.fc1(x)
        
        return self.fc2(x)
   
    

class DDQN_Agent:
    def __init__(self,q_network_1,q_network_2,policy_network, optimizer, gamma, replay_memory_size, n_actions, n_states):
        self.q_network_1 = q_network_1
        self.q_network_2 = q_network_2
        self.policy_network = policy_network
        self.optimizer = optimizer
        self.gamma = gamma
        self.n_states = n_states
        self.n_actions = n_actions
        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory(replay_memory_size)
    
    def choose_action(self,observation):
        
        with torch.no_grad():
            probs_1 = self.q_network_1(observation)
            probs_2 = self.q_network_2(observation)
            
        
        
        
        
        
        
        self.policy_network.eval()
        with torch.no_grad():
            probs = self.policy_network(observation)
        self.policy_network.train()
        
        if random.random() < EPSILON:
            action = np.random.choice(np.arange(self.n_actions))
        else:
            action = np.argmax(probs.cpu().detach().numpy())
            
        return action
        
    def one_episode(self,env):
        
        state = env.reset()[0]
        done = False
        trunc = False
        time_step = 0
        rew = []
        
        while not done and not trunc:
            state_tensor = torch.FloatTensor(state)
            action = self.choose_action(state_tensor)
            next_state, reward, done, trunc, _ = env.step(action)
            
            
            self.memory.append(state,action,reward, next_state, done)

            if time_step % OPTIMIZE_RATE == 0:
                if len(self.memory) >= BATCH_SIZE:
                    mini_batch = self.memory.sample(BATCH_SIZE)
                    self.optimize(mini_batch)
        time_step += 1
        rew.append(reward)
        state = next_state

        return sum(rew)
    
    def train(self,env,num_of_episodes):
        ploting_rewards = []
        for episode in range(num_of_episodes):
            one_ep_reward = self.one_episode(env)
            ploting_rewards.append(one_ep_reward)
            
            
            
            
            
    
    def optimize(self,mini_batch):
        states, actions, rewards, next_states, dones = mini_batch
        
        choice = random.choice((1,2))
        if choice == 1:
            q_target = self.q_network_2(next_states).detach().max(axis=1).unsqueeze(1)
            y_j = rewards + self.gamma * q_target *(1-dones)
            q_eval = self.q_network_1(states).gather(actions)
            
            loss = self.criterion(q_eval,y_j)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.sync_network_to_1()
            
        else:
            q_target = self.q_network_1(next_states).detach().max(axis=1).unsqueeze(1)
            y_j = rewards + self.gamma * q_target *(1-dones)
            q_eval = self.q_network_2(states).gather(actions)
            
            loss = self.criterion(q_eval,y_j)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.sync_network_to_2()
            
            
    def sync_network_to_1(self):
        for eval_param, target_param in zip(self.policy_network.parameters(), self.q_network_1.parameters()):
            target_param.data.copy_(1e-2*eval_param.data + (1.0-1e-2)*target_param.data)
        
        pass
    def sync_network_to_2(self):
        for eval_param, target_param in zip(self.policy_network.parameters(), self.q_network_2.parameters()):
            target_param.data.copy_(1e-2*eval_param.data + (1.0-1e-2)*target_param.data)
        
        pass
        
class ReplayMemory():
    def __init__(self,maxlen) -> None:
        self.memory = deque([],maxlen)
        self.experience = namedtuple("Experience", field_names=["state", "action" , "reward", "next_state", "done"])

    def append(self,state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self,sample_size):
        experience = random.sample(self.memory, sample_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experience if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experience if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experience if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experience if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experience if e is not None]).astype(np.uint8))
        
        
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)







