import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random

CHECKPOINT_DIR_PATH = "/home/viktor/Documents/DIPLOMSKI/reinforcement learning/RL_Exercises_1/models/ddpg"

class ReplayBuffer(object):
    def __init__(self, max_size):
        self.mem_size = max_size
        
        self.state_memory = deque([],self.mem_size)
        self.next_state_memory = deque([],self.mem_size)
        self.reward_memory = deque([],self.mem_size)
        self.done_memory = deque([],self.mem_size)
        self.action_memory = deque([],self.mem_size)
        
    def append(self,state,action,reward,next_state, done):
        self.state_memory.append(state)
        self.next_state_memory.append(next_state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.done_memory.append(done)
        
    def sample(self,batch_size):
        max_memory = len(self.state_memory)
        
        batch = np.random.choice(max_memory,batch_size, replace = False)
        print(batch)
        print(self.state_memory)
        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.done_memory[batch]
        
        return states, actions, rewards, next_states, dones

class ReplayMemory(object):
    def __init__(self,max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.next_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.done_memory = np.zeros(self.mem_size, dtype= np.float32)


    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward 
        self.next_state_memory[index] = next_state
        self.done_memory[index] = done
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        
        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        dones = self.done_memory[batch]
        
        return states, actions, rewards, next_states, dones








class CriticNetwork(nn.Module):
    def __init__(self,beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir= CHECKPOINT_DIR_PATH):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.beta = beta
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        
        
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1,f1)
        nn.init.uniform_(self.fc1.bias.data, -f1,f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2,f2)
        nn.init.uniform_(self.fc2.bias.data, -f2,f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        f3 = 0.003
        
        self.q = nn.Linear(self.fc2_dims, 1)
        nn.init.uniform_(self.q.weight.data, -f3, f3)
        nn.init.uniform_(self.q.bias.data, -f3, f3)
        
        self.optimizer = optim.Adam(self.parameters(), lr = beta)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, state, action):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        #print(action)
        #print(self.action_value(action))
        
        #time.sleep(5)
        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(torch.add(x,action_value))
        state_action_value = self.q(state_action_value)
        
        return state_action_value
    
    def save_checkpoint(self):
        print("----- saving model -----")
        torch.save(self.state_dict(),self.checkpoint_file)
        
    def load_checkpoint(self):
        print("----- loading model -----")
        self.load_state_dict(torch.load(self.checkpoint_file))
        
class ActorNetwork(nn.Module):
    def __init__(self,alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir = CHECKPOINT_DIR_PATH):
        super(ActorNetwork,self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.alpha = alpha
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_file = os.path.join(chkpt_dir, name+"_ddpg")
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1,f1)
        nn.init.uniform_(self.fc1.bias.data, -f1,f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2,f2)
        nn.init.uniform_(self.fc2.bias.data, -f2,f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        
        f3= 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        nn.init.uniform_(self.mu.weight.data, -f3,f3)
        nn.init.uniform_(self.mu.bias.data, -f3, f3)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self,state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.tanh(self.mu(x))
        
        return x
        
    def save_checkpoint(self):
        print("----- saving model -----")
        torch.save(self.state_dict(),self.checkpoint_file)
        
    def load_checkpoint(self):
        print("----- loading model -----")
        self.load_state_dict(torch.load(self.checkpoint_file))
            
        
        
class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, n_actions, gamma = 0.99,
                 max_size = 1000000, layer1_size = 256, layer2_size = 256, batch_size = 64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayMemory(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        
        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size,
                                  n_actions=n_actions,name="Actor")
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size,
                                  n_actions=n_actions,name="Target_Actor")
        self.critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                    n_actions=n_actions, name="Critic")
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                    n_actions=n_actions, name="Target_Critic")
        
        self.update_network_parameters(tau=1)
        
    def choose_action(self, obs):
        self.actor.eval()
        obs = torch.tensor(obs,dtype=torch.float32).to(self.actor.device)
        mu = self.actor(obs).to(self.actor.device)
        noise = torch.distributions.Normal(0,0.1)
        mu_prime = mu + noise.sample().to(self.actor.device)
        self.actor.train()
        
        return mu_prime.cpu().detach().numpy() * 2
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state,action,reward,next_state,done)
        
    def learn(self):
        if len(self.memory.state_memory) < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        
        state = torch.tensor(state, dtype= torch.float32).to(self.critic.device)
        action= torch.tensor(action, dtype= torch.float32).to(self.critic.device)
        reward = torch.tensor(reward, dtype= torch.float32).to(self.critic.device)
        next_state = torch.tensor(next_state, dtype= torch.float32).to(self.critic.device)
        done = torch.tensor(done, dtype= torch.float32).to(self.critic.device)
        
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        
        target_actions = self.target_actor(next_state)
        critic_next_value = self.target_critic(next_state,target_actions)
        critic_value = self.critic(state,action)
        
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_next_value[j]*(1-int(done[j])))
            
        #target = reward + self.gamma*critic_next_value*done
        target = torch.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size,1)
        
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        
        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        
        self.update_network_parameters()
        
    def update_network_parameters(self,tau = None):
        if tau is None:
            tau = self.tau
            
        
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        
        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)
        
        for name in critic_state_dict:
            critic_state_dict[name] = tau* critic_state_dict[name].clone() \
                                      +(1-tau)*target_critic_dict[name].clone()
        
        self.target_critic.load_state_dict(critic_state_dict)
        
        for name in actor_state_dict:
            actor_state_dict[name] = tau* actor_state_dict[name].clone() \
                                      +(1-tau)*target_actor_dict[name].clone()
        
        self.target_actor.load_state_dict(actor_state_dict)
        
        
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()
        
    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()
        
        
env = gym.make("Pendulum-v1",render_mode = "human")
input_dims = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

print(input_dims, n_actions)

agent = Agent(alpha=0.000025, beta = 0.00025, input_dims=input_dims, tau = 0.01, 
              env=env, n_actions = n_actions,layer1_size=400,layer2_size=300,batch_size=64)

np.random.seed(0)
num_of_episodes = 1000
score_history = []

for i in range(num_of_episodes):
    done = False
    trunc = False
    state = env.reset()[0]
    score = 0
    
    while not done and not trunc:
        action = agent.choose_action(state)
        next_state, reward, done, trunc, _ = env.step(action)
        agent.remember(state,action,reward,next_state,done)
        agent.learn()
        score += reward
        state = next_state

    score_history.append(score)
    print("episode" , i, "score %.2f" % score, "100 game average %.2f" % np.mean(score_history[-100:])
          )

    if i %25 == 0:
        agent.save_models()
        
        
plt.plot(score_history)
plt.show()



