import time
import gymnasium as gym
import torch 
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions import Normal
import math

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

class ActorNetwork(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(ActorNetwork,self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim,128),
            #nn.ReLU()
            
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128,256),
            #nn.ReLU()
        )
        self.mean = nn.Sequential(
            nn.Linear(128,action_dim),
            nn.Tanh()
        )
        
        self.std = nn.Sequential(
            nn.Linear(128,action_dim),
            nn.Softplus()
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(256,action_dim),
            #nn.Softmax(dim=-1)
            
        )
    
    def forward(self,x):
        base = torch.relu(self.fc1(x))
        
        base = torch.relu(self.fc2(base))
        probs = self.fc3(base)
        dist = torch.distributions.Categorical(torch.softmax(probs,dim=-1))
        #print(probs)
        #time.sleep(.1)
        return dist
    
 
class CriticNetwork(nn.Module):
    def __init__(self,state_dim):
        super(CriticNetwork,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim,128),
            #nn.ReLU() 
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128,256),
            #nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(256,1)
        )
        
    def forward(self,x):
        base = torch.relu(self.fc1(x))
        
        base = torch.relu(self.fc2(base))
        value = self.value(base)
        
        return value# IMPLEMENT
    
               
class AdvantageActorCritic:
    def __init__(self,actor_network,critic_network,actor_optimizer,critic_optimizer,gamma = 0.99):
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.rewards = []
        self.I = 1.0
        self.alpha_teta = 0.99
        self.alpha_w = 0.99
        
    def train(self,env,num_episodes,update_rate):
        
        for episode in range(num_episodes):
            self.generate_episode(env)
            
            if episode % update_rate == 0:
                print(f"episode {episode} done ")
                print(f"Total rewards = {self.rewards[episode]}")
                #print(f"std is {stds_means}")
        
        
        
        
        """
        for episode in range(num_episodes):
            
            ep_rewards = []
            self.I = 1.0
            
            state = env.reset()[0]
            done = False
            stds = []
            num_sums = 0
            time_step = 0
            while not done:
                
                
                state_tensor = torch.tensor(state,dtype=torch.float32)
                mean, std = self.actor_network(state_tensor)
                dist = Normal(loc=mean,scale=std + 1e-3)
                #print(dist)
                #time.sleep(1)
                #print(dist)
                action_x = dist.rsample()
                action_y = torch.tanh(action_x) * 2
                #print(action)
                log_prob = dist.log_prob(action_x)
                log_prob = log_prob - torch.sum(torch.log(2 * (1-action_y.pow(2)+ 1e-6)),dim = -1 , keepdim = True)
                stds.append(std)
                num_sums += 1
                #p1 = -((mean - action)**2) / (2*std.clamp(min = 1e-3))
                #p2 = -torch.log(torch.sqrt(2*math.pi*std))
                #calc_log_prob = p1+p2
                
                #log_prob = log_prob.sum()
                action_numpy = action_y.detach().numpy()
                #print(action_numpy)
                time_step += 1
                next_state, reward, done, trunc, _ = env.step(action_numpy)
                done = done or time_step > 200
                reward = torch.tensor(reward)
                ep_rewards.append(reward)

                next_state_tensor = torch.tensor(next_state, dtype = torch.float32)
                
                value_state = self.critic_network(state_tensor)
                #if not done :
                value_next_state = self.critic_network(next_state_tensor) 
                #else :
                    #value_next_state = torch.tensor(0.0, dtype=torch.float32)
                    
            
                #with torch.no_grad():
                delta = reward + self.gamma * value_next_state * (1-int(done)) - value_state

                #critic_loss = -self.alpha_w *delta.detach() * value_state
                critic_loss = delta**2
                loss = delta.detach() * log_prob
                loss = loss.mean()
                #actor_loss = -self.alpha_teta * self.I * delta.detach() * calc_log_prob
                actor_loss = -self.alpha_teta*self.I*loss
              
                
                
                
                
                
                #actor_loss = 1 
                
                
                self.actor_optimizer.zero_grad()
                #torch.nn.utils.clip_grad_norm_(actor_network.parameters(), 1)
                actor_loss.backward()
                self.actor_optimizer.step()
                #-self.alpha_teta* self.I * delta
                
                #critic update
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                #action = torch.from_numpy(action)
                #log_prob_current = dist.log_prob(action)
                #print(f"log_prob {log_prob_current} {type(log_prob_current)}")
               # log_tensor = torch.tensor(log_prob_current)
                #print(f"log_prob {log_tensor} {type(log_tensor)}")
                #agent update
                
                
                self.I *= self.gamma
                
                
                
                state = next_state
            print(sum(stds)/num_sums)
            #stds_means.append(sum(stds)/num_sums)
            
            if episode % 30 == 0:
                print(f"episode {episode} done ")
                print(f"Total rewards = {sum(ep_rewards)}")
                #print(f"std is {stds_means}")
            self.rewards.append(sum(ep_rewards))
        """        
    
    def generate_episode(self,env):
        
        
        done = False
        trunc = False
        current_state = env.reset()[0]
        
        rewards = []
        values = []
        #next_values = []
        log_probs = []  
        masks = []
              
        while not done and not trunc:
            
            current_state_tensor = torch.tensor(current_state,dtype=torch.float32)
            #probs = self.actor_network(current_state_tensor)
            #dist = torch.distributions.Categorical(probs)
            dist = self.actor_network(current_state_tensor)
            action = dist.sample()
            
            next_state, reward, done, trunc, _ = env.step(action.cpu().numpy())
            done = done or trunc
            #action_tensor = torch.tensor(action,dtype=torch.float32)
            
            
            
            log_prob_current = dist.log_prob(action)
            log_probs.append(log_prob_current)
            masks.append(torch.tensor([1-done],dtype=torch.float32))
            value_current_state = self.critic_network(current_state_tensor)
            values.append(value_current_state)
            #value_next_state = self.critic_network(next_state_tensor)
            #next_values.append(value_next_state)
            
            
            
            rewards.append(reward)
            current_state = next_state
            
           
        advantages = []
        adj_returns = []
        next_state_tensor = torch.tensor(next_state,dtype=torch.float32)
        
        next_value = self.critic_network(next_state_tensor)
        R = next_value
        
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1-int(done))
            done = False
            adj_returns.insert(0,R)
        
        for adj_return, value in zip(adj_returns, values):
            
            advantage = adj_return - value
            advantages.append(advantage)      
        
        log_probs = torch.stack(log_probs)
        advantages = torch.stack(advantages)
        
        actor_loss = -(advantages.detach() * log_probs).mean()
        critic_loss = advantages.pow(2).mean()
        #print(f"returns  - actor_loss {actor_loss.item()} - critic_loss {critic_loss.item()}")
        #time.sleep(1)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        
            
        self.rewards.append(sum(rewards))
    
    
    
        
        
env = gym.make("LunarLander-v2")
env.reset()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"state dim - {state_dim}, action dim {action_dim}")

learning_rate = 0.001
number_of_episodes = 5000

actor_network = ActorNetwork(state_dim,action_dim)
critic_network = CriticNetwork(state_dim)

actor_optimizer = torch.optim.Adam(actor_network.parameters(),learning_rate)
critic_optimizer = torch.optim.Adam(critic_network.parameters(),learning_rate)

agent = AdvantageActorCritic(actor_network,critic_network,actor_optimizer,critic_optimizer)
agent.train(env,number_of_episodes, 50)
plt.plot(agent.rewards)
plt.show()

print("done")

env_test = gym.make("LunarLander-v2", render_mode = "human")
agent.train(env_test, 3,1)

        
        