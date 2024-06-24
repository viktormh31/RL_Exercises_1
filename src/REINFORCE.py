import time
import gymnasium as gym
import torch 
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim,128),
            
            nn.ReLU(),
            
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
        )
        
        self.mean = nn.Sequential(
           nn.Linear(128,action_dim),
           #nn.Tanh(),
            
            
        )
        self.std = nn.Sequential(
           nn.Linear(128,action_dim),
           #nn.Softmax(dim=-1),
        
        )
        
        self.tahn = nn.Tanh()
        
        """
        x = nn.functional.relu(self.fc1(x))
        mean = self.fc_mean(x)
        mean = nn.Tanh(mean)
        log_std = self.fc_log_std(x)
        log_std = nn.functional.softplus(log_std)
        std = torch.exp(log_std)
        """
        
    def forward(self, x):
       
        base = self.fc1(x)
        base = self.fc2(base)
        mean = torch.tanh(self.mean(base))
        log_std = self.std(base)
        #std = torch.clamp(log_std, min = -20, max = 20)
        std = torch.exp(log_std)
        std = torch.clamp(std, 1e-3,2.0)
        #print(f"mean je {mean}, std je {std}")
        
        
        return mean, std
    
class REINFORCE:
    def __init__(self,policy, optimizer, gamma= 0.99):
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma
    
    def generate_episode(self,env, render = False):
        states, actions, rewards = [], [], []
        state = env.reset()[0]
        #(f"state --- {state}")
        #time.sleep(1)
        done = False
        time_step = 0
        while not done and time_step < 200:
            time_step += 1
            #print(time_step)
            state_tensor = torch.tensor(state, dtype= torch.float32).to(device)
            mean, std = self.policy(state_tensor)
            #print(f"prvi std je {std}")
            action = torch.normal(mean,std).detach().to(device)
            action = torch.clamp(action, -2.0, 2.0).cpu().numpy()
            next_state, reward, done,_,_ = env.step(action)
            #print(done)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            
            
        #print(states[0], actions[0], rewards[0])
        return states, actions, rewards
    
    def compute_returns(self,rewards):
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma*G
            returns.insert(0,G)
        #print(returns)
        returns = torch.tensor(returns, dtype= torch.float32).to(device)
        
        
        return (returns - returns.mean())/ (returns.std() + 1e-9)
        # return the normalized returns which gives us values with mean = 0 and std = 1
        
        
    def update_policy(self,states,actions,returns):
    
        for state, action, G_t in zip(states,actions,returns):
            
            state_tensor = torch.tensor(state, dtype= torch.float32).to(device)
            action_tensor = torch.tensor(action, dtype= torch.float32).to(device)
            G_t_tensor = torch.tensor(G_t, dtype= torch.float32).to(device)
            #print(state_tensor)
            mean, std = self.policy(state_tensor)
            
            #print(mean,std)
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action_tensor).sum()# probaj bez sum()-------------
            loss = -log_prob*G_t_tensor
            #time.sleep(.1)
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(),max_norm=5)
            self.optimizer.step()
          
    def train(self, env, num_episodes):
        
        for episode in range(num_episodes):
            
            states, actions, rewards = self.generate_epizode(env)
            returns = self.compute_returns(rewards)
            self.update_policy(states, actions, returns)
            if episode % 100 == 0:
                print(f"episode {episode} done ")
                print(f"Total rewards = {sum(rewards)}")
              
        
 
num_episodes = 20000
learning_rate = 0.01


env = gym.make("Pendulum-v1")
env.reset()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

policy = PolicyNetwork(state_dim,action_dim).to(device)
optimizer = torch.optim.SGD(policy.parameters(), learning_rate)
agent = REINFORCE(policy,optimizer)





agent.train(env,num_episodes)

agent.generate_episode(env,True)

env_test = gym.make("Pendulum-v1", render_mode = "human")

agent.generate_episode(env_test)
