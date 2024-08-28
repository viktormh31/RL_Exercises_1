import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np
from collections import deque, namedtuple
import random
import time
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


class DQN(nn.Module):
    def __init__(self,n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations,64)
        self.layer2 = nn.Linear(64,64)
        self.layer3 = nn.Linear(64, n_actions)
        f1 = 1 / np.sqrt(self.layer1.weight.data.size()[0])
        nn.init.uniform_(self.layer1.weight.data, -f1,f1)
        nn.init.uniform_(self.layer1.bias.data, -f1,f1)
        self.bn1 = nn.LayerNorm(64)
        
        f2 = 1 / np.sqrt(self.layer2.weight.data.size()[0])
        nn.init.uniform_(self.layer2.weight.data, -f2,f2)
        nn.init.uniform_(self.layer2.bias.data, -f2,f2)
        self.bn2 = nn.LayerNorm(64)
        
        f3 = 0.003
        
  
    def forward(self, x):
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        
        return self.layer3(x)
    
    

class DQN_Agent():
    def __init__(self,policy_dqn, target_dqn,optimizer,gamma,replay_memory_size,n_actions,n_states):
        self.policy_dqn = policy_dqn
        self.target_dqn = target_dqn
        self.optimizer = optimizer
        self.gamma = gamma
        self.memory = ReplayMemory(replay_memory_size)
        self.n_actions = n_actions
        self.n_states = n_states
        
        self.batch_size = 128
        self.epsilon = 1.0
        self.loss_f = nn.MSELoss()
        self.sync_rate = 5
        self.step = 0
        self.rewards = []
        self.los = 0
        self.epsilon_decay = 0.9988
    def choose_action(self,observation):
        #observation_tensor = torch.FloatTensor(observation)
        self.policy_dqn.eval()
        with torch.no_grad():
            probs = self.policy_dqn(observation)
        self.policy_dqn.train()
        #action = probs.argmax().item()
        
        if random.random() > self.epsilon:
            action = np.argmax(probs.cpu().detach().numpy())
               
        else:
            action = random.choice(np.arange(self.n_actions))
        
        return action
        
    
    def train(self,env,num_of_episodes, update_rate):
        reww = []
        i = 0
        for episode in range(num_of_episodes):
            x = self.one_episode(env)
            reww.append(x)
            
            if len(self.memory) > self.batch_size:
                
                
                self.step += 1
                self.epsilon = max(self.epsilon *self.epsilon_decay, 0.05)
                
                if self.step > self.sync_rate:
                    #self.optimize(mini_batch)
                    self.sync_networks()
                    self.step = 0
            if episode % update_rate == 0:
                print(f"Episode {episode}, rewards {sum(reww)/update_rate}, epsilon {self.epsilon}")
                self.rewards.append(sum(reww)/update_rate)
                reww = []
                
    
    def one_episode(self,env):
        
        state = env.reset()[0]
        done = False
        trunc = False
        #print("usao")
        time_step = 0
        rew = []
        while not done and not trunc:
            time_step+= 1
            state_tensor = torch.FloatTensor(state)
            action = self.choose_action(state_tensor)

            next_state, reward, done, trunc ,_ = env.step(action)
            #next_state_tensor = torch.FloatTensor(next_state)
            self.memory.append(state,action,reward,next_state,done)
            
            if time_step % self.sync_rate == 0:
                if len(self.memory) >= self.batch_size:
                    mini_batch = self.memory.sample(self.batch_size)
                    self.optimize(mini_batch)
                    
            
            rew.append(reward)
            state = next_state
  
        #self.rewards.append(sum(rew))
        return sum(rew)
        
    def sync_networks(self):
         for eval_param, target_param in zip(self.policy_dqn.parameters(), self.target_dqn.parameters()):
            #target_param.data.copy_(1e-3*eval_param.data + (1.0-1e-3)*target_param.data)
            target_param.data.copy_(eval_param.data)
    
    def optimize(self, mini_batch):
        
        
        states, actions, rewards, next_states, dones = mini_batch
        
        
        
        
        #q_values = self.target_dqn(next_states).detach()
        #q_target = torch.tensor([q_values[i][actions[i]] for i in range(len(q_values))]).detach().unsqueeze(1)
        
        
        ##
        """
        --Double DQN algorithm--
        The only change is that for q_target we are using different calculation in which 
        we calculate q_values from policy network and pick q_value based on action chosen by 
        target_network rather than action with highest q_value from policy_network calculations.
        
        Could be optimised more, but it works for now
        
        target_q_values = self.policy_dqn(next_states).detach()
        actionsa= torch.argmax(target_q_values,-1).unsqueeze(1)
        policy_values = self.target_dqn(next_states)
        q_target= torch.tensor([policy_values[i][actionsa[i]] for i in range(len(policy_values))]).unsqueeze(1)
        """
       
        
        #q_target = self.target_dqn(next_states).detach().max(axis=1)[0].unsqueeze(1) #double dqn?
        q_target = self.policy_dqn(next_states).detach().max(axis=1)[0].unsqueeze(1)  #vanila dqn?
        y_j = rewards + self.gamma * q_target * (1-dones)
        q_eval = self.policy_dqn(states).gather(1,actions)
        
        loss = self.loss_f(q_eval, y_j)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.sync_networks()
       
       
    def test(self,env,num_of_episodes):
        time.sleep(1)
        
        for episode in range(num_of_episodes):
            
            state = env.reset()[0]
            done = False
            rew = []
            while not done:
                if random.random() > self.epsilon:
                    state_tensor = torch.FloatTensor(state)
                    action = self.choose_action(state_tensor)
                   # print("if")
                else:
                   # print("else")
                    action = env.action_space.sample()
                
                next_state, reward, done, trunc ,_ = env.step(action)
                done = done or trunc
                #self.memory.append((state,action,reward,next_state,done))
                rew.append(reward)
                state = next_state
            print("Total reward:",sum(rew))
        
           
class ReplayMemory():
    def __init__(self,maxlen):
        self.memory = deque([], maxlen= maxlen)    
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




env = gym.make("LunarLander-v2", continuous = False)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

learning_rate = 1e-3
gamma = 0.99
replay_memory_size = 50000

policy_dqn = DQN(n_states,n_actions)
target_dqn = DQN(n_states,n_actions)

target_dqn.load_state_dict(policy_dqn.state_dict())
optimizer = optim.Adam(policy_dqn.parameters(),learning_rate)

agent = DQN_Agent(policy_dqn,target_dqn,optimizer,gamma,replay_memory_size,n_actions, n_states)

agent.train(env,3000,10)

plt.plot(agent.rewards)
plt.show()

env_test = gym.make("LunarLander-v2", continuous = False, render_mode ="human")
agent.test(env_test, 5)

