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


class ActorNetwork(nn.Module):
    def __init__(self,n_states,n_actions,alpha,hidden_dim = 128):
        
        super(ActorNetwork,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(n_states,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,n_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(),alpha)
        self.best_dict = self.state_dict()
        
    def forward(self,x):
        dist = self.fc1(x)
        dist = torch.distributions.Categorical(dist)
        return dist
    
   
        


class CriticNetwork(nn.Module):
     
    def __init__(self,n_states,alpha,hidden_dim = 128):
        super(CriticNetwork,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(n_states,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
        self.optimizer = optim.Adam(self.parameters(),alpha)
        
    def forward(self,observation):
        value = self.fc1(observation)
        
        return value

           
class ReplayMemory():
    def __init__(self,maxlen):
        self.memory = deque([], maxlen= maxlen)    
        self.experience = namedtuple("Experience", field_names=["state", "action" , "reward", 
                                                                "next_state", "done", "old_prob",
                                                                "value"])               
        
    def append(self,state, action, reward, next_state, done, old_prob,value):
        e = self.experience(state, action, reward, next_state, done, old_prob, value)
        
        self.memory.append(e)
        
    def sample(self,sample_size):
        experience = random.sample(self.memory, sample_size)
       
        states = torch.from_numpy(np.vstack([e.state for e in experience if e is not None])).float().squeeze()
        actions = torch.from_numpy(np.vstack([e.action for e in experience if e is not None])).long().squeeze()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experience if e is not None])).float().squeeze()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experience if e is not None])).float().squeeze()
        dones = torch.from_numpy(np.vstack([e.done for e in experience if e is not None]).astype(np.uint8)).squeeze()
        old_prob = torch.tensor([e.old_prob for e in experience if e is not None]).float().squeeze()
        value = torch.tensor([e.value for e in experience if e is not None])
        #print(value)
        #time.sleep(10)
        
        return (states, actions, rewards, next_states, dones, old_prob, value)

    def __len__(self):
        return len(self.memory)
    
    def clear_memory(self):
        self.memory.clear()

class Agent:
    def __init__(self,n_states,n_actions, gamma = 0.99, alpha = 0.0003, gae_lambda = 0.95,
                 policy_clip = 0.2, batch_size=64, N= 2048, n_epochs = 10, replay_memory_size = 10000
                 ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.replay_memory = ReplayMemory(replay_memory_size)
        self.actor = ActorNetwork(n_states,n_actions, alpha)
        self.critic = CriticNetwork(n_states, alpha)
       
    def choose_action(self,observation):
        state = torch.tensor(observation,dtype=torch.float32)
        
        dist = self.actor(state)
        value = self.critic(state)
        
        action = dist.sample()
        
        probs = dist.log_prob(action)
        #print(probs)
        #time.sleep(.1)
        action = action.item()
        
        return action, probs, value
        
    def learn(self):
        for _ in range(self.n_epochs):
            states, actions, rewards, next_states, dones, old_probs, values = self.replay_memory.sample(self.batch_size)
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            
            
            next_value= torch.tensor((0.))
            advantage = torch.tensor(())
            #values = torch.squeeze(self.critic(states))
            #print(values)
            #time.sleep(4)
            #print(advantage)
            last_delta = 0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * next_value - values[t].detach()
                
                adv = last_delta = delta + self.gae_lambda * self.gamma * last_delta
                
                adv = torch.unsqueeze(adv,0)
                
                advantage = torch.cat((advantage,adv),0)
                next_value = values[t].detach()

            dist = self.actor(states)
            
            #advantage = torch.tensor(advantage)
            #
            #time.sleep(50)
            new_probs = dist.log_prob(actions)
           
            prob_ratio = new_probs.exp() / old_probs.exp()
            #print(prob_ratio)
            
            weighted_probs = advantage * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1.0-self.policy_clip, 1.0+self.policy_clip)*advantage
            actor_loss = torch.min(weighted_clipped_probs,weighted_probs).mean()
            
            critic_value = torch.squeeze(self.critic(states))
            returns = advantage + values
            vf_loss1 = (critic_value - returns).pow(2.)
            vpredclipped = values + torch.clamp(critic_value - values, -self.policy_clip,self.policy_clip)
            vf_loss2 = (vpredclipped - returns).pow(2.)
            
            critic_loss = torch.max(vf_loss1,vf_loss2).mean()
            
            entropy = dist.entropy().mean()
            
            #critic_loss = ((returns - critic_value)**2).mean()
            #critic_loss = nn.MSELoss()(returns,critic_value)
            #print(critic_loss,"criticloss")
            #print(actor_loss,"actorloss")
            #time.sleep(.1)
            total_loss = -actor_loss + critic_loss - 0.01*entropy
            #print(total_loss)
            
            total_loss.backward()
            #actor_loss.backward()
           # critic_loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()
            
        
         
        
    def train(self,env,number_of_episodes, update_rate,learning_rate):
        avg_score = 0
        n_steps = 0
        score_history = []
        learn_steps = 0
        best_score = 0
        for i in range(number_of_episodes):
            state = env.reset()[0]
            done = False
            trunc = False
            score = 0
            while not done and not trunc:
                action, prob, value = self.choose_action(torch.tensor(state))
                
                next_state, reward, done, trunc, info = env.step(action)
                score += reward
                #print("prob", prob, type(prob))
                #print("value", value, type(value))
                #time.sleep(.1)
                self.replay_memory.append(state,action,reward,next_state,done,prob,value)
                #print(n_steps)
                if n_steps % learning_rate == 0:
                    #print("prvi if",n_steps, learning_rate)
                    if len(self.replay_memory) > self.batch_size:
                        #print("drugi if",len(self.replay_memory))
                        learn_steps +=1
                        self.learn()
                        if avg_score > best_score:
                            best_score = avg_score
                            self.replay_memory.clear_memory()
                            print("update on memory")
                
                state = next_state
                n_steps +=1
            
            n_steps = 0
            score_history.append(score)
            avg_score = np.mean(score_history[-30:])
            
            if i % update_rate == 0:
                print("episode", i, "score %.1f" %score, "avg score %.1f" %avg_score, "learning steps", learn_steps )
                #time.sleep(.5)
            
env = gym.make("CartPole-v1")
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

alpha = 0.95
gamma = 0.999
number_of_episodes = 5000
update_rate = 10
learning_rate = 5

agent = Agent(n_states, n_actions)
agent.train(env,number_of_episodes,update_rate,learning_rate )

env_test = gym.make("CartPole-v1", render_mode = "human")
agent.train(env_test,3, 1, 5)