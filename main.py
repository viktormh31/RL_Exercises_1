from src.BaseAgent import BaseAgent
from src.TD import SARSA
import time
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
import numpy as np


def train_one_episode(agent : BaseAgent, env : gym.Env, render = False, eval = False):
    rewards = []
    
    done = False
    pickedLeft = 0
    s, _ = env.reset()
    steps = 0
    while not done:
        a = agent.act(s, eval)
        s_prim, r, done, trunc, _ = env.step(a)
        #if done:
            #print(r)
            #time.sleep(1)
            
        agent.update(s, a, r, s_prim, done)
        s = s_prim
        steps += 1
       
        rewards.append(r) 
        #if r == 1:
            #print("nagrada", r)
            #time.sleep(1)
            #print(rewards)
        if render:
            rgb = env.render()
            plt.imshow(rgb)
            plt.show(block=False)
            plt.pause(0.5)
            
        if steps > 1000:
            break
    
    
    plt.close()
    print("number of steps ",steps)
    
    return rewards

def main():
    
    num_episodes = 50000
    num_max_steps = 1000
    sum_rew = 0
    
    
    env = gym.make('FrozenLake-v1', render_mode="rgb_array", desc=generate_random_map(size=7, p=0.7), is_slippery=False)    

    nS = env.observation_space.n
    nA = env.action_space.n
    
    agent = SARSA(nS=nS, nA=nA, gamma=.99, alpha=0.9)
    ep_reward = 0
    k = 1
    for i in range(num_episodes):
        print(i)
        agent.epsilon = k
        if i % 100 == 0 and k > 0.1:
            k -= 100/num_episodes
        print(agent.epsilon)
        ep_reward = train_one_episode(agent, env, render=False, eval=False)
        sum_rew += np.sum(ep_reward)
        #print("episoda gotova : ",i)
        #print(agent.q)
        #if sum(ep_reward) > 0:
            
            #break
        
    print("sum of rewards: ",sum_rew)    
    
    #print("gotov - ", ep_reward)
    time.sleep(1)        
    x = train_one_episode(agent, env, render=True, eval = True)
    print(agent.q)
    #print(np.sum(x))
   
if __name__ == "__main__":
    main()
