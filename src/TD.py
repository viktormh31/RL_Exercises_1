import numpy as np
from .BaseAgent import BaseAgent

class SARSA(BaseAgent):
    """
    V(S_t-1) = V(S_t-1) + alpha * [ R_t + gamma * V(S_t) - V(S_t-1) ]
    
    
    """

    
    def __init__(self, nS, nA, alpha, gamma, epsilon = 0.8):
        super().__init__(nS, nA)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = np.zeros((self.nS, self.nA))

    def act(self, state, eval = False):
        if np.random.rand() <= self.epsilon and not eval:
            return np.random.randint(self.nA)
        
        max_value = max(self.q[state])
        max_indices = [i for i, x in enumerate(self.q[state]) if x == max_value]
        #np.argmax(self.q[state])
        
        return np.random.choice(max_indices)
    
    def update(self, s, a, r, s_prim, done):
        a_prim = self.act(s)
        
        #print("a -", a)
       # print("r -", r)
        #print("s -", s)
        #print("s_prim -", s_prim)
        #print("a_prim -", a_prim)
        #print("self.q[s][a]", self.q[s][a])
        self.q[s][a] += self.alpha * (r + self.gamma * self.q[s_prim][a_prim] - self.q[s][a])
    