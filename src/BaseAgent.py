
class BaseAgent:
    
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA
    
    def act(self, state, eval) -> int:
        raise NotImplementedError
    
    def update(self,  s, a, r, s_prim, done):
        raise NotImplementedError
        
    def reset(self):
        raise NotImplementedError
