# from REINFORCE.environments.frozen_lake_custom import FrozenLakeCustomEnv

import random 
import numpy as np

import sys
sys.path.append('RL-Adventure/_environments')
from frozen_lake_custom import FrozenLakeCustomEnv

class VPGAgent():
    def __init__(self, env):
        self.GAMMA = 1
        self.alpha = 0.1
        self.actionSize = env.action_space.n
        self.obsSize = env.observation_space.n
        self.pi = {s: {a: 1/self.actionSize for a in range(self.actionSize)} for s in range(self.obsSize)}
        self.thetas = {s: {a: 0 for a in range(self.actionSize)} for s in range(self.obsSize)}
        self.env = env

    def train_agent(self, episodes):
        for episodes in range(episodes):
            self.env.reset()
            trajectory = []
            state = self.env.reset()[0]
            while True:
                action = self.policy(state)
                new_state, reward, terminated, truncated, info = self.env.step(action)
                trajectory.append([state, action, reward])
                state = new_state
                if terminated or truncated:
                    break
            self.update_policy(trajectory)

    def update_policy(self, traj):
            T = len(traj)
            for t in range(T):
                state, action, reward = traj[t]
                G = self.findDiscountedReturn(traj, t, T)
                score = self.actionToVector(action)-self.pi[state]
                self.thetas[state] = self.thetas[state] + self.alpha*score*G
                # print(self.thetas[state])
                self.update_pi(state)
        
    def update_pi(self, state):
        row = self.thetas[state, :]
        row = np.exp(row)
        for j in range(self.thetas.shape[1]):
            self.pi[state][j] = row[j]/row.sum()  # softmax

    def policy(self, state):
        return random.choices(range(len(self.pi[state])), self.pi[state])[0]
    
    def deterministicPolicy(self, state):
        return self.highest_key(self.actionProbs[state])
    
    def findDiscountedReturn(self, traj, t, T):
      G = 0
      for i in range(t, T):
          G = G + traj[i][2]*(self.GAMMA)**(i-t)
      return G
        
    @staticmethod
    def highest_key(dict):
        return max(zip(dict.values(), dict.keys()))[1]
    
    @staticmethod
    def actionToVector(action):
        vector = np.array([0, 0, 0, 0])
        vector[action] = 1
        return vector
    
  

def test(env, agent, episodes):
    for i in range(episodes+1):
        print("testing", (i/episodes)*100, "%")
        returns = 0
        env.reset()
        state = env.reset()[0]
        while True:
            action = agent.policy(state)
            nextState, reward, terminated, truncated, info = env.step(action)
            returns += reward
            state = nextState
            if terminated or truncated:
                break
        # print(returns)

if __name__ == "__main__":
    trainEnv = FrozenLakeCustomEnv(desc=None,map_name='4x4', is_slippery=False)
    VPGAgent = VPGAgent(trainEnv)
    VPGAgent.train_agent(1000)

    testEnv = FrozenLakeCustomEnv(desc=None,map_name='4x4', is_slippery=False, render_mode='human')
    test(testEnv, VPGAgent, 1000)