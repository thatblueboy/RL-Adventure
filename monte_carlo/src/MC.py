import random
import sys
from pprint import pprint

import gymnasium as gym

sys.path.append('RL-Adventure/_environments')
from frozen_lake_custom import FrozenLakeCustomEnv


class MCAgent():
    #assumes spaces.discrete for actiona and observation space
    def __init__(self, env):
        self.gamma = 1 #not used
        self.alpha = 0 #not used
        self.epsilon = 0.1
        self.actionSize = env.action_space.n
        self.obsSize = env.observation_space.n    
        self.env = env 

        self.actionProbs = {s: {a: 1/self.actionSize for a in range(self.actionSize)} for s in range(self.obsSize)}
        self.actionValueTable = {s: {a: 0 for a in range(self.actionSize)} for s in range(self.obsSize)}
        self.counts = {s: {a: 0 for a in range(self.actionSize)} for s in range(self.obsSize)}

        for state in self.actionProbs:
            self.improve_policy(state)

    class myenv():
        def __init__(self, env):
            super.__init__()
            self.render_mode = None  

    def train_policy(self, episodes):
        for i in range(episodes+1):
            print("training", (i/episodes)*100, "%")
            trajectory = []
            returns = 0
            self.env.reset()
            state = self.env.reset()[0]
            while True:
                action = self.policy(state)
                nextState, reward, terminated, truncated, info = self.env.step(action)
                trajectory.append([state, action, reward])
                state = nextState
                if terminated or truncated:
                    break
            trajectory.reverse()

            for state, action, reward in trajectory:
                returns += reward
                self.counts[state][action] += 1
                self.actionValueTable[state][action] += (1/self.counts[state][action])*(-self.actionValueTable[state][action] + returns)
                self.improve_policy(state)

    def policy(self, state):
        return random.choices(range(len(self.actionProbs[state])), self.actionProbs[state])[0]
    
    def deterministicPolicy(self, state):
        return self.highest_key(self.actionProbs[state])

    def improve_policy(self, state):
        best_policy = self.highest_key(self.actionValueTable[state])
        for action in self.actionProbs[state]:
            if action == best_policy:
                self.actionProbs[state][action] = 1-(self.epsilon*(self.actionSize-1)) 
            else: 
                self.actionProbs[state][action] = self.epsilon

    @staticmethod
    def highest_key(dict):
        max_value = max(dict.values())        
        max_keys = [key for key, value in dict.items() if value == max_value]
        return random.choice(max_keys) #"ties broken arbitrarily"
    
def test(env, agent, num):
    for i in range(num):
        state = env.reset()[0]
        while True:
            action = agent.deterministicPolicy(state) 
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            if terminated or truncated:
                break
    env.close()
        

if __name__ == '__main__':

    trainEnv = gym.make('FrozenLake-v1',map_name = '4x4', render_mode = 'ansi', is_slippery=False)
    # trainEnv = FrozenLakeCustomEnv(map_name='4x4', is_slippery=False)

    MCAgent = MCAgent(trainEnv)
    MCAgent.train_policy(5000)

    testEnv = gym.make('FrozenLake-v1',map_name = '4x4', render_mode = 'human', is_slippery=False)
    test(testEnv, MCAgent, 10)
    
    