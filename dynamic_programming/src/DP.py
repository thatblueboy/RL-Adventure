import gymnasium as gym
import numpy as np
import sys
from copy import deepcopy
from random import randrange
from pprint import pprint

# import _environments.frozen_lake_custom as flc
# from ...environments.frozen_lake_custom import FrozenLakeCustomEnv

sys.path.append('environments')
from frozen_lake_custom import FrozenLakeCustomEnv



class DPAgent():

    def __init__(self, env, deterministic = True):
        self.determistic = deterministic
        self.GAMMA = 1

        self.transitionTable = deepcopy(env.P)
        self.deterministicPolicy = deepcopy(env.P)
        self.stateValueTable = deepcopy(env.P)

        #emptying copies to initialize policy and value function table
        for state in self.deterministicPolicy:
            for action in self.deterministicPolicy[state]:
                self.deterministicPolicy[state][action] = 0
            self.deterministicPolicy[state][randrange(4)] = 1 

        for state in self.stateValueTable:
            self.stateValueTable[state] = randrange(10)

    def reset(self):
        self.transitionTable = deepcopy(env.P)
        self.deterministicPolicy = deepcopy(env.P)
        self.stateValueTable = deepcopy(env.P)

        #emptying copies to initialize policy and value function table
        for state in self.deterministicPolicy:
            for action in self.deterministicPolicy[state]:
                self.deterministicPolicy[state][action] = 0
            self.deterministicPolicy[state][randrange(4)] = 1 

        for state in self.stateValueTable:
            self.stateValueTable[state] = randrange(10)

    def evaluate_policy(self):
        for i in range(100):
            for state in self.stateValueTable:
                value = 0
                for action in self.deterministicPolicy[state]:
                    actionProb = self.deterministicPolicy[state][action]
                    for stateProb, nextState, reward, done in self.transitionTable[state][action]:
                        value += actionProb*stateProb*(reward + self.GAMMA*self.stateValueTable[nextState])
                self.stateValueTable[state] = value
        
    def evaluate_policy_old(self):
        for i in range(100):
            for state in self.stateValueTable:
                bestAction = self.highest_key(self.deterministicPolicy[state])
                _, nextState, reward, _ = self.transitionTable[state][bestAction][0][1]
                # Reward = self.transitionTable[state][bestAction][0][2]
                nextStateValue = self.stateValueTable[nextState]
                self.stateValueTable[state] = reward + self.GAMMA*nextStateValue

    def improve_policy(self):
        for state in self.deterministicPolicy:
            actionValues = np.zeros(4)
            for action in self.deterministicPolicy[state]:
                _, nextState, reward, _ = self.transitionTable[state][action][0]
                nextValueFunction = self.stateValueTable[nextState]
                actionValues[action] = nextValueFunction +reward
            bestAction = np.argmax(actionValues)

            for action in self.deterministicPolicy[state]:
                if action == bestAction:
                    self.deterministicPolicy[state][action] = 1
                else:    
                    self.deterministicPolicy[state][action] = 0

    def stochastic_start(self):
        for state in self.deterministicPolicy:
            for action in self.deterministicPolicy[state]:
                self.deterministicPolicy[state][action] = 0.25
        self.evaluate_policy()

    def policy(self, state):
        return self.highest_key(self.deterministicPolicy[state])

    @staticmethod
    def highest_key(dict):
        return max(zip(dict.values(), dict.keys()))[1]
    
def test(env, agent):
    for episodes in range(1):
            initialInfo = env.reset()
            state = initialInfo[0]
            for timestamp in range(10):
                action = agent.policy(state)
                newState, reward, terminated, truncated, info = env.step(action)
                env.render()
                state = newState
                if terminated or truncated:
                    break

if __name__ == '__main__':
    # print(sys.path)
    # sys.path.append('RL/RL-Adventure/_environments')
    # print(sys.path)
    # import frozen_lake_custom

    env = FrozenLakeCustomEnv(map_name='4x4', is_slippery=False, render_mode='human')
    # env = gym.make('FrozenLake-v1', render_mode = 'human')
    DPAgent = DPAgent(env)

    DPAgent.stochastic_start()
    DPAgent.improve_policy()

    for i in range(2):
        DPAgent.evaluate_policy()
        DPAgent.improve_policy()
        pprint(DPAgent.stateValueTable)
        
    test(env, DPAgent)

    
 
    



        
        
       