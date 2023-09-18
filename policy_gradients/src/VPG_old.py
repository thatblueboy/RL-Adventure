import gymnasium as gym
import numpy as np
import random

# 0: LEFT
# 1: DOWN
# 2: RIGHT
# 3: UP

thetas = np.zeros((16, 4), dtype=float)
pi = np.zeros((16, 4), dtype=float)
pi[pi == 0] = 0.25
# print(pi)
NUMBER_OF_STATES = 16
NUMBER_OF_ACTIONS = 4
ALPHA = 0.5
GAMMA = 0.1
EPISODES = 100
Returns = []

def getReward(observation):
    if 0 <= observation <= 12:
        return 1
    if observation == 13 or observation == 14:
        return 0
    if observation == 15:
        return 10
    
def findDiscountedReturn(traj, t, T):
    G = 0
    for i in range(t, T):
        G = G + traj[i][2]*(GAMMA)**(i-t)
    return G

def updatePI(state):
    row = thetas[state, :]
    row = np.exp(row)
    for j in range(thetas.shape[1]):
        pi[state][j] = row[j]/row.sum()  # softmax

def actionToVector(action):
    vector = np.array([0, 0, 0, 0])
    vector[action] = 1
    return vector
    
def update(traj):
    print(traj)
    T = len(traj)
    for t in range(T):
        state, action, reward = traj[t]
        G = findDiscountedReturn(traj, t, T)
        score = actionToVector(action)-pi[state]
        thetas[state] = thetas[state] + ALPHA*score*G
        print(thetas[state])
        updatePI(state)

def policy(state):
    action = np.random.choice([0, 1, 2, 3], 1, p = pi[state])
    return int(action)

def deterministicPolicy(state):
    prob = max(list(pi[state]))
    return list(pi[state]).index(prob)

# DISPLAY RANDOM POLICY
env = gym.make("FrozenLake-v1", desc=["FFFF", "FFFF","FFFF", "SHHG"], render_mode="human", is_slippery=False)
for episodes in range(5):
    env.reset()
    state = 12
    for timestamp in range(100):
        action = policy(state)
        new_state, reward, terminated, truncated, info = env.step(action)
        env.render()
        state = new_state
        if terminated or truncated:
            break


# TRAIN
env = gym.make("FrozenLake-v1", desc=["FFFF", "FFFF","FFFF", "SHHG"], render_mode="", is_slippery=False)
for episodes in range(EPISODES):
    env.reset()
    traj = []
    state = 12
    print(thetas)
    print(pi)

    for timestamp in range(100):
        action = policy(state)
        new_state, reward, terminated, truncated, info = env.step(action)
        reward = getReward(new_state)
        traj.append([state, action, reward])
        state = new_state
        if terminated or truncated:
            break
    Returns.append(findDiscountedReturn(traj, 0, len(traj)))
    print(traj)
    update(traj)




# DISPLAY RESULT
print(thetas)
print(pi)
#display result
env = gym.make("FrozenLake-v1", desc=["FFFF", "FFFF","FFFF", "SHHG"], render_mode="human", is_slippery=False)
for episodes in range(5):
    env.reset()
    state = 12
    for timestamp in range(100):
        action = deterministicPolicy(state)
        new_state, reward, terminated, truncated, info = env.step(action)
        env.render()
        state = new_state
        # print("Reward: {:.2f}".format(reward))
        if terminated or truncated:
            break

import matplotlib as plt

