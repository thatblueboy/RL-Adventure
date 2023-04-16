import gymnasium as gym
import numpy as np
import random



# 0: LEFT
# 1: DOWN
# 2: RIGHT
# 3: UP

env = gym.make("FrozenLake-v1", desc=["FFFF", "FFFF","FFFF", "SHHG"], render_mode="", is_slippery=False)
thetas = np.zeros((16, 4), dtype=float)
pi = np.zeros((16, 4), dtype=float)
NUMBER_OF_STATES = 16
NUMBER_OF_ACTIONS = 4
ALPHA = 0.1
GAMMA = 0.1
EPISODES = 100

def getReward(observation):
    if 0 <= observation <= 12:
        return -1
    if observation == 13 or observation == 14:
        return -100
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
        # print(state)
        # print(G)
        score = actionToVector(action)-pi[state]
        thetas[state] = thetas[state] + ALPHA*score*G
        updatePI(state)

def policy(state):
    return random.choice(random.choices([[0], [1], [2], [3]], weights = list(pi[state]), k = 100))

for episodes in range(EPISODES):
    env.reset()
    traj = []
    state = 13
    print(thetas)
    print(pi)

    for timestamp in range(100):
        action = policy(state)
        action = env.action_space.sample()
        new_state, reward, terminated, truncated, info = env.step(action)
        reward = getReward(new_state)
        # env.render()
        traj.append([state, action, reward])
        state = new_state
        # print("Reward: {:.2f}".format(reward))
        if terminated or truncated:
            break

    update(traj)

print(thetas)
print(pi)