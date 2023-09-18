from REINFORCE.environments.frozen_lake_custom import FrozenLakeCustomEnv

env = FrozenLakeCustomEnv(desc=None,map_name='4x4', is_slippery=False, render_mode='human')

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample() 
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)

    if terminated or truncated:
        observation, info = env.reset()

env.close()