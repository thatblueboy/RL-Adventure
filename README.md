# Reinforcement Learning Algorithms

Welcome to my repository of Reinforcement Learning (RL) implementations! This repository contains my implementations of various RL algorithms from scratch. A modified verision of FrozenLake has been used as environment to allow easy debugging and fast protoytyping. I've strived to make the code as readable and understandable as possible to facilitate learning and collaboration. Currently, the repository is a work in progress, and I will be adding more algorithms and environments in the future.

<img src="https://github.com/thatblueboy/RL-Adventure/assets/100462736/63b0341b-cb8a-41d0-ad53-3a84d91671f1" width="45%"></img> 

## Algorithms Implemented

1. [**Dynamic Programming (DP):**](dynamic_programming/)
   <!-- - Description: DP applied using a determinstic policy to FrozenLake. -->

2. [**Monte Carlo (MC):**](monte_carlo/)
   <!-- - Description: MC applied using a determinstic policy to FrozenLake. -->

3. [**Temporal Difference (TD):**](temporal_difference/)
   <!-- - Description: TD applied using a determinstic policy to FrozenLake. -->

4. [**Policy Gradients (PG):**](policy_gradients/)
   <!-- - Description: PG applied using a determinstic policy to FrozenLake. -->

## Repository Structure

- Each algorithm has its dedicated folder (`dynamic_programming/`, `monte_carlo/`, `temporal_difference/`, `policy_gradients/`) containing code and related documentation.
- Each folder contains `src/` for the source code, `/notebooks` for testing the code and for documentation. `/notebooks` is under construction. Temporarily the files in `src/` can be ran directly to train and test the algorithms.
- `_environments/` contains a modified verion of farama-foundations's FrozenLake environment that allows for modifying the reward structure.
- The code is designed to be readable and well-documented to aid in understanding and learning.

<!-- ## Getting Started

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git -->
