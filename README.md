# Reinforcement Learning Algorithms

Welcome to my repository of Reinforcement Learning (RL) implementations! This repository contains my implementations of various RL algorithms from scratch. A modified verision of FrozenLake has been used as environment to allow easy debugging and fast protoytyping. I've strived to make the code as readable and understandable as possible to facilitate learning and collaboration. Currently, the repository is a work in progress, and I will be adding more algorithms and environments in the future.

## Algorithms Implemented

1. **Dynamic Programming (DP):**
   - [Link to DP Implementation](dynamic_programming/)
   <!-- - Description: DP applied using a determinstic policy to FrozenLake. -->

2. **Monte Carlo (MC):**
   - [Link to MC Implementation](monte_carlo/)
   <!-- - Description: Here, you'll find my implementation of Monte Carlo for solving FrozenLake. -->

3. **Temporal Difference (TD):**
   - [Link to TD Implementation](temporal_difference/)
   <!-- - TD(0) has been implemented for FrozenLake. -->

4. **Policy Gradient (PG):**
   - [Link to PG Implementation](policy_gradients/)
   <!-- - Currently only vanilla policy gradient (REINFORCE) has implemented. Working on making it more modular and adding more algorithms. -->

## Repository Structure

- Each algorithm has its dedicated folder (`dynamic_programming/`, `monte_carlo/`, `temporal_difference/`, `policy_gradients/`) containing code and related documentation.
- Each folder contains `src/` for the source code, `/examples` for testing the code and `notes/` for documentation. `/examples` is under construction. Temporarily the implementations in `src/` are executables and can be used to train and test the algorithms.
- `_environments/` contains a modified verion of farama-foundations's FrozenLake environment that allows for modifying the reward structure.
- The code is designed to be readable and well-documented to aid in understanding and learning.

<!-- ## Getting Started

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git -->
