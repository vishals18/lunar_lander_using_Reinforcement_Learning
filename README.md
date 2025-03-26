# Lunar Lander Simulation using Reinforcement Learning

This project implements various Reinforcement Learning (RL) techniques to control a Lunar Lander, focusing on stability and precision in simulated environments. The code explores and compares multiple Deep Q-Learning (DQN) approaches, each optimized for improved performance.

---

## Project Structure

- **code/**: Contains all Python scripts for different RL models.
  - `vanilla_fns.py`, `vanilla_dqn_main.py`: Vanilla DQN implementation.
  - `ddqn_fns.py`, `ddqn_main.py`: Double DQN implementation.
  - `ddqn_per_fns.py`, `ddqn_per_main.py`: Double DQN with Prioritized Experience Replay (PER).
  - `duel_ddqn_per_fns.py`, `duel_ddqn_per_main.py`: Dueling DDQN combined with PER for advanced decision-making.
  - `lunar_lander/`: Contains auxiliary files related to the Lunar Lander environment.
  
- **Report.pdf**: Detailed report explaining the methodology, results, and insights from the project.

---

## Techniques Explored

1. **Vanilla DQN**: Baseline Deep Q-Network implementation.
2. **Double DQN (DDQN)**: Reduces overestimation bias in Q-learning.
3. **DDQN with Prioritized Experience Replay (PER)**: Enhances sampling efficiency by prioritizing important experiences.
4. **Dueling DDQN with PER**: Combines Dueling architecture with PER to optimise learning and decision-making.


