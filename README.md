# Reinforcement Learning Algorithms

A collection of Reinforcement Learning algorithms implemented from scratch in Python using NumPy and Matplotlib. This project covers multi-armed bandit problems, gridworld policy evaluation and iteration, and temporal-difference learning methods including Q-Learning and SARSA — all built without RL frameworks.

---

## Project Structure

```
reinforcement-learning/
├── hw1_bandit_distributions.py      # Multi-armed bandit: gamma distribution visualization
├── hw1_bandit_epsilon_ucb.py        # Multi-armed bandit: epsilon-greedy vs UCB comparison
├── hw2_gridworld_setup.py           # 10x10 gridworld maze environment setup
├── hw2_policy_iteration.py          # Policy evaluation and policy iteration on gridworld
├── hw3_cliff_environment.py         # Cliff walking environment setup
└── hw3_qlearning_vs_sarsa.py        # Q-Learning vs SARSA on cliff walking environment
```

---

## Homework 1: Multi-Armed Bandit

### `hw1_bandit_distributions.py`
Visualizes the reward distributions of a 3-arm bandit problem using gamma distributions. Plots the theoretical PDF against sampled data for each arm to understand the reward landscape before applying any strategy.

### `hw1_bandit_epsilon_ucb.py`
Implements and compares two exploration strategies on a 3-arm bandit problem:

- **ε-Greedy** — with ε = 0.01, 0.1, and 0.5
- **UCB (Upper Confidence Bound)** — using the standard UCB1 formula

Tracks optimal action selection rate over 1000 games and plots convergence curves for all four strategies.

---

## Homework 2: Gridworld Policy Iteration

### `hw2_gridworld_setup.py` / `hw2_policy_iteration.py`
Implements policy evaluation and policy iteration on a **10×10 gridworld** with:

- Goal state at top-right corner (0, 9)
- Fence obstacles blocking certain cells
- Reward of -1 per step, 0 at goal
- Discount factor γ = 0.975

**Algorithms implemented:**
- **Policy Evaluation** — iterative evaluation until convergence (||V_k+1 - V_k|| < 1e-6)
- **Policy Improvement** — greedy improvement based on action values
- **Policy Iteration** — full loop until stable optimal policy found

Generates heatmap visualizations of value functions at early, mid, and final iterations with policy arrows overlaid.

---

## Homework 3: Q-Learning vs SARSA

### `hw3_cliff_environment.py` / `hw3_qlearning_vs_sarsa.py`
Implements and compares two temporal-difference learning algorithms on a **5×10 cliff walking environment**:

- Start state: bottom-left (4, 0)
- Goal state: top-right (0, 9)
- Cliff cells: (4,1), (4,2), (4,3) — penalty of -100, resets to start
- Standard reward: -1 per step

**Algorithms implemented:**
- **Q-Learning** — off-policy TD control using max future Q-value
- **SARSA** — on-policy TD control using next action's Q-value

Both run for 500 episodes across 5 independent runs. Results show mean reward with standard deviation bands, demonstrating the classic Q-Learning vs SARSA behavior difference — Q-Learning finds the optimal path near the cliff while SARSA takes the safer route.

---

## Key Concepts Demonstrated

- **Markov Decision Processes (MDP)** — state, action, reward, transition modeling
- **Policy Evaluation** — computing value functions for a fixed policy
- **Policy Iteration** — alternating evaluation and improvement to reach optimality
- **Temporal Difference Learning** — bootstrapping without full episode rollouts
- **Q-Learning** — off-policy control with greedy target policy
- **SARSA** — on-policy control with behavioral policy
- **Exploration vs Exploitation** — epsilon-greedy and UCB bandit strategies
- **Convergence Analysis** — tracking ||V_k+1 - V_k|| across iterations

---

## Requirements

```bash
pip install numpy matplotlib seaborn
```

---

## How to Run

### Multi-Armed Bandit
```bash
python3 hw1_bandit_epsilon_ucb.py
```

### Policy Iteration on Gridworld
```bash
python3 hw2_policy_iteration.py
```

### Q-Learning vs SARSA
```bash
python3 hw3_qlearning_vs_sarsa.py
```

---

## Skills Demonstrated

- **Python & NumPy** — numerical computation, array operations, random sampling
- **Algorithm Implementation** — RL algorithms built from scratch without frameworks
- **Data Visualization** — matplotlib and seaborn heatmaps, convergence plots, reward curves
- **Machine Learning Theory** — MDPs, Bellman equations, temporal difference learning
- **Experimental Design** — multiple independent runs with mean/std deviation analysis

---

## Author

**Triston Barrientos**
CS Senior — Texas Tech University (May 2026)
[LinkedIn](https://linkedin.com/in/triston00barrientos) | [GitHub](https://github.com/TKB100)
