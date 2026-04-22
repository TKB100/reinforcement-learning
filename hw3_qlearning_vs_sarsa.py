import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Environment Constants
ROWS = 5
COLS = 10
GOAL_STATE = (0, 9)
START_STATE = (4, 0)
CLIFF = [(4, 1), (4, 2), (4, 3)]
GAMMA = 0.975

# Actions: 0: Up, 1: Right, 2: Down, 3: Left
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ACTION_SYMBOLS = ['↑', '→', '↓', '←']

# Updated Functions from RL_HW2.py to fit the new environment and tasks

def step(state, action_idx):
    row, col = state
    row_change, col_change = ACTIONS[action_idx]
    new_row = row + row_change
    new_col = col + col_change
    if (new_row < 0 or new_row >= ROWS or new_col < 0 or new_col >= COLS):
        return state, -1
    next_state = (new_row, new_col)
    if next_state in CLIFF:
        return START_STATE, -100
    if next_state == GOAL_STATE:
        return next_state, 0
    return next_state, -1

def generate_trajectory(start_state, policy=None, max_steps=200):
    trajectory = []
    state = start_state
    for _ in range(max_steps):
        if policy is None:
            action_idx = np.random.randint(0, 4)
        else:
            action_idx = policy[state[0] * COLS + state[1]]
        next_state, reward = step(state, action_idx)
        trajectory.append((state, action_idx, reward))
        state = next_state
        if state == GOAL_STATE:
            break
    return trajectory

random_policy = np.random.randint(0, 4, ROWS * COLS)

def evaluate_policy(policy):
    V = np.zeros(ROWS * COLS)
    norm_history = []         
    while True:
        V_old = V.copy()
        for s in range(ROWS * COLS):
            action = policy[s]
            row = s // COLS
            col = s % COLS
            state = (row, col)
            next_state, reward = step(state, action)
            next_index = next_state[0] * COLS + next_state[1]
            V[s] = reward + GAMMA * V[next_index]
        norm_history.append(np.linalg.norm(V - V_old))  
        if np.linalg.norm(V - V_old) < 1e-6:
            break
    return V, norm_history 

def improve_policy(V):
    new_policy = np.zeros(ROWS * COLS, dtype=int)
    for s in range(ROWS * COLS):
        row = s // COLS
        col = s % COLS
        state = (row, col)
        action_values = np.zeros(4)
        for a in range(4):
            next_state, reward = step(state, a)
            next_index = next_state[0] * COLS + next_state[1]
            action_values[a] = reward + GAMMA * V[next_index]
        new_policy[s] = np.argmax(action_values)
    return new_policy

def policy_iteration():
    policy = np.copy(random_policy)
    U_history = []
    norm_history = []       
    V_prev = np.zeros(ROWS * COLS)   
    while True:
        old_policy = policy.copy()
        V, _ = evaluate_policy(policy)
        U_history.append(V.reshape(ROWS, COLS))
        norm_history.append(np.linalg.norm(V - V_prev)) 
        V_prev = V.copy()  
        policy = improve_policy(V)
        if np.array_equal(old_policy, policy):
            break
    return policy, U_history, norm_history


def plot_heatmap(data, title, filename, policy=None, annot=True, fmt=".1f"):
    plt.figure(figsize=(10, 5))
    mask = np.zeros_like(data, dtype=bool)

    ax = sns.heatmap(data, annot=annot, fmt=fmt, cmap="viridis", mask=mask, 
                     cbar=True, linewidths=0.5, linecolor='black', square=True)

    # Mark cliff
    for c in CLIFF:
        ax.text(c[1]+0.5, c[0]+0.5, 'C', color='red', ha='center', va='center', fontsize=14, weight='bold')
    
    # Mark goal and start
    ax.text(GOAL_STATE[1]+0.5, GOAL_STATE[0]+0.5, 'G', color='white', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(START_STATE[1]+0.5, START_STATE[0]+0.5, 'S', color='white', ha='center', va='center', fontsize=14, weight='bold')

    if policy is not None:
        for s in range(ROWS * COLS):
            row = s // COLS
            col = s % COLS
            if (row, col) in CLIFF or (row, col) == GOAL_STATE:
                continue
            action = policy[s]
            dx, dy = ACTIONS[action]
            ax.annotate('',
                xy=(col + 0.5 + dy*0.4, row + 0.5 + dx*0.4),
                xytext=(col + 0.5 - dy*0.4, row + 0.5 - dx*0.4),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    plt.title(title)
    plt.savefig(filename)
    plt.show()

# Task 1: Random vs Optimal Policy Trajectories

np.random.seed(42)
random_policy = np.random.randint(0, 4, ROWS * COLS)
optimal_policy, U_history, norm_history = policy_iteration()

# 10 random starting states
start_states = [(np.random.randint(0, ROWS), np.random.randint(0, COLS)) 
                for _ in range(10)]

# Random policy trajectories
print("Random Policy Trajectories:")
for i, start in enumerate(start_states):
    traj = generate_trajectory(start, policy=None)
    total_reward = sum(r for _, _, r in traj)
    print(f"  Start {start}: {len(traj)} steps, total reward = {total_reward}")

# Optimal policy trajectories
print("\nOptimal Policy Trajectories:")
for i, start in enumerate(start_states):
    traj = generate_trajectory(start, policy=optimal_policy)
    total_reward = sum(r for _, _, r in traj)
    print(f"  Start {start}: {len(traj)} steps, total reward = {total_reward}")


# Task 2: Q-Learning 

def q_learning(num_episodes=500, alpha=0.1, epsilon=0.1):
    Q = np.zeros((ROWS * COLS, 4))
    rewards_per_episode = []

    for episode in range(num_episodes):
        state = START_STATE
        total_reward = 0
        while state != GOAL_STATE:
            state_index = state[0] * COLS + state[1]
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(0, 4)
            else:
                action_idx = np.argmax(Q[state_index])
            next_state, reward = step(state, action_idx)
            next_index = next_state[0] * COLS + next_state[1]
            Q[state_index, action_idx] += alpha * (reward + GAMMA * np.max(Q[next_index]) - Q[state_index, action_idx])
            state = next_state
            total_reward += reward
        rewards_per_episode.append(total_reward)
    return Q, rewards_per_episode

all_rewards = []
for run in range(5):
    Q, rewards = q_learning()
    all_rewards.append(rewards)

all_rewards = np.array(all_rewards)
mean_rewards = np.mean(all_rewards, axis=0)
std_rewards = np.std(all_rewards, axis=0)

plt.figure(figsize=(10, 5))
for i in range(5):
    plt.plot(all_rewards[i], alpha=0.3, color='steelblue')
plt.plot(mean_rewards, color='blue', label='Mean Reward')
plt.fill_between(range(500), mean_rewards - std_rewards, mean_rewards + std_rewards, color='lightblue', alpha=0.2, label='Std Dev')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Task 2: Q-Learning")
plt.savefig("task2_qlearning.png")
plt.show()

# Task 3: SARSA

def sarsa(num_episodes=500, alpha=0.1, epsilon=0.1):
    Q = np.zeros((ROWS * COLS, 4))
    rewards_per_episode = []

    for episode in range(num_episodes):
        state = START_STATE
        state_index = state[0] * COLS + state[1]
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(0, 4)
        else:
            action_idx = np.argmax(Q[state_index])
        total_reward = 0

        while state != GOAL_STATE:
            next_state, reward = step(state, action_idx)
            next_index = next_state[0] * COLS + next_state[1]
            if np.random.rand() < epsilon:
                next_action_idx = np.random.randint(0, 4)
            else:
                next_action_idx = np.argmax(Q[next_index])
            Q[state_index, action_idx] += alpha * (reward + GAMMA * Q[next_index, next_action_idx] - Q[state_index, action_idx])
            state = next_state
            state_index = state[0] * COLS + state[1]
            action_idx = next_action_idx
            total_reward += reward
        rewards_per_episode.append(total_reward)
    return Q, rewards_per_episode

all_rewards_sarsa = []
for run in range(5):
    Q_sarsa, rewards = sarsa()
    all_rewards_sarsa.append(rewards)

all_rewards_sarsa = np.array(all_rewards_sarsa)
mean_rewards_sarsa = np.mean(all_rewards_sarsa, axis=0)
std_rewards_sarsa = np.std(all_rewards_sarsa, axis=0)

plt.figure(figsize=(10, 5))
for i in range(5):
    plt.plot(all_rewards_sarsa[i], alpha=0.3, color='Red')
plt.plot(mean_rewards_sarsa, color='salmon', label='Mean Reward')
plt.fill_between(range(500), mean_rewards_sarsa - std_rewards_sarsa, mean_rewards_sarsa + std_rewards_sarsa, color='lightcoral', alpha=0.2, label='Std Dev')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Task 3: SARSA")
plt.savefig("task3_sarsa.png")
plt.show()

# Task 4: Compare Q-Learning and SARSA

plt.figure(figsize=(10, 5))

# Q-learning
plt.plot(mean_rewards, color='blue', linewidth=2, label='Q-Learning Mean')
plt.fill_between(range(500), mean_rewards - std_rewards, mean_rewards + std_rewards, color='cornflowerblue', alpha=0.2, label='Q-Learning Std')

# SARSA
plt.plot(mean_rewards_sarsa, color='red', linewidth=2, label='SARSA Mean')
plt.fill_between(range(500), mean_rewards_sarsa - std_rewards_sarsa, mean_rewards_sarsa + std_rewards_sarsa, color='salmon', alpha=0.2, label='SARSA Std')

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Task 4: Q-Learning vs SARSA")
plt.legend()
plt.savefig("task4_comparison.png")
plt.show()

# Task 5: Value Funtion & Policy 

V_q = np.max(Q, axis=1)
pi_q = np.argmax(Q, axis=1)
V_sarsa = np.max(Q_sarsa, axis=1)
pi_sarsa = np.argmax(Q_sarsa, axis=1)

plot_heatmap(V_q.reshape(ROWS, COLS), "Task 5: Q-Learning Value Function", "task5_qlearning.png", policy=pi_q)
plot_heatmap(V_sarsa.reshape(ROWS, COLS), "Task 5: SARSA Value Function", "task5_sarsa.png", policy=pi_sarsa)