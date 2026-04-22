import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Environment Constants
GRID_SIZE = 10
GOAL_STATE = (0, 9) # Top-right corner
FENCES = [(1, 1), (8, 8), (5, 4)] 
GAMMA = 0.975

# Actions: 0: Up, 1: Right, 2: Down, 3: Left
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ACTION_SYMBOLS = ['↑', '→', '↓', '←']

def step(state, action_idx):
    row, col = state
    row_change, col_change = ACTIONS[action_idx]
    new_row = row + row_change
    new_col = col + col_change
    if (new_row < 0 or new_row >= GRID_SIZE or new_col < 0 or new_col >= GRID_SIZE or (new_row, new_col) in FENCES):
        next_state = state
        reward = -1
    else:
        next_state = (new_row, new_col)
        reward = -1
        if next_state == GOAL_STATE:
            reward = 0

    return next_state, reward


def plot_heatmap(data, title, filename, policy=None, annot=True, fmt=".1f"):
    plt.figure(figsize=(8, 8))
    mask = np.zeros_like(data, dtype=bool)
    for f in FENCES:
        mask[f] = True

    ax = sns.heatmap(data, annot=annot, fmt=fmt, cmap="viridis", mask=mask, cbar=True, linewidths=0.5, linecolor='black', square=True)

    for f in FENCES:
        ax.text(f[1]+0.5, f[0]+0.5, 'X', color='red', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(GOAL_STATE[1]+0.5, GOAL_STATE[0]+0.5, 'G', color='white', ha='center', va='center', fontsize=14, weight='bold')

    if policy is not None:
        for s in range(100):
            row = s // GRID_SIZE
            col = s % GRID_SIZE
            if (row, col) in FENCES or (row, col) == GOAL_STATE:
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

# Visualize Task 1
env_grid = np.zeros((GRID_SIZE, GRID_SIZE))
plot_heatmap(env_grid, "Task 1: Initial Maze Layout", "task1_maze.png", annot=False)



#Task 2: Evaluate a Random Policy

def evaluate_policy(policy):
    V = np.zeros(100)
    norm_history = []         
    while True:
        V_old = V.copy()
        for s in range(100):
            action = policy[s]
            row = s // GRID_SIZE
            col = s % GRID_SIZE
            state = (row, col)
            next_state, reward = step(state, action)
            next_index = next_state[0] * GRID_SIZE + next_state[1]
            V[s] = reward + GAMMA * V[next_index]
        norm_history.append(np.linalg.norm(V - V_old))  
        if np.linalg.norm(V - V_old) < 1e-6:
            break
    return V, norm_history     

np.random.seed(41)
random_policy = np.random.randint(0, 4, 100)

#Visualize Task 2

U_random, norm_history_eval = evaluate_policy(random_policy)
plot_heatmap(U_random.reshape(GRID_SIZE, GRID_SIZE), "Task 2: Value Function of a Random Policy", "task2_value.png")

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(norm_history_eval)+1), norm_history_eval, color='steelblue')
plt.xlabel("Policy Evaluation Iteration k")
plt.ylabel("||V_k+1 - V_k||")
plt.title("Task 2: Convergence of Policy Evaluation")
plt.savefig("task2_norm.png")
plt.show()


#Task 3: Modify the Random Policy to Move Towards the Goal

modified_policy = np.copy(random_policy)

def set_optimal_actions(state, radius=2):
    distance = abs(state[0] - GOAL_STATE[0]) + abs(state[1] - GOAL_STATE[1])
    row, col = state
    if distance <= radius:
        if col < GOAL_STATE[1]:
            return 1
        elif row > GOAL_STATE[0]:
            return 0

for s in range(100):
    row = s // GRID_SIZE
    col = s % GRID_SIZE
    action = set_optimal_actions((row, col))
    if action is not None:
        modified_policy[s] = action

#Visualize Task 3

U_modified, _ = evaluate_policy(modified_policy)
plot_heatmap(U_modified.reshape(GRID_SIZE, GRID_SIZE), "Task 3: Value Function of the Modified Random Policy", "task3_value.png")


#Task 4: Optimal Policy and Value Function

def improve_policy(V):
    new_policy = np.zeros(100, dtype=int)
    for s in range(100):
        row = s // GRID_SIZE
        col = s % GRID_SIZE
        state = (row, col)
        action_values = np.zeros(4)
        for a in range(4):
            next_state, reward = step(state, a)
            next_index = next_state[0] * GRID_SIZE + next_state[1]
            action_values[a] = reward + GAMMA * V[next_index]
        new_policy[s] = np.argmax(action_values)
    return new_policy

def policy_iteration():
    policy = np.copy(random_policy)
    U_history = []
    norm_history = []       
    V_prev = np.zeros(100)   
    while True:
        old_policy = policy.copy()
        V, _ = evaluate_policy(policy)
        U_history.append(V.reshape(GRID_SIZE, GRID_SIZE))
        norm_history.append(np.linalg.norm(V - V_prev)) 
        V_prev = V.copy()  
        policy = improve_policy(V)
        if np.array_equal(old_policy, policy):
            break
    return policy, U_history, norm_history


optimal_policy, U_history, norm_history = policy_iteration()

# Plot 3 different iterations (Early, Mid, Final)
indices_to_plot = [0, len(U_history)//2, len(U_history)-1]
labels = ["Early Iteration", "Middle Iteration", "Final Iteration (Converged)"]

for idx, label in zip(indices_to_plot, labels):
    plot_heatmap(U_history[idx], f"Task 4: {label}", f"task4_{label}.png", policy=optimal_policy)

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(norm_history)+1), norm_history, marker='o', color='steelblue')
plt.xlabel("Policy Iteration Index")
plt.ylabel("||V_new - V_old||")
plt.title("Task 4: Convergence of Policy Iteration")
plt.savefig("task4_norm.png")
plt.show()