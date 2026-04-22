import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Environment Constants
GRID_SIZE = 10
GOAL_STATE = (0, 9) # Top-right corner
FENCES = [(1, 1), (8, 8), (5, 4)] 

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


def plot_heatmap(data, title, annot=True, fmt=".1f"):
    """Helper to plot utilities with a visible grid structure."""
    plt.figure(figsize=(8, 8)) # Increased size for clarity
    mask = np.zeros_like(data, dtype=bool)
    for f in FENCES:
        mask[f] = True

    # linewidths adds the border, linecolor sets the color of the grid lines
    ax = sns.heatmap(data, annot=annot, fmt=fmt, cmap="viridis", mask=mask, cbar=True, linewidths=0.5, linecolor='black',square=True)

    # Mark Fences and Goal
    for f in FENCES:
        ax.text(f[1]+0.5, f[0]+0.5, 'X', color='red', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(GOAL_STATE[1]+0.5, GOAL_STATE[0]+0.5, 'G', color='white', ha='center', va='center', fontsize=14, weight='bold')

    plt.title(title)
    plt.savefig("task1_maze.png")
    plt.show()

# Visualize Task 1
env_grid = np.zeros((GRID_SIZE, GRID_SIZE))
plot_heatmap(env_grid, "Task 1: Initial Maze Layout (X = Fence, G = Goal)", annot=False)