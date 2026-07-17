"""
Maze Solver using Q-Learning
-----------------------------
A tiny reinforcement learning project: an agent learns to navigate
a grid maze from Start (S) to Goal (G) using Q-learning.

Run it with:  python maze_q_learning.py
"""

import random

# ---------------------------------------------------------
# 1. Define the maze
# ---------------------------------------------------------
# '#' = wall, '.' = open path, 'S' = start, 'G' = goal
MAZE = [
    "S . . # . . .",
    ". # . # . # .",
    ". # . . . # .",
    ". # # # . # .",
    ". . . # . . .",
    "# # . # # # .",
    ". . . . . . G",
]

# Convert the string maze into a 2D list of characters
grid = [row.split() for row in MAZE]
ROWS, COLS = len(grid), len(grid[0])

start_pos = next((r, c) for r in range(ROWS) for c in range(COLS) if grid[r][c] == "S")
goal_pos = next((r, c) for r in range(ROWS) for c in range(COLS) if grid[r][c] == "G")

# ---------------------------------------------------------
# 2. Actions the agent can take
# ---------------------------------------------------------
ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}
ACTION_LIST = list(ACTIONS.keys())


def is_valid(pos):
    r, c = pos
    if 0 <= r < ROWS and 0 <= c < COLS:
        return grid[r][c] != "#"
    return False


def step(pos, action):
    """Move the agent, returning (new_pos, reward, done)."""
    dr, dc = ACTIONS[action]
    new_pos = (pos[0] + dr, pos[1] + dc)

    if not is_valid(new_pos):
        return pos, -5, False  # bumped into a wall or edge, stay put

    if new_pos == goal_pos:
        return new_pos, 100, True  # reached the goal!

    return new_pos, -1, False  # small penalty per step, encourages shortest path


# ---------------------------------------------------------
# 3. Q-learning setup
# ---------------------------------------------------------
Q = {}  # Q[(state, action)] = value

ALPHA = 0.1       # learning rate
GAMMA = 0.9       # discount factor
EPSILON = 0.2     # exploration rate
EPISODES = 500
MAX_STEPS = 200


def get_q(state, action):
    return Q.get((state, action), 0.0)


def best_action(state):
    values = [get_q(state, a) for a in ACTION_LIST]
    max_v = max(values)
    # break ties randomly
    best = [a for a, v in zip(ACTION_LIST, values) if v == max_v]
    return random.choice(best)


def choose_action(state):
    if random.random() < EPSILON:
        return random.choice(ACTION_LIST)
    return best_action(state)


# ---------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------
def train():
    for episode in range(EPISODES):
        pos = start_pos
        for _ in range(MAX_STEPS):
            action = choose_action(pos)
            new_pos, reward, done = step(pos, action)

            old_q = get_q(pos, action)
            future_best = max(get_q(new_pos, a) for a in ACTION_LIST)
            Q[(pos, action)] = old_q + ALPHA * (reward + GAMMA * future_best - old_q)

            pos = new_pos
            if done:
                break


# ---------------------------------------------------------
# 5. Show the learned path
# ---------------------------------------------------------
def show_path():
    pos = start_pos
    path = [pos]
    for _ in range(MAX_STEPS):
        action = best_action(pos)
        pos, _, done = step(pos, action)
        path.append(pos)
        if done:
            break

    # Draw the maze with the path marked as '*'
    display = [row[:] for row in grid]
    for r, c in path:
        if display[r][c] == ".":
            display[r][c] = "*"

    print("\nLearned path from S to G:\n")
    for row in display:
        print(" ".join(row))
    print(f"\nSteps taken: {len(path) - 1}")


if __name__ == "__main__":
    print("Training agent with Q-learning...")
    train()
    print("Training complete!")
    show_path()