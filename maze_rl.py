"""
Maze Reinforcement Learning + Pygame GUI
------------------------------------------
An agent starts out bumping into walls and wandering randomly. Through
pure trial and error (Q-learning), it gradually learns the shortest
path from Start to Goal. We track how many steps it takes to finish
each training episode, so you can watch the learning curve drop from
"clueless" to "optimal."

Controls in the GUI:
  - SPACE: replay the agent's current best route from Start to Goal.
  - N: generate a brand new random maze and retrain the agent on it.
  - ESC / close window: quit.

Run it with:  python maze_rl.py
(requires pygame:  pip install pygame)
"""

import random
import sys
import pygame

# ============================================================
# 1. Maze generation
# ============================================================
ROWS, COLS = 12, 16


def generate_maze(rows=ROWS, cols=COLS, wall_prob=0.28):
    """
    Generates a random maze, guaranteeing a path exists from start to
    goal (we just keep regenerating until one does -- simple and reliable
    for grids this size).
    """
    while True:
        grid = [[1 if random.random() < wall_prob else 0 for _ in range(cols)] for _ in range(rows)]
        start, goal = (0, 0), (rows - 1, cols - 1)
        grid[start[0]][start[1]] = 0
        grid[goal[0]][goal[1]] = 0
        if _path_exists(grid, start, goal):
            return grid, start, goal


def _path_exists(grid, start, goal):
    """Simple BFS just to confirm the maze is solvable before we train on it."""
    from collections import deque
    rows, cols = len(grid), len(grid[0])
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return False


# ============================================================
# 2. Environment mechanics
# ============================================================
ACTIONS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
ACTION_LIST = list(ACTIONS.keys())


class MazeEnv:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])

    def is_valid(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] == 0

    def step(self, pos, action):
        dr, dc = ACTIONS[action]
        new_pos = (pos[0] + dr, pos[1] + dc)
        if not self.is_valid(new_pos):
            return pos, -5, False  # bumped a wall/edge -- penalty, stay put
        if new_pos == self.goal:
            return new_pos, 100, True
        return new_pos, -1, False  # small step penalty encourages the shortest route


# ============================================================
# 3. Q-learning agent + training with progress tracking
# ============================================================
class QAgent:
    def __init__(self, alpha=0.15, gamma=0.9):
        self.Q = {}
        self.alpha = alpha
        self.gamma = gamma

    def get_q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def best_action(self, state):
        values = [self.get_q(state, a) for a in ACTION_LIST]
        max_v = max(values)
        best = [a for a, v in zip(ACTION_LIST, values) if v == max_v]
        return random.choice(best)

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(ACTION_LIST)
        return self.best_action(state)

    def update(self, state, action, reward, next_state):
        old_q = self.get_q(state, action)
        future_best = max(self.get_q(next_state, a) for a in ACTION_LIST)
        self.Q[(state, action)] = old_q + self.alpha * (reward + self.gamma * future_best - old_q)


def train(env, agent, episodes=800, max_steps=300,
          eps_start=0.9, eps_end=0.05, progress_cb=None):
    """
    Standard one-step Q-learning, run for many episodes. We record steps
    taken (and whether the goal was reached) each episode so we can chart
    the agent's improvement over training.
    """
    history = []  # list of (episode, steps_taken, reached_goal)
    for ep in range(1, episodes + 1):
        epsilon = eps_start + (eps_end - eps_start) * (ep / episodes)
        pos = env.start
        steps = 0
        reached = False
        for _ in range(max_steps):
            action = agent.choose_action(pos, epsilon)
            new_pos, reward, done = env.step(pos, action)
            agent.update(pos, action, reward, new_pos)
            pos = new_pos
            steps += 1
            if done:
                reached = True
                break
        history.append((ep, steps, reached))
        if progress_cb and (ep % 50 == 0 or ep == episodes):
            progress_cb(ep, episodes, steps, reached)
    return history


def get_greedy_path(env, agent, max_steps=300):
    pos = env.start
    path = [pos]
    for _ in range(max_steps):
        action = agent.best_action(pos)
        pos, _, done = env.step(pos, action)
        path.append(pos)
        if done:
            break
    return path


# ============================================================
# 4. Pygame GUI
# ============================================================
CELL = 40
GRID_W, GRID_H = COLS * CELL, ROWS * CELL
MARGIN = 30
CHART_X = MARGIN * 2 + GRID_W
CHART_W = 380
WIDTH = CHART_X + CHART_W + 30
HEIGHT = max(GRID_H + 140, 520)

BG = (24, 26, 32)
WALL_COLOR = (70, 74, 84)
PATH_BG = (40, 43, 52)
START_COLOR = (90, 210, 120)
GOAL_COLOR = (230, 190, 70)
TRAIL_COLOR = (90, 150, 230)
AGENT_COLOR = (240, 100, 100)
TEXT_COLOR = (230, 230, 235)
CHART_BG = (32, 35, 43)
CHART_LINE = (90, 210, 120)
GRID_LINE = (45, 48, 56)


class MazeApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Maze Solver -- Q-Learning")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 22)
        self.small_font = pygame.font.SysFont("arial", 16)

        self.setup_new_maze()

        self.replaying = False
        self.replay_path = []
        self.replay_index = 0
        self.replay_timer = 0

    def setup_new_maze(self):
        grid, start, goal = generate_maze()
        self.env = MazeEnv(grid, start, goal)
        self.agent = QAgent()

        def cb(ep, total, steps, reached):
            pass  # console progress happens in __main__; GUI just needs final history

        self.history = train(self.env, self.agent, episodes=800, progress_cb=cb)
        self.solution_path = get_greedy_path(self.env, self.agent)

    def start_replay(self):
        self.replaying = True
        self.replay_index = 0
        self.replay_timer = pygame.time.get_ticks()

    def draw_maze(self):
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                x, y = MARGIN + c * CELL, MARGIN + r * CELL
                color = WALL_COLOR if self.env.grid[r][c] == 1 else PATH_BG
                pygame.draw.rect(self.screen, color, (x, y, CELL, CELL))
                pygame.draw.rect(self.screen, GRID_LINE, (x, y, CELL, CELL), 1)

        sr, sc = self.env.start
        gr, gc = self.env.goal
        pygame.draw.rect(self.screen, START_COLOR,
                          (MARGIN + sc * CELL + 4, MARGIN + sr * CELL + 4, CELL - 8, CELL - 8), border_radius=6)
        pygame.draw.rect(self.screen, GOAL_COLOR,
                          (MARGIN + gc * CELL + 4, MARGIN + gr * CELL + 4, CELL - 8, CELL - 8), border_radius=6)

        if self.replaying:
            shown = self.solution_path[:self.replay_index + 1]
        else:
            shown = self.solution_path

        for (r, c) in shown[:-1]:
            cx = MARGIN + c * CELL + CELL // 2
            cy = MARGIN + r * CELL + CELL // 2
            pygame.draw.circle(self.screen, TRAIL_COLOR, (cx, cy), 6)

        if shown:
            r, c = shown[-1]
            cx = MARGIN + c * CELL + CELL // 2
            cy = MARGIN + r * CELL + CELL // 2
            pygame.draw.circle(self.screen, AGENT_COLOR, (cx, cy), CELL // 3)

        title = self.font.render("Maze Solver", True, TEXT_COLOR)
        self.screen.blit(title, (MARGIN, 4))

        reached = self.solution_path[-1] == self.env.goal
        status = ("Learned path reaches the goal!" if reached else
                   "Agent hasn't found the goal yet -- try N for a new maze")
        info = self.small_font.render(
            f"{status}   |   steps in learned route: {len(self.solution_path) - 1}",
            True, TEXT_COLOR)
        self.screen.blit(info, (MARGIN, MARGIN + GRID_H + 10))

        controls = self.small_font.render("SPACE = replay agent   |   N = new maze + retrain   |   ESC = quit",
                                           True, TEXT_COLOR)
        self.screen.blit(controls, (MARGIN, MARGIN + GRID_H + 34))

    def draw_chart(self):
        rect = pygame.Rect(CHART_X, MARGIN, CHART_W, GRID_H)
        pygame.draw.rect(self.screen, CHART_BG, rect, border_radius=8)

        title = self.font.render("Training Progress", True, TEXT_COLOR)
        self.screen.blit(title, (CHART_X, 4))
        subtitle = self.small_font.render("Steps taken to reach the goal, per episode", True, TEXT_COLOR)
        self.screen.blit(subtitle, (CHART_X, MARGIN + GRID_H + 10))

        pad = 20
        plot_rect = pygame.Rect(rect.x + pad, rect.y + pad, rect.w - 2 * pad, rect.h - 2 * pad)

        # Cap the y-axis at a sensible max so a few early flailing episodes
        # (which can hit max_steps) don't crush the rest of the chart flat.
        steps = [min(s, 300) for (_, s, _) in self.history]
        max_steps_shown = max(max(steps), 10)
        n = len(steps)

        def to_point(i, val):
            px = plot_rect.x + (i / max(n - 1, 1)) * plot_rect.w
            py = plot_rect.bottom - (val / max_steps_shown) * plot_rect.h
            return (px, py)

        for frac in (0.0, 0.5, 1.0):
            y = plot_rect.bottom - frac * plot_rect.h
            pygame.draw.line(self.screen, (55, 58, 66), (plot_rect.x, y), (plot_rect.right, y), 1)
            label = self.small_font.render(f"{int(frac * max_steps_shown)}", True, TEXT_COLOR)
            self.screen.blit(label, (plot_rect.x - 26, y - 8))

        points = [to_point(i, s) for i, s in enumerate(steps)]
        if len(points) >= 2:
            pygame.draw.lines(self.screen, CHART_LINE, False, points, 2)

        best = min(s for (_, s, reached) in self.history if reached) if any(r for (_, _, r) in self.history) else None
        summary = f"Best route found during training: {best} steps" if best else "Agent never reached the goal during training"
        label = self.small_font.render(summary, True, TEXT_COLOR)
        self.screen.blit(label, (CHART_X, MARGIN + GRID_H + 30))

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_SPACE:
                        self.start_replay()
                    elif event.key == pygame.K_n:
                        self.setup_new_maze()
                        self.replaying = False

            if self.replaying:
                now = pygame.time.get_ticks()
                if now - self.replay_timer > 180:
                    self.replay_timer = now
                    if self.replay_index < len(self.solution_path) - 1:
                        self.replay_index += 1
                    else:
                        self.replaying = False

            self.screen.fill(BG)
            self.draw_maze()
            self.draw_chart()
            pygame.display.flip()
            self.clock.tick(60)


# ============================================================
# 5. Entry point
# ============================================================
if __name__ == "__main__":
    print("Generating a solvable random maze...")
    grid, start, goal = generate_maze()
    env = MazeEnv(grid, start, goal)
    agent = QAgent()

    def progress_cb(ep, total, steps, reached):
        tag = "reached goal" if reached else "did not finish"
        print(f"  episode {ep:>4}/{total}  steps: {steps:>3}  ({tag})")

    print("Training via trial-and-error Q-learning...")
    history = train(env, agent, episodes=800, progress_cb=progress_cb)
    print("Training complete!\n")

    app = MazeApp()
    # reuse the maze/agent/history we already trained instead of retraining again
    app.env, app.agent, app.history = env, agent, history
    app.solution_path = get_greedy_path(env, agent)
    app.run()