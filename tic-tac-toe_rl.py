"""
Tic-Tac-Toe Reinforcement Learning (Self-Play) + Pygame GUI
-------------------------------------------------------------
The agent learns entirely by playing itself thousands of times.
No hardcoded strategy, no hand-coded rules -- just trial, error,
and reward. We track its win-rate against a random player over the
course of training so you can literally watch it get good.

Controls in the GUI:
  - Click a cell to place your mark.
  - Buttons at the bottom let you start a new game as X or O.
  - The right-hand panel shows the learning curve from training.

Run it with:  python tic_tac_toe_rl.py
(requires pygame:  pip install pygame)
"""

import random
import sys
import pygame

# ============================================================
# 1. Game rules (shared by training and the GUI)
# ============================================================
# Board cells are stored as 0 (empty), 1 (X), or -1 (O).
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


def check_winner(board):
    """Returns 1 if X won, -1 if O won, 0 for a draw, None if still playing."""
    for a, b, c in WIN_LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    if 0 not in board:
        return 0
    return None


def available_moves(board):
    return [i for i, v in enumerate(board) if v == 0]


def canonical(board, player):
    """
    Perspective transform: the agent always sees ITSELF as +1 and the
    opponent as -1, no matter whether it's actually playing X or O.
    This lets one Q-table learn from playing BOTH sides at once.
    """
    return tuple(v * player for v in board)


# ============================================================
# 2. The Q-learning agent (tabular, learned via self-play)
# ============================================================
class QAgent:
    def __init__(self, alpha=0.3, gamma=0.95):
        self.Q = {}
        self.alpha = alpha
        self.gamma = gamma

    def get_q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def best_action(self, state, moves):
        values = [self.get_q(state, m) for m in moves]
        max_v = max(values)
        best = [m for m, v in zip(moves, values) if v == max_v]
        return random.choice(best)

    def choose_action(self, state, moves, epsilon):
        if random.random() < epsilon:
            return random.choice(moves)
        return self.best_action(state, moves)


# ============================================================
# 3. Self-play training with progress tracking
# ============================================================
def play_self_play_episode(agent, epsilon):
    """Agent plays BOTH sides against itself for one full game."""
    board = [0] * 9
    current = random.choice([1, -1])  # randomize who starts, for variety
    moves_record = {1: [], -1: []}

    while True:
        state = canonical(board, current)
        moves = available_moves(board)
        action = agent.choose_action(state, moves, epsilon)
        board[action] = current
        moves_record[current].append((state, action))

        result = check_winner(board)
        if result is not None:
            return moves_record, result
        current = -current


def update_from_episode(agent, moves_record, winner):
    """
    Monte-Carlo style update: at the end of the game we know exactly how
    good/bad the outcome was, so we push every move each player made
    towards that outcome, with moves closer to the end getting more
    credit (discounted by gamma the further back they are).
    """
    for player in (1, -1):
        recs = moves_record[player]
        n = len(recs)
        if winner == 0:
            terminal_reward = 0.0
        elif winner == player:
            terminal_reward = 1.0
        else:
            terminal_reward = -1.0

        for i, (state, action) in enumerate(recs):
            steps_from_end = (n - 1 - i)
            target = terminal_reward * (agent.gamma ** steps_from_end)
            old_q = agent.get_q(state, action)
            agent.Q[(state, action)] = old_q + agent.alpha * (target - old_q)


def evaluate_against_random(agent, games=200):
    """Play the current (greedy) policy against a random opponent."""
    wins = draws = losses = 0
    for _ in range(games):
        board = [0] * 9
        current = random.choice([1, -1])
        agent_side = 1  # doesn't matter which side we call "agent"; policy is symmetric
        while True:
            moves = available_moves(board)
            if current == agent_side:
                state = canonical(board, current)
                action = agent.best_action(state, moves)
            else:
                action = random.choice(moves)
            board[action] = current
            result = check_winner(board)
            if result is not None:
                if result == 0:
                    draws += 1
                elif result == agent_side:
                    wins += 1
                else:
                    losses += 1
                break
            current = -current
    total = wins + draws + losses
    return wins / total, draws / total, losses / total


def train(agent, episodes=40000, eval_every=1000, eval_games=200,
          eps_start=0.4, eps_end=0.02, progress_cb=None):
    """
    Trains via self-play. Every `eval_every` episodes, freezes exploration
    and checks how the agent does against a random player, recording that
    to `history` so we can plot the learning curve afterwards.
    """
    history = []  # list of (episode, win_rate, draw_rate, loss_rate)
    for ep in range(1, episodes + 1):
        epsilon = eps_start + (eps_end - eps_start) * (ep / episodes)
        moves_record, winner = play_self_play_episode(agent, epsilon)
        update_from_episode(agent, moves_record, winner)

        if ep % eval_every == 0 or ep == episodes:
            w, d, l = evaluate_against_random(agent, eval_games)
            history.append((ep, w, d, l))
            if progress_cb:
                progress_cb(ep, episodes, w, d, l)

    return history


# ============================================================
# 4. Pygame GUI
# ============================================================
WIDTH, HEIGHT = 900, 560
BOARD_MARGIN = 40
BOARD_SIZE = 420
CELL = BOARD_SIZE // 3
CHART_X = BOARD_MARGIN * 2 + BOARD_SIZE
CHART_W = WIDTH - CHART_X - 40

BG = (24, 26, 32)
GRID_COLOR = (90, 96, 110)
X_COLOR = (240, 100, 100)
O_COLOR = (100, 170, 240)
TEXT_COLOR = (230, 230, 235)
BTN_COLOR = (60, 130, 90)
BTN_HOVER = (80, 160, 115)
CHART_BG = (32, 35, 43)
CHART_LINE_WIN = (90, 210, 120)
CHART_LINE_DRAW = (220, 200, 90)
CHART_LINE_LOSS = (220, 100, 100)


class Button:
    def __init__(self, rect, text):
        self.rect = pygame.Rect(rect)
        self.text = text

    def draw(self, surface, font, mouse_pos):
        color = BTN_HOVER if self.rect.collidepoint(mouse_pos) else BTN_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        label = font.render(self.text, True, TEXT_COLOR)
        surface.blit(label, label.get_rect(center=self.rect.center))

    def clicked(self, pos):
        return self.rect.collidepoint(pos)


class TicTacToeApp:
    def __init__(self, agent, history):
        pygame.init()
        pygame.display.set_caption("Tic-Tac-Toe vs Self-Trained RL Agent")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 22)
        self.small_font = pygame.font.SysFont("arial", 16)
        self.big_font = pygame.font.SysFont("arial", 40, bold=True)

        self.agent = agent
        self.history = history  # [(episode, win, draw, loss), ...]

        self.stats = {"agent_wins": 0, "human_wins": 0, "draws": 0}
        self.new_game(human_first=False)

        btn_y = BOARD_MARGIN + BOARD_SIZE + 30
        self.btn_you_first = Button((BOARD_MARGIN, btn_y, 200, 44), "New Game: You First")
        self.btn_ai_first = Button((BOARD_MARGIN + 220, btn_y, 200, 44), "New Game: AI First")

    def new_game(self, human_first):
        self.board = [0] * 9
        self.human_mark = -1 if human_first else 1
        # if AI goes first, flip so AI is X(1) and moves immediately
        self.human_mark = 1 if human_first else -1
        self.current = 1  # X always moves first
        self.game_over = False
        self.result_text = ""
        self.ai_pending = self.current != self.human_mark

    def agent_move(self):
        state = canonical(self.board, self.current)
        moves = available_moves(self.board)
        if moves:
            action = self.agent.best_action(state, moves)
            self.board[action] = self.current
            self.after_move()

    def after_move(self):
        result = check_winner(self.board)
        if result is not None:
            self.game_over = True
            if result == 0:
                self.result_text = "Draw!"
                self.stats["draws"] += 1
            elif result == self.human_mark:
                self.result_text = "You win!"
                self.stats["human_wins"] += 1
            else:
                self.result_text = "Agent wins!"
                self.stats["agent_wins"] += 1
        else:
            self.current *= -1
            self.ai_pending = self.current != self.human_mark

    def handle_click(self, pos):
        if self.btn_you_first.clicked(pos):
            self.new_game(human_first=True)
            return
        if self.btn_ai_first.clicked(pos):
            self.new_game(human_first=False)
            return

        if self.game_over or self.current != self.human_mark:
            return
        x, y = pos
        if BOARD_MARGIN <= x < BOARD_MARGIN + BOARD_SIZE and BOARD_MARGIN <= y < BOARD_MARGIN + BOARD_SIZE:
            col = (x - BOARD_MARGIN) // CELL
            row = (y - BOARD_MARGIN) // CELL
            idx = row * 3 + col
            if self.board[idx] == 0:
                self.board[idx] = self.human_mark
                self.after_move()

    def draw_board(self):
        for i in range(1, 3):
            pygame.draw.line(self.screen, GRID_COLOR,
                              (BOARD_MARGIN + i * CELL, BOARD_MARGIN),
                              (BOARD_MARGIN + i * CELL, BOARD_MARGIN + BOARD_SIZE), 3)
            pygame.draw.line(self.screen, GRID_COLOR,
                              (BOARD_MARGIN, BOARD_MARGIN + i * CELL),
                              (BOARD_MARGIN + BOARD_SIZE, BOARD_MARGIN + i * CELL), 3)

        for i, v in enumerate(self.board):
            row, col = divmod(i, 3)
            cx = BOARD_MARGIN + col * CELL + CELL // 2
            cy = BOARD_MARGIN + row * CELL + CELL // 2
            if v == 1:
                label = self.big_font.render("X", True, X_COLOR)
                self.screen.blit(label, label.get_rect(center=(cx, cy)))
            elif v == -1:
                label = self.big_font.render("O", True, O_COLOR)
                self.screen.blit(label, label.get_rect(center=(cx, cy)))

        title = self.font.render("Tic-Tac-Toe", True, TEXT_COLOR)
        self.screen.blit(title, (BOARD_MARGIN, 8))

        if self.game_over:
            msg = self.font.render(self.result_text, True, TEXT_COLOR)
            self.screen.blit(msg, (BOARD_MARGIN, BOARD_MARGIN + BOARD_SIZE + 5))
        else:
            turn = "Your turn" if self.current == self.human_mark else "Agent thinking..."
            msg = self.font.render(turn, True, TEXT_COLOR)
            self.screen.blit(msg, (BOARD_MARGIN, BOARD_MARGIN + BOARD_SIZE + 5))

        stats_text = (f"Session -> You: {self.stats['human_wins']}  "
                      f"Agent: {self.stats['agent_wins']}  Draws: {self.stats['draws']}")
        self.screen.blit(self.small_font.render(stats_text, True, TEXT_COLOR),
                          (BOARD_MARGIN, HEIGHT - 30))

    def draw_chart(self):
        rect = pygame.Rect(CHART_X, BOARD_MARGIN, CHART_W, BOARD_SIZE)
        pygame.draw.rect(self.screen, CHART_BG, rect, border_radius=8)

        title = self.font.render("Self-Play Training Progress", True, TEXT_COLOR)
        self.screen.blit(title, (CHART_X, 8))
        subtitle = self.small_font.render("Win / Draw / Loss rate vs. a random player", True, TEXT_COLOR)
        self.screen.blit(subtitle, (CHART_X, BOARD_MARGIN + BOARD_SIZE + 5))

        if len(self.history) < 2:
            return

        pad = 20
        plot_rect = pygame.Rect(rect.x + pad, rect.y + pad, rect.w - 2 * pad, rect.h - 2 * pad)
        episodes = [h[0] for h in self.history]
        max_ep = max(episodes)

        def to_point(ep, rate):
            px = plot_rect.x + (ep / max_ep) * plot_rect.w
            py = plot_rect.bottom - rate * plot_rect.h
            return (px, py)

        # gridlines at 0%, 50%, 100%
        for frac in (0.0, 0.5, 1.0):
            y = plot_rect.bottom - frac * plot_rect.h
            pygame.draw.line(self.screen, (55, 58, 66), (plot_rect.x, y), (plot_rect.right, y), 1)
            label = self.small_font.render(f"{int(frac * 100)}%", True, TEXT_COLOR)
            self.screen.blit(label, (plot_rect.x - 4, y - 18))

        for idx, color in ((1, CHART_LINE_WIN), (2, CHART_LINE_DRAW), (3, CHART_LINE_LOSS)):
            points = [to_point(h[0], h[idx]) for h in self.history]
            if len(points) >= 2:
                pygame.draw.lines(self.screen, color, False, points, 3)

        legend_y = plot_rect.y + plot_rect.h + 6
        legend = [("Win", CHART_LINE_WIN), ("Draw", CHART_LINE_DRAW), ("Loss", CHART_LINE_LOSS)]
        lx = plot_rect.x
        for text, color in legend:
            pygame.draw.rect(self.screen, color, (lx, legend_y, 14, 14))
            label = self.small_font.render(text, True, TEXT_COLOR)
            self.screen.blit(label, (lx + 20, legend_y - 2))
            lx += 90

        final_w, final_d, final_l = self.history[-1][1:]
        final_text = (f"Final vs random (after {episodes[-1]:,} training games): "
                       f"{final_w*100:.1f}% win, {final_d*100:.1f}% draw, {final_l*100:.1f}% loss")
        label = self.small_font.render(final_text, True, TEXT_COLOR)
        self.screen.blit(label, (CHART_X, BOARD_MARGIN + BOARD_SIZE + 24))

    def run(self):
        while True:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)

            if not self.game_over and self.current != self.human_mark:
                pygame.time.wait(250)  # tiny pause so the AI's move feels natural
                self.agent_move()

            self.screen.fill(BG)
            self.draw_board()
            self.draw_chart()
            self.btn_you_first.draw(self.screen, self.small_font, mouse_pos)
            self.btn_ai_first.draw(self.screen, self.small_font, mouse_pos)
            pygame.display.flip()
            self.clock.tick(30)


# ============================================================
# 5. Entry point
# ============================================================
if __name__ == "__main__":
    agent = QAgent()

    def progress_cb(ep, total, w, d, l):
        print(f"  episode {ep:>6}/{total}  win {w*100:5.1f}%  draw {d*100:5.1f}%  loss {l*100:5.1f}%")

    print("Training agent via self-play (playing itself thousands of times)...")
    history = train(agent, episodes=40000, eval_every=2000, eval_games=200, progress_cb=progress_cb)
    print(f"Training complete! Learned {len(agent.Q)} state-action values.\n")

    app = TicTacToeApp(agent, history)
    app.run()