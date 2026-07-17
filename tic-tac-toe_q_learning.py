"""
Tic-Tac-Toe with Q-Learning
----------------------------
A small reinforcement learning project: an agent (X) learns to play
Tic-Tac-Toe through self-play using Q-learning, then you can play
against it.

Run it with:  python tic_tac_toe_q_learning.py
"""

import random

# ---------------------------------------------------------
# Board helpers
# ---------------------------------------------------------
EMPTY = " "
PLAYER_X = "X"  # the learning agent
PLAYER_O = "O"  # opponent (random during training, human during play)

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
]


def new_board():
    return [EMPTY] * 9


def winner(board):
    for a, b, c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    if EMPTY not in board:
        return "DRAW"
    return None


def available_moves(board):
    return [i for i, v in enumerate(board) if v == EMPTY]


def print_board(board):
    rows = [board[i:i + 3] for i in range(0, 9, 3)]
    print()
    for i, row in enumerate(rows):
        print(" " + " | ".join(row))
        if i < 2:
            print("---+---+---")
    print()


# ---------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------
class QAgent:
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=0.2):
        self.Q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _state(self, board):
        return tuple(board)

    def get_q(self, board, action):
        return self.Q.get((self._state(board), action), 0.0)

    def choose_action(self, board, explore=True):
        moves = available_moves(board)
        if explore and random.random() < self.epsilon:
            return random.choice(moves)
        values = [self.get_q(board, m) for m in moves]
        max_v = max(values)
        best = [m for m, v in zip(moves, values) if v == max_v]
        return random.choice(best)

    def update(self, board, action, reward, next_board, done):
        state = self._state(board)
        old_q = self.Q.get((state, action), 0.0)
        if done:
            future = 0.0
        else:
            next_moves = available_moves(next_board)
            future = max([self.get_q(next_board, m) for m in next_moves], default=0.0)
        self.Q[(state, action)] = old_q + self.alpha * (reward + self.gamma * future - old_q)


# ---------------------------------------------------------
# Training via self-play (agent as X vs a random-move O)
# ---------------------------------------------------------
def train(agent, episodes=20000):
    for _ in range(episodes):
        board = new_board()
        history = []  # (board_before, action) pairs for X's moves

        while True:
            # --- X's turn (the learning agent) ---
            action = agent.choose_action(board, explore=True)
            board_before = board.copy()
            board[action] = PLAYER_X
            history.append((board_before, action))

            result = winner(board)
            if result is not None:
                reward = 1 if result == PLAYER_X else 0 if result == "DRAW" else -1
                agent.update(board_before, action, reward, board, True)
                break

            # --- O's turn (random opponent) ---
            o_move = random.choice(available_moves(board))
            board[o_move] = PLAYER_O

            result = winner(board)
            if result is not None:
                reward = -1 if result == PLAYER_O else 0 if result == "DRAW" else 1
                agent.update(board_before, action, reward, board, True)
                break
            else:
                # game continues: normal update using the board after O's move
                agent.update(board_before, action, 0, board, False)

    return agent


# ---------------------------------------------------------
# Play against the trained agent
# ---------------------------------------------------------
def play_against_agent(agent):
    board = new_board()
    print("You are O. Board positions are numbered 0-8 like this:\n")
    print_board([str(i) for i in range(9)])

    while True:
        # Agent (X) move — no exploration, pick its best learned move
        action = agent.choose_action(board, explore=False)
        board[action] = PLAYER_X
        print(f"Agent (X) plays position {action}")
        print_board(board)

        result = winner(board)
        if result is not None:
            announce(result)
            break

        # Human move
        moves = available_moves(board)
        move = None
        while move not in moves:
            try:
                move = int(input(f"Your move {moves}: "))
            except ValueError:
                continue
        board[move] = PLAYER_O
        print_board(board)

        result = winner(board)
        if result is not None:
            announce(result)
            break


def announce(result):
    if result == "DRAW":
        print("It's a draw!")
    elif result == PLAYER_X:
        print("Agent (X) wins!")
    else:
        print("You (O) win!")


if __name__ == "__main__":
    agent = QAgent()
    print("Training agent via self-play (this takes a few seconds)...")
    train(agent, episodes=20000)
    print(f"Training complete! Learned {len(agent.Q)} state-action values.\n")

    while True:
        play_against_agent(agent)
        again = input("\nPlay again? (y/n): ").strip().lower()
        if again != "y":
            break