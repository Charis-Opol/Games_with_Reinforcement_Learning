import matplotlib.pyplot as plt


class LivePlot:
    def __init__(self, window=50, figsize=(10, 5)):
        plt.ion()
        self.window = window
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=figsize)
        self.rewards = []
        self.scores = []
        self.reward_line, = self.ax1.plot([], [], color="tab:blue", label="Episode Reward")
        self.reward_ma_line, = self.ax1.plot([], [], color="tab:orange", label=f"MA({window})")
        self.ax1.set_ylabel("Reward")
        self.ax1.legend()

        self.score_line, = self.ax2.plot([], [], color="tab:green", label="Episode Score")
        self.score_ma_line, = self.ax2.plot([], [], color="tab:olive", label=f"MA({window})")
        self.ax2.set_ylabel("Score")
        self.ax2.set_xlabel("Episode")
        self.ax2.legend()

        self.fig.tight_layout()

    def _moving_average(self, data):
        if len(data) < self.window or self.window <= 1:
            return []
        import numpy as np
        return np.convolve(data, np.ones(self.window) / self.window, mode="valid")

    def update(self, reward, score, episode=None, pause=0.001):
        self.rewards.append(reward)
        self.scores.append(score)

        episodes = list(range(len(self.rewards)))

        # update reward lines
        self.reward_line.set_data(episodes, self.rewards)
        ma = self._moving_average(self.rewards)
        if len(ma) > 0:
            self.reward_ma_line.set_data(range(self.window - 1, self.window - 1 + len(ma)), ma)
        else:
            self.reward_ma_line.set_data([], [])
        self.ax1.relim()
        self.ax1.autoscale_view()

        # update score lines
        self.score_line.set_data(episodes, self.scores)
        ma_s = self._moving_average(self.scores)
        if len(ma_s) > 0:
            self.score_ma_line.set_data(range(self.window - 1, self.window - 1 + len(ma_s)), ma_s)
        else:
            self.score_ma_line.set_data([], [])
        self.ax2.relim()
        self.ax2.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(pause)

    def close(self):
        plt.ioff()
        plt.close(self.fig)
