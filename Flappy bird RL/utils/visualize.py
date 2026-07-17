import matplotlib.pyplot as plt


def moving_average(data, window):
    if window <= 1:
        return data
    import numpy as np
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_rewards(rewards, window=50, title="Training Rewards", save_path=None):
    """Plot episode rewards and a smoothed curve.

    Args:
        rewards (list[float]): reward value per episode.
        window (int): smoothing window for moving average.
        title (str): plot title.
        save_path (str|None): optional path to save the figure.

    Example:
        from utils.visualize import plot_rewards
        plot_rewards(reward_history, window=20, save_path="rewards.png")
    """
    plt.figure(figsize=(10,5))
    plt.plot(rewards, color="lightgray", label="Episode rewards")

    if len(rewards) >= window and window > 1:
        ma = moving_average(rewards, window)
        plt.plot(range(window - 1, window - 1 + len(ma)), ma, color="tab:blue", label=f"MA({window})")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def plot_scores(scores, window=50, title="Training Scores", save_path=None):
    plt.figure(figsize=(10,5))
    plt.plot(scores, color="lightgray", label="Episode scores")
    if len(scores) >= window and window > 1:
        ma = moving_average(scores, window)
        plt.plot(range(window - 1, window - 1 + len(ma)), ma, color="tab:green", label=f"MA({window})")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
