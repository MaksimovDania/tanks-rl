"""Generate training and evaluation plots from log files."""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def _rolling_avg(data, window=1000):
    """Compute rolling average with given window."""
    if len(data) < window:
        window = max(len(data) // 5, 1)
    cumsum = np.cumsum(data)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    result = np.full_like(data, np.nan, dtype=float)
    result[window - 1:] = cumsum[window - 1:] / window
    return result


def _read_train_log(path):
    """Read train_log.csv into dict of lists."""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                data.setdefault(key, []).append(float(val))
    return data


def _read_eval_log(path):
    """Read eval_log.csv into dict of lists."""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                try:
                    data.setdefault(key, []).append(float(val))
                except ValueError:
                    data.setdefault(key, []).append(float("nan"))
    return data


def generate_plots(run_dir):
    """Generate all training plots and save to run_dir/plots/."""
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    train_path = os.path.join(run_dir, "train_log.csv")
    eval_path = os.path.join(run_dir, "eval_log.csv")

    train = _read_train_log(train_path)
    eval_data = _read_eval_log(eval_path) if os.path.exists(eval_path) else {}

    fig_size = (10, 5)

    # 1. Episode reward (rolling avg)
    if "reward" in train:
        plt.figure(figsize=fig_size)
        rewards = np.array(train["reward"])
        plt.plot(rewards, alpha=0.1, color="blue", label="raw")
        plt.plot(_rolling_avg(rewards), color="blue", label="rolling avg (1000)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "reward.png"), dpi=100)
        plt.close()

    # 2. Win rate over time (from eval)
    if "win_rate" in eval_data:
        plt.figure(figsize=fig_size)
        plt.plot(eval_data["episode"], eval_data["win_rate"], marker="o")
        plt.xlabel("Episode")
        plt.ylabel("Win Rate")
        plt.title("Win Rate Over Training")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "win_rate.png"), dpi=100)
        plt.close()

    # 3. Q-value statistics
    for metric in ["mean_q", "max_q", "std_q"]:
        if metric in train:
            plt.figure(figsize=fig_size)
            vals = np.array(train[metric])
            plt.plot(_rolling_avg(vals), label=metric)
            plt.xlabel("Episode")
            plt.ylabel(metric)
            plt.title(f"Q-value: {metric}")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"q_{metric}.png"), dpi=100)
            plt.close()

    # 4. TD error and Q-delta (learning dynamics)
    for metric, title in [("mean_td_error", "TD Error"), ("mean_q_delta", "Q-value Delta (grad norm analog)")]:
        if metric in train:
            plt.figure(figsize=fig_size)
            vals = np.array(train[metric])
            plt.plot(vals, alpha=0.1, color="red")
            plt.plot(_rolling_avg(vals), color="red")
            plt.xlabel("Episode")
            plt.ylabel(metric)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{metric}.png"), dpi=100)
            plt.close()

    # 5. Epsilon decay
    if "epsilon" in train:
        plt.figure(figsize=fig_size)
        plt.plot(train["epsilon"])
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Exploration Rate (Epsilon)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "epsilon.png"), dpi=100)
        plt.close()

    # 6. State space coverage
    if "n_states" in train:
        plt.figure(figsize=fig_size)
        plt.plot(train["n_states"])
        plt.xlabel("Episode")
        plt.ylabel("Unique States Visited")
        plt.title("State Space Coverage")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "state_coverage.png"), dpi=100)
        plt.close()

    # 7. Episode length
    if "episode_length" in train:
        plt.figure(figsize=fig_size)
        lengths = np.array(train["episode_length"])
        plt.plot(lengths, alpha=0.1, color="green")
        plt.plot(_rolling_avg(lengths), color="green")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.title("Episode Length")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "episode_length.png"), dpi=100)
        plt.close()

    # 8. Eval metrics dashboard
    if eval_data:
        metrics_to_plot = ["survival_rate", "castle_survival_rate", "avg_episode_length",
                           "avg_shots_fired", "shot_accuracy", "exploration_coverage"]
        available = [m for m in metrics_to_plot if m in eval_data]
        if available:
            n = len(available)
            cols = 2
            rows = (n + 1) // 2
            fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
            axes = axes.flatten() if n > 1 else [axes]
            for i, metric in enumerate(available):
                axes[i].plot(eval_data["episode"], eval_data[metric], marker="o")
                axes[i].set_title(metric.replace("_", " ").title())
                axes[i].set_xlabel("Episode")
                axes[i].grid(True, alpha=0.3)
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "eval_dashboard.png"), dpi=100)
            plt.close()

    print(f"Plots saved to {plots_dir}/")
