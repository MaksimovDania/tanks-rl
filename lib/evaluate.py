"""Reward-independent evaluation suite for comparing agent versions."""

import csv
import os
import numpy as np


def evaluate(env, agent, n_episodes=100, seed=42):
    """
    Run n_episodes with greedy policy, return reward-independent metrics.

    Returns dict with:
        win_rate, survival_rate, castle_survival_rate, avg_episode_length,
        avg_kill_time, avg_shots_fired, shot_accuracy, exploration_coverage
    """
    wins = 0
    survivals = 0
    castle_survivals = 0
    episode_lengths = []
    kill_times = []
    shots_list = []
    coverage_list = []

    # Count walkable cells for coverage metric
    n_walkable = int((env.grid_template == 0).sum())

    for i in range(n_episodes):
        state, _ = env.reset(seed=seed + i)
        done = False
        steps = 0

        while not done:
            action = agent.act(state, greedy=True)
            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        episode_lengths.append(steps)
        shots_list.append(info["shots_fired"])
        coverage_list.append(info["cells_visited"])

        if info["win"]:
            wins += 1
            kill_times.append(steps)
        if info["player_alive"]:
            survivals += 1
        if info["castle_alive"]:
            castle_survivals += 1

    total_shots = sum(shots_list)

    return {
        "win_rate": wins / n_episodes,
        "survival_rate": survivals / n_episodes,
        "castle_survival_rate": castle_survivals / n_episodes,
        "avg_episode_length": np.mean(episode_lengths),
        "avg_kill_time": np.mean(kill_times) if kill_times else float("nan"),
        "avg_shots_fired": np.mean(shots_list),
        "shot_accuracy": wins / total_shots if total_shots > 0 else 0.0,
        "exploration_coverage": np.mean(coverage_list) / max(n_walkable, 1),
    }


def save_eval(metrics, episode, path):
    """Append eval metrics to CSV file."""
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode"] + list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow({"episode": episode, **metrics})
