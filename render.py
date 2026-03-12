"""Visualize a trained agent playing Battle City with pygame."""

import argparse
import os
import time

from lib.env import BattleCityEnv
from lib.agent import QLearningAgent


def main():
    parser = argparse.ArgumentParser(description="Watch trained agent play Battle City")
    parser.add_argument("run_dir", help="Path to run directory (e.g. runs/bold_curie)")
    parser.add_argument("--level", type=str, default=None, help="Level file (default: from config)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to play")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second")
    args = parser.parse_args()

    # Load config
    import json
    config_path = os.path.join(args.run_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    level_path = args.level or config.get("level", "levels/1.txt")

    env = BattleCityEnv(level_path=level_path, max_steps=500, render_mode="human")
    env.metadata["render_fps"] = args.fps

    agent = QLearningAgent(n_actions=6)
    agent.load(os.path.join(args.run_dir, "q_table.pkl"))
    print(f"Loaded Q-table with {len(agent.q_table)} states")

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.act(state, greedy=True)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        result = "WIN" if info["win"] else "LOSE"
        print(f"Episode {ep}: {result} | reward={total_reward:.1f} | steps={steps}")
        time.sleep(1)  # pause between episodes

    env.close()


if __name__ == "__main__":
    main()
