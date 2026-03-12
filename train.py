"""Training entry point for Battle City Q-learning agent."""

import argparse
import csv
import json
import os
import random

from lib.env import BattleCityEnv
from lib.agent import QLearningAgent
from lib.evaluate import evaluate, save_eval
from lib.plots import generate_plots

# Random run name generator
ADJECTIVES = [
    "bold", "calm", "cool", "dark", "fast", "keen", "loud", "neat", "rare",
    "warm", "wise", "wild", "lazy", "busy", "epic", "slim", "fair", "grim",
]
SCIENTISTS = [
    "newton", "curie", "euler", "gauss", "tesla", "fermi", "dirac", "bohr",
    "planck", "faraday", "pascal", "turing", "lorenz", "hubble", "kepler",
]


def make_run_name():
    return f"{random.choice(ADJECTIVES)}_{random.choice(SCIENTISTS)}"


def main():
    parser = argparse.ArgumentParser(description="Train Q-learning agent on Battle City")
    parser.add_argument("--level", type=str, default="levels/1.txt", help="Path to level file")
    parser.add_argument("--episodes", type=int, default=50000, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="Final epsilon")
    parser.add_argument("--epsilon-decay", type=int, default=None, help="Epsilon decay steps (default: 80%% of episodes)")
    parser.add_argument("--eval-interval", type=int, default=1000, help="Evaluate every N episodes")
    parser.add_argument("--run-name", type=str, default=None, help="Run name (auto-generated if not set)")
    args = parser.parse_args()

    if args.epsilon_decay is None:
        args.epsilon_decay = int(args.episodes * 0.8)

    run_name = args.run_name or make_run_name()
    run_dir = os.path.join("runs", run_name)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    # Save config
    config = vars(args)
    config["run_name"] = run_name
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Run: {run_name}")
    print(f"Config: {config}")
    print()

    # Init
    env = BattleCityEnv(level_path=args.level, max_steps=args.max_steps)
    agent = QLearningAgent(
        n_actions=6,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay,
    )

    train_log_path = os.path.join(run_dir, "train_log.csv")
    eval_log_path = os.path.join(run_dir, "eval_log.csv")

    # Training CSV header
    train_fields = [
        "episode", "reward", "episode_length", "epsilon",
        "mean_td_error", "mean_q_delta", "mean_q", "max_q", "std_q", "n_states",
    ]
    with open(train_log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=train_fields).writeheader()

    # Training loop
    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update(state, action, reward, next_state, terminated or truncated)
            state = next_state
            total_reward += reward
            steps += 1
            done = terminated or truncated

        agent.decay_epsilon(ep)

        # Log training stats
        ep_stats = agent.get_episode_stats()
        q_stats = agent.get_q_stats()
        row = {
            "episode": ep,
            "reward": total_reward,
            "episode_length": steps,
            "epsilon": agent.epsilon,
            "mean_td_error": ep_stats["mean_td_error"],
            "mean_q_delta": ep_stats["mean_q_delta"],
            "mean_q": q_stats["mean_q"],
            "max_q": q_stats["max_q"],
            "std_q": q_stats["std_q"],
            "n_states": q_stats["n_states"],
        }
        with open(train_log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=train_fields).writerow(row)

        # Periodic eval + console print
        if ep % args.eval_interval == 0:
            metrics = evaluate(env, agent, n_episodes=100, seed=42)
            save_eval(metrics, ep, eval_log_path)

            print(
                f"Ep {ep:6d} | "
                f"WinRate: {metrics['win_rate']:.2f} | "
                f"AvgReward: {total_reward:7.1f} | "
                f"AvgLen: {metrics['avg_episode_length']:5.0f} | "
                f"TDErr: {ep_stats['mean_td_error']:.3f} | "
                f"QDelta: {ep_stats['mean_q_delta']:.3f} | "
                f"States: {q_stats['n_states']} | "
                f"ε: {agent.epsilon:.3f}"
            )

    # Save Q-table
    agent.save(os.path.join(run_dir, "q_table.pkl"))
    print(f"\nQ-table saved ({q_stats['n_states']} states)")

    # Generate plots
    generate_plots(run_dir)

    # Final eval
    print("\n── Final Evaluation ──")
    final_metrics = evaluate(env, agent, n_episodes=100, seed=42)
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nRun complete: {run_dir}/")


if __name__ == "__main__":
    main()
