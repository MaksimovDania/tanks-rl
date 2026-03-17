from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Hashable, Protocol

import numpy as np

from agent import QLearningAgent, QLearningConfig
from env import LevelBattleCityEnv
from env_adapter import BattleCityEnvAdapter, StateIndexer

logger = logging.getLogger(__name__)

State = Hashable
DEFAULT_STATE_BASES: tuple[int, ...] = (17, 17, 4, 4, 2, 2, 2, 2, 2, 5, 5)
TRAIN_LOG_FIELDS: tuple[str, ...] = (
    "episode",
    "reward",
    "win",
    "win_rate",
    "epsilon",
    "steps",
    "td_error_mean",
    "q_table_states",
)


class EnvAdapterProtocol(Protocol):
    @property
    def n_actions(self) -> int:
        ...

    def reset(self, seed: int | None = None) -> tuple[State, dict[str, Any]]:
        ...

    def step(self, action: int) -> tuple[State, float, bool, bool, dict[str, Any]]:
        ...


@dataclass(slots=True, frozen=True)
class TrainConfig:
    episodes: int
    alpha: float
    gamma: float
    epsilon: float
    epsilon_min: float
    epsilon_decay: float
    seed: int
    max_steps: int
    level_map_path: Path
    artifacts_dir: Path
    state_mode: str
    checkpoint_every: int
    log_every: int
    win_rate_window: int


@dataclass(slots=True, frozen=True)
class TrainRow:
    episode: int
    reward: float
    win: int
    win_rate: float
    epsilon: float
    steps: int
    td_error_mean: float
    q_table_states: int


class CsvMetricLogger:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(TRAIN_LOG_FIELDS))
            writer.writeheader()

    def append(self, row: TrainRow) -> None:
        with self.csv_path.open("a", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(TRAIN_LOG_FIELDS))
            writer.writerow(asdict(row))


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning agent for Battle City.")
    parser.add_argument("--episodes", type=int, default=2_000)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--level-map-path", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--state-mode", choices=("tuple", "index"), default="tuple")
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--win-rate-window", type=int, default=100)

    args = parser.parse_args()
    return TrainConfig(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
        max_steps=args.max_steps,
        level_map_path=args.level_map_path,
        artifacts_dir=args.artifacts_dir,
        state_mode=args.state_mode,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        win_rate_window=args.win_rate_window,
    )


def setup_logging(artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifacts_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_config(config: TrainConfig) -> None:
    config_path = config.artifacts_dir / "config.json"
    payload: dict[str, Any] = asdict(config)
    payload["level_map_path"] = str(config.level_map_path)
    payload["artifacts_dir"] = str(config.artifacts_dir)
    with config_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def build_env_adapter(config: TrainConfig, episode_seed: int) -> EnvAdapterProtocol:
    level_map_str = config.level_map_path.read_text(encoding="utf-8")
    env = LevelBattleCityEnv(
        level_map_str=level_map_str,
        max_steps=config.max_steps,
        seed=episode_seed,
    )

    state_indexer = StateIndexer(DEFAULT_STATE_BASES) if config.state_mode == "index" else None
    return BattleCityEnvAdapter(env=env, state_indexer=state_indexer)


def build_agent(config: TrainConfig, n_actions: int) -> QLearningAgent:
    agent_config = QLearningConfig(
        n_actions=n_actions,
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon=config.epsilon,
        epsilon_min=config.epsilon_min,
        epsilon_decay=config.epsilon_decay,
        seed=config.seed,
    )
    return QLearningAgent(agent_config)


def train(config: TrainConfig) -> None:
    setup_logging(config.artifacts_dir)
    set_global_seed(config.seed)
    save_config(config)

    metrics_logger = CsvMetricLogger(config.artifacts_dir / "train_log.csv")
    win_window: deque[int] = deque(maxlen=config.win_rate_window)

    bootstrap_adapter = build_env_adapter(config, episode_seed=config.seed)
    agent = build_agent(config, n_actions=bootstrap_adapter.n_actions)

    logger.info("Training started: episodes=%d, state_mode=%s", config.episodes, config.state_mode)

    for episode in range(1, config.episodes + 1):
        adapter = build_env_adapter(config, episode_seed=config.seed + episode)
        state, _ = adapter.reset(seed=config.seed + episode)

        done = False
        episode_reward = 0.0
        steps = 0
        td_errors: list[float] = []
        info: dict[str, Any] = {}

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = adapter.step(action)
            td_error = agent.update(
                state=state,
                action=action,
                reward=float(reward),
                next_state=next_state,
                done=bool(terminated or truncated),
            )
            td_errors.append(abs(td_error))

            state = next_state
            episode_reward += float(reward)
            steps += 1
            done = bool(terminated or truncated)

        agent.decay_epsilon()

        win = int(bool(info.get("win", False)))
        win_window.append(win)
        row = TrainRow(
            episode=episode,
            reward=episode_reward,
            win=win,
            win_rate=float(mean(win_window)) if win_window else 0.0,
            epsilon=agent.epsilon,
            steps=steps,
            td_error_mean=float(mean(td_errors)) if td_errors else 0.0,
            q_table_states=agent.num_states(),
        )
        metrics_logger.append(row)

        if episode % config.log_every == 0 or episode == 1 or episode == config.episodes:
            logger.info(
                "episode=%d reward=%.3f win=%d win_rate=%.3f epsilon=%.4f steps=%d states=%d",
                row.episode,
                row.reward,
                row.win,
                row.win_rate,
                row.epsilon,
                row.steps,
                row.q_table_states,
            )

        if config.checkpoint_every > 0 and episode % config.checkpoint_every == 0:
            agent.save(config.artifacts_dir / "checkpoints" / f"q_agent_ep_{episode}.pkl")

    agent.save(config.artifacts_dir / "q_agent_final.pkl")
    logger.info("Training finished. Artifacts saved to %s", config.artifacts_dir)


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
