from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, DefaultDict, Hashable

import numpy as np

logger = logging.getLogger(__name__)

State = Hashable


@dataclass(slots=True, frozen=True)
class QLearningConfig:
    n_actions: int
    alpha: float
    gamma: float
    epsilon: float
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9995
    seed: int = 42


class QLearningAgent:
    """Tabular Q-learning agent.

    Supports both integer state ids and hashable tuple states.
    For tuple states the Q-table is stored as a dict: state -> np.ndarray[n_actions].
    """

    def __init__(self, config: QLearningConfig) -> None:
        self.config = config
        self.n_actions = config.n_actions
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self._rng = np.random.default_rng(config.seed)
        self._q_table: DefaultDict[State, np.ndarray] = defaultdict(self._zero_row)

    def _zero_row(self) -> np.ndarray:
        return np.zeros(self.n_actions, dtype=np.float32)

    def _argmax_random_tie(self, q_values: np.ndarray) -> int:
        max_q: float = float(np.max(q_values))
        best_actions: np.ndarray = np.flatnonzero(q_values == max_q)
        return int(self._rng.choice(best_actions))

    def get_action(self, state: State, greedy: bool = False) -> int:
        if not greedy and float(self._rng.random()) < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return self._argmax_random_tie(self._q_table[state])

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> float:
        current_q: float = float(self._q_table[state][action])
        max_next_q: float = 0.0 if done else float(np.max(self._q_table[next_state]))
        td_target: float = reward + self.gamma * max_next_q
        td_error: float = td_target - current_q

        self._q_table[state][action] = current_q + self.alpha * td_error
        return td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def q_values(self, state: State) -> np.ndarray:
        return self._q_table[state].copy()

    def state_value(self, state: State) -> float:
        return float(np.max(self._q_table[state]))

    def num_states(self) -> int:
        return len(self._q_table)

    def save(self, path: str | Path) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "config": asdict(self.config),
            "epsilon": self.epsilon,
            "q_table": {state: values.copy() for state, values in self._q_table.items()},
        }

        with checkpoint_path.open("wb") as file_obj:
            pickle.dump(payload, file_obj)

        logger.info("Q-table saved to %s", checkpoint_path)
        return checkpoint_path

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        checkpoint_path = Path(path)
        with checkpoint_path.open("rb") as file_obj:
            payload: dict[str, Any] = pickle.load(file_obj)

        agent = cls(QLearningConfig(**payload["config"]))
        agent.epsilon = float(payload.get("epsilon", agent.epsilon))
        restored_table: dict[State, np.ndarray] = payload.get("q_table", {})
        agent._q_table = defaultdict(
            agent._zero_row,
            {state: np.asarray(values, dtype=np.float32) for state, values in restored_table.items()},
        )

        logger.info("Q-table loaded from %s", checkpoint_path)
        return agent
