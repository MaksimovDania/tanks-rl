from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

Observation = tuple[int, ...]
State = int | Observation


@dataclass(slots=True, frozen=True)
class StateIndexer:
    """Encodes a discrete tuple state into a single integer index.

    Example:
        state=(s0, s1, s2), bases=(b0, b1, b2)
        index = s0 + s1*b0 + s2*b0*b1

    This is useful when the environment already returns a compact discrete
    observation but you want to store Q-values by integer state id.
    """

    bases: tuple[int, ...]

    def __init__(self, bases: Sequence[int]) -> None:
        object.__setattr__(self, "bases", tuple(int(base) for base in bases))
        if not self.bases:
            raise ValueError("bases must not be empty")
        if any(base <= 0 for base in self.bases):
            raise ValueError("all bases must be positive integers")

    def encode(self, state: Sequence[int]) -> int:
        if len(state) != len(self.bases):
            raise ValueError(f"state length {len(state)} does not match bases length {len(self.bases)}")

        index = 0
        multiplier = 1
        for value, base in zip(state, self.bases):
            int_value = int(value)
            if int_value < 0 or int_value >= base:
                raise ValueError(f"state value {int_value} is out of range [0, {base - 1}]")
            index += int_value * multiplier
            multiplier *= base
        return index

    def decode(self, index: int) -> Observation:
        if index < 0:
            raise ValueError("index must be non-negative")

        remainder = int(index)
        values: list[int] = []
        for base in self.bases:
            values.append(remainder % base)
            remainder //= base
        return tuple(values)

    @property
    def n_states(self) -> int:
        total = 1
        for base in self.bases:
            total *= base
        return total


class BattleCityEnvAdapter:
    """Thin wrapper over env.py.

    Responsibilities:
    - normalize raw env observations into a stable tuple[int, ...]
    - optionally encode that tuple into a single integer via StateIndexer
    - expose a gym-like reset/step contract used by train.py
    """

    def __init__(self, env: Any, state_indexer: StateIndexer | None = None) -> None:
        self.env = env
        self.state_indexer = state_indexer

    @property
    def n_actions(self) -> int:
        action_space = getattr(self.env, "action_space", None)
        if action_space is not None and hasattr(action_space, "n"):
            return int(action_space.n)

        n_actions = getattr(self.env, "N_ACTIONS", None)
        if n_actions is not None:
            return int(n_actions)

        raise AttributeError("Unable to infer number of actions from the wrapped environment")

    def reset(self, seed: int | None = None) -> tuple[State, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed)
        return self._adapt_state(observation), dict(info)

    def step(self, action: int) -> tuple[State, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(int(action))
        return (
            self._adapt_state(observation),
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info),
        )

    def _adapt_state(self, observation: Any) -> State:
        discrete_state = self._to_discrete_tuple(observation)
        if self.state_indexer is None:
            return discrete_state
        return self.state_indexer.encode(discrete_state)

    @staticmethod
    def _to_discrete_tuple(observation: Any) -> Observation:
        if isinstance(observation, tuple):
            return tuple(int(value) for value in observation)
        if isinstance(observation, list):
            return tuple(int(value) for value in observation)
        if hasattr(observation, "tolist"):
            materialized = observation.tolist()
            if isinstance(materialized, Iterable):
                return tuple(int(value) for value in materialized)
        raise TypeError(f"Unsupported observation type for discretization: {type(observation)!r}")
