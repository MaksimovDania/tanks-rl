from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

# Actions
NOOP: int = 0
MOVE_UP: int = 1
MOVE_RIGHT: int = 2
MOVE_DOWN: int = 3
MOVE_LEFT: int = 4
SHOOT: int = 5

# Directions
UP: int = 0
RIGHT: int = 1
DOWN: int = 2
LEFT: int = 3

N_ACTIONS: int = 6

# Grid cells
EMPTY: int = 0
BRICK: int = 1
STEEL: int = 2
WATER: int = 3
CASTLE: int = 4
BUSH: int = 5

BLOCKING_CELLS_MOVE: set[int] = {BRICK, STEEL, WATER, CASTLE}
BLOCKING_CELLS_SHOT: set[int] = {BRICK, STEEL, CASTLE}

Grid = NDArray[np.int32]
Position = list[int]
Observation = tuple[int, int, int, int, int, int, int, int, int, int, int]
Bullet = dict[str, Any]


@dataclass(slots=True)
class SimpleDiscrete:
    n: int

    def sample(self) -> int:
        return int(np.random.randint(self.n))


def dist_bin(distance: int) -> int:
    if distance <= 0:
        return 0
    if distance <= 2:
        return 1
    if distance <= 4:
        return 2
    if distance <= 6:
        return 3
    return 4


class SimpleBattleCityEnv:
    def __init__(self, grid_size: int = 9, max_steps: int = 150, seed: int = 42) -> None:
        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.rng: Generator = np.random.default_rng(seed)
        self.action_space = SimpleDiscrete(N_ACTIONS)

        self.grid: Grid | None = None
        self.grid_template: Grid | None = None
        self.player_pos: Position | None = None
        self.enemy_pos: Position | None = None
        self.castle_pos: Position | None = None

        self.player_dir: int = UP
        self.enemy_dir: int = DOWN

        self.player_bullet: Bullet | None = None
        self.enemy_bullet: Bullet | None = None

        self.player_alive: bool = True
        self.enemy_alive: bool = True
        self.castle_alive: bool = True

        self.step_count: int = 0
        self.score: float = 0.0

        self.shots_fired: int = 0
        self.hits_on_enemy: int = 0
        self.visited_cells: set[tuple[int, int]] = set()

    def reset(self, seed: int | None = None, **_: Any) -> tuple[Observation, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.grid = self._build_map()
        self.grid_template = self.grid.copy()

        self.castle_pos = [self.grid_size - 1, self.grid_size // 2]
        self.player_pos = [self.grid_size - 2, self.grid_size // 2]
        self.enemy_pos = [1, self.grid_size // 2]

        self.player_dir = UP
        self.enemy_dir = DOWN
        self.player_bullet = None
        self.enemy_bullet = None
        self.player_alive = True
        self.enemy_alive = True
        self.castle_alive = True
        self.step_count = 0
        self.score = 0.0
        self.shots_fired = 0
        self.hits_on_enemy = 0
        self.visited_cells = {tuple(self.player_pos)}

        obs = self._get_obs()
        info = self._get_info(win=False, trade=False)
        return obs, info

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        reward = -0.02

        prev_enemy_alive = self.enemy_alive
        prev_player_alive = self.player_alive
        prev_castle_alive = self.castle_alive
        prev_obs = self._get_obs()
        prev_enemy_dist = self._manhattan(self.player_pos, self.enemy_pos) if self.enemy_alive else 0

        reward += self._apply_player_action(action)

        if self.enemy_alive and self.player_alive and self.castle_alive:
            enemy_action = self._enemy_policy()
            reward += self._apply_enemy_action(enemy_action)

        reward += self._advance_bullets()
        obs = self._get_obs()

        if self.enemy_alive and self.player_alive:
            new_enemy_dist = self._manhattan(self.player_pos, self.enemy_pos)
            if new_enemy_dist < prev_enemy_dist:
                reward += 0.05

        clear_enemy = obs[6]
        facing_enemy = obs[7]
        castle_threat = obs[8]

        if clear_enemy and not prev_obs[6]:
            reward += 0.5

        if clear_enemy and facing_enemy:
            reward += 0.10
            if action == SHOOT and self.player_bullet is not None:
                reward += 1.0
        elif action == SHOOT and self.player_bullet is None:
            reward -= 0.10

        if prev_obs[8] and not castle_threat:
            reward += 0.5
        if castle_threat:
            reward -= 0.10

        if obs[10] > prev_obs[10]:
            reward += 0.05

        enemy_killed_now = prev_enemy_alive and (not self.enemy_alive)
        player_died_now = prev_player_alive and (not self.player_alive)
        castle_died_now = prev_castle_alive and (not self.castle_alive)

        if enemy_killed_now and self.player_alive and self.castle_alive:
            reward += 100.0
            self.score += 100.0
        elif enemy_killed_now and (not self.player_alive):
            reward += 20.0
            self.score += 20.0

        if player_died_now:
            reward -= 80.0
            self.score -= 80.0

        if castle_died_now:
            reward -= 120.0
            self.score -= 120.0

        win = bool((not self.enemy_alive) and self.player_alive and self.castle_alive)
        trade = bool((not self.enemy_alive) and (not self.player_alive))

        terminated = (not self.player_alive) or (not self.enemy_alive) or (not self.castle_alive)
        truncated = (self.step_count >= self.max_steps) and (not terminated)

        if truncated and self.enemy_alive:
            reward -= 10.0
            self.score -= 10.0

        obs = self._get_obs()
        info = self._get_info(win=win, trade=trade)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        return None

    def _build_map(self) -> Grid:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        grid[self.grid_size - 1, self.grid_size // 2] = CASTLE
        grid[2, 2:7] = BRICK
        grid[2, 4] = EMPTY
        grid[4, 1] = STEEL
        grid[4, 7] = STEEL
        grid[4, 3] = BRICK
        grid[4, 5] = BRICK
        grid[6, 2] = BRICK
        grid[6, 4] = BRICK
        grid[6, 6] = BRICK
        grid[7, 3] = STEEL
        grid[7, 5] = STEEL
        grid[1, 1] = WATER
        grid[1, 7] = WATER
        grid[3, 1] = BUSH
        grid[3, 7] = BUSH
        grid[5, 2] = BUSH
        grid[5, 6] = BUSH
        return grid

    def _get_info(self, win: bool = False, trade: bool = False) -> dict[str, Any]:
        assert self.grid_template is not None
        walkable_mask = ~np.isin(self.grid_template, list(BLOCKING_CELLS_MOVE))
        n_walkable = int(np.sum(walkable_mask))
        return {
            "env_version": "battle_city_v3_logged",
            "score": float(self.score),
            "win": bool(win),
            "trade": bool(trade),
            "castle_survived": bool(self.castle_alive),
            "castle_alive": bool(self.castle_alive),
            "player_alive": bool(self.player_alive),
            "enemy_alive": bool(self.enemy_alive),
            "steps": int(self.step_count),
            "shots_fired": int(self.shots_fired),
            "enemy_hits": int(self.hits_on_enemy),
            "cells_visited": int(len(self.visited_cells)),
            "n_walkable_cells": int(n_walkable),
        }

    def _pos_to_idx(self, pos: Position) -> int:
        return int(pos[0] * self.grid_size + pos[1])

    def _get_obs(self) -> Observation:
        assert self.player_pos is not None
        assert self.enemy_pos is not None
        assert self.castle_pos is not None

        pr, pc = self.player_pos
        er, ec = self.enemy_pos

        dr_enemy = (er - pr) + (self.grid_size - 1)
        dc_enemy = (ec - pc) + (self.grid_size - 1)

        player_to_castle = self._manhattan(self.player_pos, self.castle_pos)
        enemy_to_castle = self._manhattan(self.enemy_pos, self.castle_pos)

        aligned_enemy = int(self._aligned(self.player_pos, self.enemy_pos))
        clear_enemy = int(aligned_enemy and self._clear_line(self.player_pos, self.enemy_pos))
        facing_enemy = int(self._facing_target(self.player_pos, self.enemy_pos, self.player_dir))
        castle_threat = int(
            self.enemy_alive
            and self._aligned(self.enemy_pos, self.castle_pos)
            and self._clear_line(self.enemy_pos, self.castle_pos)
        )

        return (
            int(dr_enemy),
            int(dc_enemy),
            int(self.player_dir),
            int(self.enemy_dir),
            int(self.player_bullet is not None),
            int(aligned_enemy),
            int(clear_enemy),
            int(facing_enemy),
            int(castle_threat),
            int(dist_bin(player_to_castle)),
            int(dist_bin(enemy_to_castle)),
        )

    def _apply_player_action(self, action: int) -> float:
        reward = 0.0
        if not self.player_alive:
            return reward
        if action == NOOP:
            return reward
        if action == SHOOT:
            self.shots_fired += 1
            reward += self._shoot(owner="player")
            return reward

        self.player_dir = self._action_to_dir(action)
        next_pos = self._next_position(self.player_pos, self.player_dir)
        if self._can_move_to(next_pos, moving_unit="player"):
            self.player_pos = next_pos
            self.visited_cells.add(tuple(self.player_pos))
        return reward

    def _apply_enemy_action(self, action: int) -> float:
        reward = 0.0
        if not self.enemy_alive:
            return reward
        if action == SHOOT:
            reward += self._shoot(owner="enemy")
            return reward
        if action in (MOVE_UP, MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT):
            self.enemy_dir = self._action_to_dir(action)
            next_pos = self._next_position(self.enemy_pos, self.enemy_dir)
            if self._can_move_to(next_pos, moving_unit="enemy"):
                self.enemy_pos = next_pos
        return reward

    def _shoot(self, owner: str = "player") -> float:
        reward = 0.0
        if owner == "player":
            if self.player_bullet is not None or not self.player_alive:
                return reward
            bullet = {"owner": "player", "pos": self._next_position(self.player_pos, self.player_dir), "dir": int(self.player_dir)}
            reward += self._resolve_bullet_spawn(bullet)
        else:
            if self.enemy_bullet is not None or not self.enemy_alive:
                return reward
            bullet = {"owner": "enemy", "pos": self._next_position(self.enemy_pos, self.enemy_dir), "dir": int(self.enemy_dir)}
            reward += self._resolve_bullet_spawn(bullet)
        return reward

    def _resolve_bullet_spawn(self, bullet: Bullet) -> float:
        assert self.grid is not None
        reward = 0.0
        r, c = bullet["pos"]

        if not self._in_bounds((r, c)):
            return reward

        cell = int(self.grid[r, c])
        if cell == BRICK:
            self.grid[r, c] = EMPTY
            if bullet["owner"] == "player":
                self.score += 5
                reward += 0.2
            return reward

        if cell in (STEEL, WATER):
            return reward

        if bullet["owner"] == "player":
            if self.enemy_alive and [r, c] == self.enemy_pos:
                self.enemy_alive = False
                self.hits_on_enemy += 1
                self.score += 100
                reward += 5.0
                return reward
            self.player_bullet = bullet
        else:
            if self.player_alive and [r, c] == self.player_pos:
                self.player_alive = False
                self.score -= 50
                reward -= 5.0
                return reward
            if self.castle_alive and [r, c] == self.castle_pos:
                self.castle_alive = False
                self.score -= 100
                reward -= 10.0
                return reward
            self.enemy_bullet = bullet
        return reward

    def _advance_bullets(self) -> float:
        reward = 0.0
        self.player_bullet, reward_player = self._advance_single_bullet(self.player_bullet)
        self.enemy_bullet, reward_enemy = self._advance_single_bullet(self.enemy_bullet)
        reward += reward_player + reward_enemy

        if self.player_bullet is not None and self.enemy_bullet is not None:
            if self.player_bullet["pos"] == self.enemy_bullet["pos"]:
                self.player_bullet = None
                self.enemy_bullet = None

        return reward

    def _advance_single_bullet(self, bullet: Bullet | None) -> tuple[Bullet | None, float]:
        assert self.grid is not None
        if bullet is None:
            return None, 0.0

        reward = 0.0
        next_pos = self._next_position(bullet["pos"], int(bullet["dir"]))
        if not self._in_bounds(next_pos):
            return None, reward

        r, c = next_pos
        cell = int(self.grid[r, c])

        if cell == BRICK:
            self.grid[r, c] = EMPTY
            if bullet["owner"] == "player":
                self.score += 5
                reward += 0.2
            return None, reward

        if cell in (STEEL, WATER):
            return None, reward

        if bullet["owner"] == "player":
            if self.enemy_alive and [r, c] == self.enemy_pos:
                self.enemy_alive = False
                self.hits_on_enemy += 1
                self.score += 100
                reward += 5.0
                return None, reward

        if bullet["owner"] == "enemy":
            if self.player_alive and [r, c] == self.player_pos:
                self.player_alive = False
                self.score -= 50
                reward -= 5.0
                return None, reward
            if self.castle_alive and [r, c] == self.castle_pos:
                self.castle_alive = False
                self.score -= 100
                reward -= 10.0
                return None, reward

        bullet["pos"] = list(next_pos)
        return bullet, reward

    def _enemy_policy(self) -> int:
        assert self.enemy_pos is not None
        assert self.castle_pos is not None
        if not self.enemy_alive:
            return NOOP

        if self._aligned(self.enemy_pos, self.castle_pos) and self._clear_line(self.enemy_pos, self.castle_pos):
            if self._facing_target(self.enemy_pos, self.castle_pos, self.enemy_dir) and self.enemy_bullet is None:
                return SHOOT

        if self.player_alive and self._aligned(self.enemy_pos, self.player_pos) and self._clear_line(self.enemy_pos, self.player_pos):
            if self._facing_target(self.enemy_pos, self.player_pos, self.enemy_dir):
                if self.enemy_bullet is None and self.rng.random() < 0.8:
                    return SHOOT

        target = self.player_pos if self.player_alive and self.rng.random() < 0.3 else self.castle_pos
        return self._move_towards(self.enemy_pos, target)

    def _action_to_dir(self, action: int) -> int:
        if action == MOVE_UP:
            return UP
        if action == MOVE_RIGHT:
            return RIGHT
        if action == MOVE_DOWN:
            return DOWN
        if action == MOVE_LEFT:
            return LEFT
        return UP

    def _next_position(self, pos: Position, direction: int) -> Position:
        r, c = pos
        if direction == UP:
            return [r - 1, c]
        if direction == RIGHT:
            return [r, c + 1]
        if direction == DOWN:
            return [r + 1, c]
        if direction == LEFT:
            return [r, c - 1]
        return [r, c]

    def _in_bounds(self, pos: tuple[int, int] | Position) -> bool:
        r, c = int(pos[0]), int(pos[1])
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _can_move_to(self, pos: Position, moving_unit: str = "player") -> bool:
        assert self.grid is not None
        if not self._in_bounds(pos):
            return False
        r, c = pos
        if int(self.grid[r, c]) in BLOCKING_CELLS_MOVE:
            return False
        if moving_unit == "player" and self.enemy_alive and pos == self.enemy_pos:
            return False
        if moving_unit == "enemy" and self.player_alive and pos == self.player_pos:
            return False
        return True

    @staticmethod
    def _manhattan(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _aligned(a: Position, b: Position) -> bool:
        return a[0] == b[0] or a[1] == b[1]

    @staticmethod
    def _facing_target(src: Position, dst: Position, facing: int) -> bool:
        if src[0] == dst[0]:
            return facing == (RIGHT if dst[1] > src[1] else LEFT)
        if src[1] == dst[1]:
            return facing == (DOWN if dst[0] > src[0] else UP)
        return False

    def _clear_line(self, src: Position, dst: Position) -> bool:
        assert self.grid is not None
        r1, c1 = src
        r2, c2 = dst
        if r1 == r2:
            step = 1 if c2 > c1 else -1
            for c in range(c1 + step, c2, step):
                if int(self.grid[r1, c]) in BLOCKING_CELLS_SHOT:
                    return False
            return True
        if c1 == c2:
            step = 1 if r2 > r1 else -1
            for r in range(r1 + step, r2, step):
                if int(self.grid[r, c1]) in BLOCKING_CELLS_SHOT:
                    return False
            return True
        return False

    def _move_towards(self, src: Position, dst: Position) -> int:
        dr = dst[0] - src[0]
        dc = dst[1] - src[1]
        prefer_vertical = abs(dr) >= abs(dc)

        candidates: list[int] = []
        if prefer_vertical:
            if dr < 0:
                candidates.append(MOVE_UP)
            elif dr > 0:
                candidates.append(MOVE_DOWN)
            if dc > 0:
                candidates.append(MOVE_RIGHT)
            elif dc < 0:
                candidates.append(MOVE_LEFT)
        else:
            if dc > 0:
                candidates.append(MOVE_RIGHT)
            elif dc < 0:
                candidates.append(MOVE_LEFT)
            if dr < 0:
                candidates.append(MOVE_UP)
            elif dr > 0:
                candidates.append(MOVE_DOWN)

        candidates += [MOVE_UP, MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT]
        for action in candidates:
            direction = self._action_to_dir(action)
            next_pos = self._next_position(src, direction)
            if self._can_move_to(next_pos, moving_unit="enemy"):
                return action
        return NOOP

    def render_rgb_array(self, cell_size: int = 36) -> NDArray[np.uint8]:
        assert self.grid is not None
        assert self.grid_template is not None
        assert self.castle_pos is not None

        height = self.grid_size * cell_size
        width = self.grid_template.shape[1] * cell_size
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        colors: dict[int, NDArray[np.uint8]] = {
            EMPTY: np.array([235, 235, 235], dtype=np.uint8),
            BRICK: np.array([156, 102, 31], dtype=np.uint8),
            STEEL: np.array([90, 90, 105], dtype=np.uint8),
            WATER: np.array([70, 130, 220], dtype=np.uint8),
            BUSH: np.array([90, 150, 90], dtype=np.uint8),
        }

        for r in range(self.grid_size):
            for c in range(self.grid_template.shape[1]):
                y0, y1 = r * cell_size, (r + 1) * cell_size
                x0, x1 = c * cell_size, (c + 1) * cell_size
                frame[y0:y1, x0:x1] = colors.get(int(self.grid[r, c]), np.array([235, 235, 235], dtype=np.uint8))
                frame[y0:y0 + 1, x0:x1] = 200
                frame[y1 - 1:y1, x0:x1] = 200
                frame[y0:y1, x0:x0 + 1] = 200
                frame[y0:y1, x1 - 1:x1] = 200

        cr, cc = self.castle_pos
        frame[cr * cell_size:(cr + 1) * cell_size, cc * cell_size:(cc + 1) * cell_size] = np.array([220, 60, 60], dtype=np.uint8)

        if self.player_alive and self.player_pos is not None:
            pr, pc = self.player_pos
            frame[pr * cell_size + 6:(pr + 1) * cell_size - 6, pc * cell_size + 6:(pc + 1) * cell_size - 6] = np.array([40, 170, 70], dtype=np.uint8)

        if self.enemy_alive and self.enemy_pos is not None:
            er, ec = self.enemy_pos
            frame[er * cell_size + 6:(er + 1) * cell_size - 6, ec * cell_size + 6:(ec + 1) * cell_size - 6] = np.array([180, 40, 140], dtype=np.uint8)

        if self.player_bullet is not None:
            br, bc = self.player_bullet["pos"]
            if self._in_bounds((br, bc)):
                frame[br * cell_size + cell_size // 3: br * cell_size + 2 * cell_size // 3,
                      bc * cell_size + cell_size // 3: bc * cell_size + 2 * cell_size // 3] = np.array([255, 210, 0], dtype=np.uint8)

        if self.enemy_bullet is not None:
            br, bc = self.enemy_bullet["pos"]
            if self._in_bounds((br, bc)):
                frame[br * cell_size + cell_size // 3: br * cell_size + 2 * cell_size // 3,
                      bc * cell_size + cell_size // 3: bc * cell_size + 2 * cell_size // 3] = np.array([255, 120, 0], dtype=np.uint8)

        return frame


LEVEL_3_MAP: str = """
E.#@#.E..
..#...#..
##..#..##
..#.@.#..
....#....
..#.@.#..
##..#..##
..#...#..
...PCP...
""".strip()


class LevelBattleCityEnv(SimpleBattleCityEnv):
    def __init__(self, level_map_str: str, max_steps: int = 160, seed: int = 42) -> None:
        self.level_map_str = level_map_str
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.action_space = SimpleDiscrete(N_ACTIONS)

        self.grid: Grid | None = None
        self.grid_template: Grid | None = None
        self.player_starts: list[Position] = []
        self.enemy_starts: list[Position] = []
        self.player_pos: Position | None = None
        self.enemy_pos: Position | None = None
        self.castle_pos: Position | None = None

        self.player_dir = UP
        self.enemy_dir = DOWN
        self.player_bullet = None
        self.enemy_bullet = None
        self.player_alive = True
        self.enemy_alive = True
        self.castle_alive = True
        self.step_count = 0
        self.score = 0.0
        self.shots_fired = 0
        self.hits_on_enemy = 0
        self.visited_cells: set[tuple[int, int]] = set()

        self._parse_level_map(level_map_str)

    def _parse_level_map(self, level_map_str: str) -> None:
        lines = [line.strip() for line in level_map_str.strip().splitlines() if line.strip()]
        if not lines:
            raise ValueError("Empty level map.")

        rows = len(lines)
        cols = len(lines[0])
        if any(len(line) != cols for line in lines):
            raise ValueError("All rows in the level map must have equal width.")

        self.grid_size = rows
        self.grid_template = np.zeros((rows, cols), dtype=np.int32)
        self.player_starts = []
        self.enemy_starts = []
        self.castle_pos = None

        char_to_cell: dict[str, int] = {".": EMPTY, "#": BRICK, "@": STEEL, "~": WATER}
        for r, line in enumerate(lines):
            for c, ch in enumerate(line):
                if ch in char_to_cell:
                    self.grid_template[r, c] = char_to_cell[ch]
                elif ch == "P":
                    self.player_starts.append([r, c])
                elif ch == "E":
                    self.enemy_starts.append([r, c])
                elif ch == "C":
                    self.castle_pos = [r, c]
                else:
                    raise ValueError(f"Unknown map symbol: {ch!r}")

        if not self.player_starts:
            raise ValueError("Level must contain at least one P.")
        if not self.enemy_starts:
            raise ValueError("Level must contain at least one E.")
        if self.castle_pos is None:
            raise ValueError("Level must contain one C.")

    def reset(self, seed: int | None = None, **_: Any) -> tuple[Observation, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        assert self.grid_template is not None
        self.grid = self.grid_template.copy()
        self.player_pos = list(self.player_starts[int(self.rng.integers(len(self.player_starts)))])
        self.enemy_pos = list(self.enemy_starts[int(self.rng.integers(len(self.enemy_starts)))])
        self.player_dir = UP
        self.enemy_dir = DOWN
        self.player_bullet = None
        self.enemy_bullet = None
        self.player_alive = True
        self.enemy_alive = True
        self.castle_alive = True
        self.step_count = 0
        self.score = 0.0
        self.shots_fired = 0
        self.hits_on_enemy = 0
        self.visited_cells = {tuple(self.player_pos)}

        obs = self._get_obs()
        info = self._get_info(win=False, trade=False)
        return obs, info


__all__ = [
    "NOOP",
    "MOVE_UP",
    "MOVE_RIGHT",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "SHOOT",
    "UP",
    "RIGHT",
    "DOWN",
    "LEFT",
    "N_ACTIONS",
    "LEVEL_3_MAP",
    "SimpleBattleCityEnv",
    "LevelBattleCityEnv",
    "SimpleDiscrete",
]
