"""Battle City grid-world environment for tabular RL."""

import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Directions
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIR_DELTA = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}

# Actions
NOOP, MOVE_UP, MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT, SHOOT = range(6)

# Cell types
EMPTY, BRICK, STEEL, WATER = 0, 1, 2, 3


class BattleCityEnv(gym.Env):
    """
    Simplified Battle City on a 9x9 grid.

    State: (player_pos_idx, player_dir, enemy_pos_idx, enemy_dir, bullet_alive)
    Actions: NOOP, UP, RIGHT, DOWN, LEFT, SHOOT
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}

    def __init__(self, level_path=None, max_steps=500, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps

        # Load level
        if level_path is None:
            level_path = os.path.join(os.path.dirname(__file__), "..", "levels", "1.txt")
        self._load_level(level_path)

        self.action_space = spaces.Discrete(6)
        # Observation: flat tuple encoded as integer (see get_state)
        self.observation_space = spaces.Discrete(self._max_state_size())

        # Pygame surface (lazy init)
        self._screen = None
        self._clock = None

    # ── Level loading ──────────────────────────────────────────────

    def _load_level(self, path):
        """Parse level file into grid, find spawn positions."""
        with open(path) as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]

        self.rows = len(lines)
        self.cols = max(len(line) for line in lines)

        self.grid_template = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.player_start = None
        self.enemy_starts = []
        self.castle_pos = None

        char_map = {"#": BRICK, "@": STEEL, "~": WATER}

        for r, line in enumerate(lines):
            for c, ch in enumerate(line):
                if ch in char_map:
                    self.grid_template[r, c] = char_map[ch]
                elif ch == "P":
                    self.player_start = (r, c)
                elif ch == "E":
                    self.enemy_starts.append((r, c))
                elif ch == "C":
                    self.castle_pos = (r, c)
                # else: EMPTY

        # Walkable cells: all cells that are not steel or water in the template
        # (bricks can be destroyed, so they start as obstacles but can become walkable)
        self.all_cells = [
            (r, c) for r in range(self.rows) for c in range(self.cols)
        ]
        self.cell_to_idx = {cell: i for i, cell in enumerate(self.all_cells)}
        self.n_cells = len(self.all_cells)

    def _max_state_size(self):
        # player_pos * 4 dirs * enemy_pos * 4 dirs * 2 bullet states
        return self.n_cells * 4 * self.n_cells * 4 * 2

    # ── Reset / Step ───────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = self.grid_template.copy()
        self.step_count = 0

        # Player
        self.player_pos = self.player_start
        self.player_dir = UP
        self.player_alive = True

        # Enemy (pick first spawn point)
        self.enemy_pos = self.enemy_starts[0] if self.enemy_starts else (0, 0)
        self.enemy_dir = DOWN
        self.enemy_alive = True

        # Bullets: (row, col, direction, owner)  owner: 'player' or 'enemy'
        self.bullets = []

        # Castle
        self.castle_alive = True

        # Stats for evaluation
        self.stats = {"shots_fired": 0, "cells_visited": {self.player_pos}}

        return self._get_state(), {}

    def step(self, action):
        reward = -0.1  # time penalty
        terminated = False
        truncated = False
        self.step_count += 1

        # 1. Player action
        if self.player_alive:
            self._do_player_action(action)

        # 2. Enemy action
        if self.enemy_alive:
            self._do_enemy_action()

        # 3. Move all bullets and check collisions
        reward += self._update_bullets()

        # 4. Check terminal conditions
        if not self.player_alive:
            reward -= 100
            terminated = True
        elif not self.castle_alive:
            reward -= 200
            terminated = True
        elif not self.enemy_alive:
            reward += 100
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        return self._get_state(), reward, terminated, truncated, self._get_info()

    # ── Player actions ─────────────────────────────────────────────

    def _do_player_action(self, action):
        if action == NOOP:
            return
        elif action == SHOOT:
            self._shoot("player")
        elif action in (MOVE_UP, MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT):
            direction = action - 1  # MOVE_UP=1 -> UP=0, etc.
            self.player_dir = direction
            new_pos = self._move(self.player_pos, direction)
            if new_pos and self._can_walk(new_pos) and new_pos != self.enemy_pos:
                self.player_pos = new_pos
                self.stats["cells_visited"].add(new_pos)

    def _shoot(self, owner):
        if owner == "player":
            # Only 1 bullet at a time
            if any(b[3] == "player" for b in self.bullets):
                return
            pos = self.player_pos
            d = self.player_dir
            self.stats["shots_fired"] += 1
        else:
            if any(b[3] == "enemy" for b in self.bullets):
                return
            pos = self.enemy_pos
            d = self.enemy_dir

        # Spawn bullet one cell ahead
        br, bc = pos[0] + DIR_DELTA[d][0], pos[1] + DIR_DELTA[d][1]
        if self._in_bounds(br, bc):
            self.bullets.append((br, bc, d, owner))

    # ── Enemy AI ───────────────────────────────────────────────────

    def _do_enemy_action(self):
        # Simple AI: if aligned with player, shoot. Otherwise, random move.
        pr, pc = self.player_pos
        er, ec = self.enemy_pos

        # Check alignment for shooting
        if er == pr and ec < pc and self.enemy_dir == RIGHT:
            self._shoot("enemy")
            return
        if er == pr and ec > pc and self.enemy_dir == LEFT:
            self._shoot("enemy")
            return
        if ec == pc and er < pr and self.enemy_dir == DOWN:
            self._shoot("enemy")
            return
        if ec == pc and er > pr and self.enemy_dir == UP:
            self._shoot("enemy")
            return

        # If aligned but not facing right way, turn toward player
        if er == pr:
            self.enemy_dir = RIGHT if ec < pc else LEFT
            if self.np_random.random() < 0.3:
                self._shoot("enemy")
            return
        if ec == pc:
            self.enemy_dir = DOWN if er < pr else UP
            if self.np_random.random() < 0.3:
                self._shoot("enemy")
            return

        # Random movement
        if self.np_random.random() < 0.3:
            self.enemy_dir = self.np_random.integers(4)

        new_pos = self._move(self.enemy_pos, self.enemy_dir)
        if new_pos and self._can_walk(new_pos) and new_pos != self.player_pos:
            self.enemy_pos = new_pos

    # ── Bullet update ──────────────────────────────────────────────

    def _update_bullets(self):
        """Move all bullets, handle collisions. Returns reward delta."""
        reward = 0.0
        new_bullets = []

        for br, bc, d, owner in self.bullets:
            # Move bullet
            nr, nc = br + DIR_DELTA[d][0], bc + DIR_DELTA[d][1]

            # Out of bounds
            if not self._in_bounds(nr, nc):
                continue

            # Hit wall
            cell = self.grid[nr, nc]
            if cell == STEEL:
                continue  # bullet destroyed, wall stays
            if cell == BRICK:
                self.grid[nr, nc] = EMPTY  # destroy brick
                continue
            if cell == WATER:
                continue  # bullet stops

            # Hit player
            if (nr, nc) == self.player_pos and owner == "enemy" and self.player_alive:
                self.player_alive = False
                continue

            # Hit enemy
            if (nr, nc) == self.enemy_pos and owner == "player" and self.enemy_alive:
                self.enemy_alive = False
                continue

            # Hit castle
            if (nr, nc) == self.castle_pos and self.castle_alive:
                self.castle_alive = False
                continue

            # Bullet survives
            new_bullets.append((nr, nc, d, owner))

        self.bullets = new_bullets
        return reward

    # ── Helpers ─────────────────────────────────────────────────────

    def _move(self, pos, direction):
        """Return new position after moving in direction, or None if out of bounds."""
        dr, dc = DIR_DELTA[direction]
        nr, nc = pos[0] + dr, pos[1] + dc
        if self._in_bounds(nr, nc):
            return (nr, nc)
        return None

    def _in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _can_walk(self, pos):
        """Check if a cell is walkable (not steel, water, or intact brick)."""
        cell = self.grid[pos[0], pos[1]]
        return cell == EMPTY

    # ── State encoding ─────────────────────────────────────────────

    def _get_state(self):
        """Encode state as a tuple for Q-table key."""
        p_idx = self.cell_to_idx.get(self.player_pos, 0)
        e_idx = self.cell_to_idx.get(self.enemy_pos, 0) if self.enemy_alive else 0
        bullet_alive = int(any(b[3] == "player" for b in self.bullets))
        return (p_idx, self.player_dir, e_idx, self.enemy_dir, bullet_alive)

    def _get_info(self):
        return {
            "win": not self.enemy_alive,
            "player_alive": self.player_alive,
            "castle_alive": self.castle_alive,
            "steps": self.step_count,
            "shots_fired": self.stats["shots_fired"],
            "cells_visited": len(self.stats["cells_visited"]),
        }

    # ── Rendering ──────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            return self._render_pygame()

    def _render_ansi(self):
        symbols = {EMPTY: ".", BRICK: "#", STEEL: "@", WATER: "~"}
        dir_symbols = {UP: "^", RIGHT: ">", DOWN: "v", LEFT: "<"}
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) == self.player_pos and self.player_alive:
                    row.append(dir_symbols[self.player_dir])
                elif (r, c) == self.enemy_pos and self.enemy_alive:
                    row.append("X")
                elif (r, c) == self.castle_pos and self.castle_alive:
                    row.append("C")
                elif any(b[0] == r and b[1] == c for b in self.bullets):
                    row.append("*")
                else:
                    row.append(symbols.get(self.grid[r, c], "."))
            lines.append(" ".join(row))
        return "\n".join(lines)

    def _render_pygame(self):
        import pygame

        cell_size = 60
        width = self.cols * cell_size
        height = self.rows * cell_size

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Battle City RL")
            self._clock = pygame.time.Clock()

        colors = {
            EMPTY: (40, 40, 40),
            BRICK: (180, 100, 50),
            STEEL: (160, 160, 160),
            WATER: (30, 100, 200),
        }

        self._screen.fill((0, 0, 0))

        for r in range(self.rows):
            for c in range(self.cols):
                rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
                pygame.draw.rect(self._screen, colors[self.grid[r, c]], rect)
                pygame.draw.rect(self._screen, (20, 20, 20), rect, 1)

        # Castle
        if self.castle_alive and self.castle_pos:
            cr, cc = self.castle_pos
            rect = pygame.Rect(cc * cell_size + 5, cr * cell_size + 5, cell_size - 10, cell_size - 10)
            pygame.draw.rect(self._screen, (255, 215, 0), rect)

        # Enemy
        if self.enemy_alive:
            er, ec = self.enemy_pos
            cx = ec * cell_size + cell_size // 2
            cy = er * cell_size + cell_size // 2
            pygame.draw.circle(self._screen, (220, 50, 50), (cx, cy), cell_size // 3)
            self._draw_direction(cx, cy, self.enemy_dir, cell_size, (255, 100, 100))

        # Player
        if self.player_alive:
            pr, pc = self.player_pos
            cx = pc * cell_size + cell_size // 2
            cy = pr * cell_size + cell_size // 2
            pygame.draw.circle(self._screen, (50, 180, 50), (cx, cy), cell_size // 3)
            self._draw_direction(cx, cy, self.player_dir, cell_size, (100, 255, 100))

        # Bullets
        for br, bc, d, owner in self.bullets:
            bx = bc * cell_size + cell_size // 2
            by = br * cell_size + cell_size // 2
            color = (200, 200, 50) if owner == "player" else (255, 100, 100)
            pygame.draw.circle(self._screen, color, (bx, by), 5)

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._screen = None

    def _draw_direction(self, cx, cy, direction, cell_size, color):
        """Draw a small line showing tank direction."""
        import pygame
        length = cell_size // 3
        dx, dy = DIR_DELTA[direction]
        end = (cx + dy * length, cy + dx * length)  # note: dx is row delta, dy is col delta
        pygame.draw.line(self._screen, color, (cx, cy), end, 3)

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
