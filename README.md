# Tabular RL for Battle City

Tabular Q-learning agent that learns to play a simplified [Battle City](https://en.wikipedia.org/wiki/Battle_City) (NES, 1985) on a discrete grid world.

## 1. Mathematical Formulation

### 1.1 Notation

| Symbol | Meaning |
|--------|---------|
| $\mathcal{S}$ | Finite set of states |
| $\mathcal{A}$ | Finite set of actions |
| $s_t \in \mathcal{S}$ | State at time step $t$ |
| $a_t \in \mathcal{A}$ | Action taken at time step $t$ |
| $r_t \in \mathbb{R}$ | Reward received after taking $a_t$ in $s_t$ |
| $\gamma \in [0, 1)$ | Discount factor |
| $\alpha \in (0, 1]$ | Learning rate |
| $\varepsilon \in [0, 1]$ | Exploration rate |
| $Q(s, a)$ | Action-value function — expected return from taking $a$ in $s$ |
| $\pi(s)$ | Policy — mapping from states to actions |
| $P(s' \mid s, a)$ | Transition probability |
| $R(s, a, s')$ | Reward function |

### 1.2 Markov Decision Process (MDP)

We model the game as a finite MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$.

**State** $s_t$ is a tuple:

$$s_t = (p_{\text{pos}},\; p_{\text{dir}},\; e_{\text{pos}},\; e_{\text{dir}},\; b) \in \mathcal{S}$$

where:
- $p_{\text{pos}} \in \{0, \ldots, N-1\}$ — player cell index on the grid ($N = 81$ for 9×9)
- $p_{\text{dir}} \in \{0, 1, 2, 3\}$ — player direction (up, right, down, left)
- $e_{\text{pos}} \in \{0, \ldots, N-1\}$ — enemy cell index
- $e_{\text{dir}} \in \{0, 1, 2, 3\}$ — enemy direction
- $b \in \{0, 1\}$ — whether a player bullet is in flight

**Action space:**

$$\mathcal{A} = \{\text{noop},\; \text{up},\; \text{right},\; \text{down},\; \text{left},\; \text{shoot}\}, \quad |\mathcal{A}| = 6$$

**State space size:**

$$|\mathcal{S}| = N \times 4 \times N \times 4 \times 2 = 81 \times 4 \times 81 \times 4 \times 2 = 209{,}952$$

In practice, only reachable states are visited. The trained agent discovers $\approx 77{,}000$ states.

### 1.3 Markov Property Justification

The Markov property requires that the future depends only on the current state, not on history:

$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)$$

Our state $s_t$ satisfies this because it captures all information needed to determine the next state:

1. **Tank positions and directions** — fully determine where each tank is and which way it faces. Movement is deterministic given position + direction.
2. **Bullet flag** — determines whether the player can shoot (at most 1 bullet in flight). Since bullets move 1 cell/step in a fixed direction and are destroyed on collision, their trajectory is deterministic given the grid.
3. **Grid** — brick walls can be destroyed, but the grid is implicitly determined by the sequence of bullet impacts, which are tracked through the game mechanics within each step.

The enemy AI introduces stochasticity (random direction changes with probability 0.3), which makes the transition function $P(s' \mid s, a)$ probabilistic — but it still depends only on the current state, not on history. This is the standard stochastic MDP setting.

> **Note on discretization:** The original Battle City has continuous pixel-level movement (tanks move 2px/tick on a 416×416 pixel field). We discretize to a 9×9 grid where tanks occupy exactly one cell and move one cell per step. This makes the state space finite and tractable for tabular methods, at the cost of losing sub-cell resolution.

### 1.4 Objective

We maximize the expected discounted return:

$$G_t = \sum_{k=0}^{\infty} \gamma^k \, r_{t+k+1}$$

The optimal action-value function satisfies the Bellman optimality equation:

$$Q^{\ast}(s, a) = \mathbb{E}\left[ r + \gamma \max_{a'} Q^{\ast}(s', a') \;\middle|\; s, a \right]$$

### 1.5 Q-Learning Update

Q-learning is an off-policy TD(0) method. After observing transition $(s_t, a_t, r_t, s_{t+1})$:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \underbrace{\left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]}_{\delta_t \;(\text{TD error})}$$

We track two key quantities during training:

- **TD error**: $|\delta_t| = |r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)|$ — analogous to loss in deep RL
- **Q-value delta**: $|\alpha \cdot \delta_t|$ — the actual magnitude of each update, analogous to gradient norm

Both should decrease as the Q-table converges.

### 1.6 Exploration: $\varepsilon$-Greedy Policy

$$\pi(a \mid s) = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q(s, a') \\ \frac{\varepsilon}{|\mathcal{A}|} & \text{otherwise} \end{cases}$$

$\varepsilon$ decays linearly from 1.0 to 0.01 over 80% of training episodes.

### 1.7 Reward Function

| Event | $r$ |
|-------|-----|
| Kill enemy | $+100$ |
| Player killed | $-100$ |
| Castle destroyed | $-200$ |
| Each timestep | $-0.1$ |

Episode terminates when player dies, castle is destroyed, enemy is killed, or 500 steps are reached.

### 1.8 Convergence

Q-learning converges to $Q^{\ast}$ under the following conditions (Watkins & Dayan, 1992):

1. All state-action pairs are visited infinitely often — guaranteed by $\varepsilon$-greedy with $\varepsilon > 0$
2. Learning rate satisfies $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$ — we use constant $\alpha = 0.1$ which works in practice for finite MDPs
3. The MDP has bounded rewards — our rewards are in $[-200, +100]$

## 2. Repository Structure

```
tanks-rl/
├── lib/
│   ├── env.py            # BattleCityEnv — gymnasium environment (9×9 grid)
│   ├── agent.py          # QLearningAgent — tabular Q-learning
│   ├── evaluate.py       # Reward-independent evaluation metrics
│   └── plots.py          # Training & eval plot generation
├── levels/
│   ├── 1.txt             # Open arena (easy)
│   ├── 2.txt             # Corridors and cover (medium)
│   └── 3.txt             # Dense obstacles (hard)
├── train.py              # Entry point: train agent, log metrics, generate plots
├── render.py             # Entry point: visualize trained agent with pygame
├── runs/                 # Auto-created per training run
│   └── <run_name>/       # e.g. "bold_curie"
│       ├── config.json   # Hyperparameter snapshot
│       ├── q_table.pkl   # Trained Q-table
│       ├── train_log.csv # Per-episode: reward, TD error, Q-delta, epsilon, ...
│       ├── eval_log.csv  # Periodic eval: win rate, accuracy, coverage, ...
│       └── plots/        # Generated PNG plots
└── requirements.txt
```

### Key files

| File | Purpose |
|------|---------|
| `lib/env.py` | Gymnasium environment. Loads a 9×9 level, runs game logic (movement, bullets, collisions, enemy AI), encodes state as a tuple, computes rewards. |
| `lib/agent.py` | Q-learning agent. Maintains a `defaultdict` Q-table, epsilon-greedy action selection, tracks TD error and Q-delta per update. |
| `lib/evaluate.py` | Runs 100 greedy episodes and computes reward-independent metrics: win rate, survival rate, castle survival, kill time, shot accuracy, exploration coverage. |
| `lib/plots.py` | Reads CSV logs, generates 11 plots: reward curve, win rate, Q-value stats (mean/max/std), Q-delta, TD error, epsilon, state coverage, episode length, eval dashboard. |
| `train.py` | Parses CLI args, creates a uniquely-named run directory, trains the agent, logs everything, runs periodic evals, saves Q-table and plots. |
| `render.py` | Loads a trained Q-table and plays episodes with pygame rendering. |

## 3. Evaluation Metrics

All metrics are reward-independent and comparable across runs:

| Metric | Description |
|--------|-------------|
| Win rate | Fraction of episodes where agent kills the enemy |
| Survival rate | Fraction of episodes where agent is alive at termination |
| Castle survival rate | Fraction of episodes where castle is intact |
| Avg episode length | Mean steps per episode (lower = more efficient) |
| Kill time | Mean steps to kill enemy (winning episodes only) |
| Shot accuracy | Kills / total shots fired |
| Exploration coverage | Fraction of walkable cells visited per episode |

## 4. Training Observability

Console output every N episodes:
```
Ep  5000 | WinRate: 0.29 | AvgReward: -105.7 | AvgLen:   42 | TDErr: 2.472 | QDelta: 0.247 | States: 36219 | ε: 0.876
```

Plots generated after training:
1. **Episode reward** — rolling average over 1000 episodes
2. **Win rate** — from periodic eval, should increase
3. **Q-value mean / max / std** — tracks value scale and detects divergence
4. **Q-value delta** — mean $|\alpha \cdot \delta_t|$ per episode (grad norm analog), should decrease
5. **TD error** — mean $|\delta_t|$ per episode (loss analog), should decrease
6. **Epsilon** — exploration rate decay curve
7. **State coverage** — unique states discovered over time
8. **Episode length** — should decrease as agent becomes more efficient
9. **Eval dashboard** — multi-panel plot of all eval metrics

## 5. Usage

### Install

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Train

```bash
python train.py                                    # defaults: 50K episodes, level 1
python train.py --episodes 100000 --level levels/2.txt --alpha 0.05
python train.py --run-name my_experiment           # custom run name
```

### Visualize

```bash
python render.py runs/<run_name>
python render.py runs/<run_name> --fps 3 --episodes 20
```

### Compare runs

Eval logs are in `runs/<name>/eval_log.csv` — compare final rows across runs.

## 6. Results

Training on level 1, 50K episodes:

| Metric | Value |
|--------|-------|
| Win rate | **96%** |
| Survival rate | 97% |
| Castle survival | 98% |
| Avg episode length | 24 steps |
| Shot accuracy | 27% |
| Q-table states | 77,105 |
