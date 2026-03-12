"""Q-learning agent for Battle City."""

from collections import defaultdict
import numpy as np
import pickle


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration."""

    def __init__(self, n_actions=6, alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=50000):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

        # Training stats for observability
        self.td_errors = []
        self.q_deltas = []

    def act(self, state, greedy=False):
        """Choose action using epsilon-greedy policy."""
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = self.q_table[state]
        # Break ties randomly
        max_q = q_values.max()
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """Q-learning update: Q(s,a) += α [r + γ max Q(s',a') - Q(s,a)]."""
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state].max()

        td_error = target - current_q
        self.q_table[state][action] = current_q + self.alpha * td_error

        # Track stats
        self.td_errors.append(abs(td_error))
        self.q_deltas.append(abs(self.alpha * td_error))

    def decay_epsilon(self, episode):
        """Linear epsilon decay."""
        frac = min(episode / self.epsilon_decay_steps, 1.0)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def get_episode_stats(self):
        """Return and reset per-episode training stats."""
        stats = {
            "mean_td_error": np.mean(self.td_errors) if self.td_errors else 0.0,
            "mean_q_delta": np.mean(self.q_deltas) if self.q_deltas else 0.0,
        }
        self.td_errors.clear()
        self.q_deltas.clear()
        return stats

    def get_q_stats(self):
        """Global Q-table statistics."""
        if not self.q_table:
            return {"mean_q": 0, "max_q": 0, "std_q": 0, "n_states": 0}
        all_q = np.array([v for v in self.q_table.values()])
        return {
            "mean_q": float(all_q.mean()),
            "max_q": float(all_q.max()),
            "std_q": float(all_q.std()),
            "n_states": len(self.q_table),
        }

    def save(self, path):
        """Save Q-table to file."""
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path):
        """Load Q-table from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), data)
