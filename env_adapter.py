from dataclasses import dataclass

@dataclass(frozen=True)
class StateIndexer:
    n_cells: int          # например, 9x9 -> 81
    n_dirs: int = 4
    n_bullet: int = 2

    @property
    def n_states(self) -> int:
        return self.n_cells * self.n_dirs * self.n_cells * self.n_dirs * self.n_bullet

    def encode(self, s_t) -> int:
        """
        s_t = (p_idx, p_dir, e_idx, e_dir, bullet_alive)
        """
        p_idx, p_dir, e_idx, e_dir, bullet_alive = s_t

        idx = int(p_idx)
        idx = idx * self.n_dirs + int(p_dir)
        idx = idx * self.n_cells + int(e_idx)
        idx = idx * self.n_dirs + int(e_dir)
        idx = idx * self.n_bullet + int(bullet_alive)
        return idx

    def decode(self, idx: int):
        bullet_alive = idx % self.n_bullet
        idx //= self.n_bullet

        e_dir = idx % self.n_dirs
        idx //= self.n_dirs

        e_idx = idx % self.n_cells
        idx //= self.n_cells

        p_dir = idx % self.n_dirs
        idx //= self.n_dirs

        p_idx = idx
        return (p_idx, p_dir, e_idx, e_dir, bullet_alive)

def to_discrete_state(obs, env=None):
    """
    Должна вернуть дискретное состояние в форме:
        s_t = (p_idx, p_dir, e_idx, e_dir, bullet_alive)

    По умолчанию предполагается, что obs уже такой tuple.
    """
    return obs

class IndexedEnv:
    def __init__(self, env, state_indexer: StateIndexer):
        self.env = env
        self.state_indexer = state_indexer
        self.action_space = env.action_space

    def reset(self, **kwargs):
        s_t, info = self.env.reset(**kwargs)
        return self.state_indexer.encode(s_t), info

    def step(self, a_t: int):
        s_tp1, r_t, terminated, truncated, info = self.env.step(a_t)
        return self.state_indexer.encode(s_tp1), r_t, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)