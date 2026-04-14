"""
HMM-based rat belief tracker.

State space: 64 cells (8x8 board), indexed as idx = y*8 + x.
Transition: known matrix T (given at game start).
Observations: (noise_type, estimated_manhattan_distance) each turn.

Key timing:
  - Between our turns, the rat moves TWICE (once before opponent's turn, once before ours).
  - We observe only on our own turns.
  - Opponent search result is visible to us (intermediate rat position).
"""

import numpy as np
from typing import Tuple, Optional

BOARD_SIZE = 8
N_CELLS = 64

# Noise probabilities per cell type: NOISE_TABLE[cell_type, noise_idx]
# Cell types: SPACE=0, PRIMED=1, CARPET=2, BLOCKED=3
# Noise types: SQUEAK=0, SCRATCH=1, SQUEAL=2
NOISE_TABLE = np.array([
    [0.70, 0.15, 0.15],  # SPACE
    [0.10, 0.80, 0.10],  # PRIMED
    [0.10, 0.10, 0.80],  # CARPET
    [0.50, 0.30, 0.20],  # BLOCKED
], dtype=np.float64)

# Distance error model: P(reported = actual + offset) for each offset
DIST_ERROR_OFFSETS = np.array([-1, 0, 1, 2], dtype=np.int32)
DIST_ERROR_PROBS   = np.array([0.12, 0.70, 0.12, 0.06], dtype=np.float64)

# Precompute cell coordinates (fixed for all games)
_CELL_XS = (np.arange(N_CELLS) % BOARD_SIZE).astype(np.int32)
_CELL_YS = (np.arange(N_CELLS) // BOARD_SIZE).astype(np.int32)


class RatBelief:
    """
    Bayesian belief over rat position, maintained as a 64-element probability vector.
    Indexing: idx = y * 8 + x  (matches game engine convention).
    """

    def __init__(self, T):
        """
        T: 64x64 transition matrix. T[i,j] = P(rat moves from cell i to cell j).
           May be a JAX array; will be converted to numpy float64.
        """
        self.T = np.array(T, dtype=np.float64)
        # Compute stationary distribution via power iteration
        self.stationary = self._compute_stationary()
        # Initial belief: stationary (rat had 1000 warm-up steps)
        self.belief = self.stationary.copy()

    # ------------------------------------------------------------------
    # Core HMM operations
    # ------------------------------------------------------------------

    def predict(self):
        """One-step prediction: advance belief through transition model."""
        self.belief = self.belief @ self.T
        self._normalize()

    def update(self, noise: int, distance: int,
               worker_pos: Tuple[int, int], board):
        """
        Update belief with an observation.

        Parameters
        ----------
        noise    : int  0=SQUEAK, 1=SCRATCH, 2=SQUEAL
        distance : int  estimated manhattan distance (possibly noisy)
        worker_pos: (x, y) of our worker
        board    : Board object (for cell-type lookup)
        """
        # 1. Noise likelihood from cell type
        cell_types = self._get_cell_types(board)           # (64,) int
        noise_lk = NOISE_TABLE[cell_types, int(noise)]     # (64,) float

        # 2. Distance likelihood
        dist_lk = self._distance_likelihood(worker_pos, distance)  # (64,) float

        # 3. Bayesian update
        self.belief *= noise_lk * dist_lk
        self._normalize(fallback_to_stationary=True)

    # ------------------------------------------------------------------
    # Event-driven updates
    # ------------------------------------------------------------------

    def zero_cell(self, cell: Tuple[int, int]):
        """Rat was NOT at `cell` → zero that cell and renormalise."""
        idx = cell[1] * BOARD_SIZE + cell[0]
        self.belief[idx] = 0.0
        self._normalize(fallback_to_stationary=True)

    def reset_to_stationary(self):
        """Rat was just caught and a new one spawned with 1000 warm-up steps."""
        self.belief = self.stationary.copy()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def best_search(self) -> Tuple[Tuple[int, int], float]:
        """
        Return (best_cell, expected_value) for a rat search.
        EV = 6 * P(rat_there) - 2.
        """
        best_idx = int(np.argmax(self.belief))
        ev = 6.0 * float(self.belief[best_idx]) - 2.0
        pos = (int(best_idx % BOARD_SIZE), int(best_idx // BOARD_SIZE))
        return pos, ev

    def top_k_searches(self, k: int = 5):
        """Return top-k [(cell_pos, ev)] sorted by EV."""
        evs = 6.0 * self.belief - 2.0
        top_idx = np.argsort(evs)[::-1][:k]
        return [
            ((int(i % BOARD_SIZE), int(i // BOARD_SIZE)), float(evs[i]))
            for i in top_idx
        ]

    def get_belief_2d(self) -> np.ndarray:
        """Return (8, 8) belief array (for neural network input)."""
        return self.belief.reshape(BOARD_SIZE, BOARD_SIZE).astype(np.float32)

    def max_prob(self) -> float:
        return float(np.max(self.belief))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, fallback_to_stationary: bool = False):
        s = self.belief.sum()
        if s > 1e-300:
            self.belief /= s
        elif fallback_to_stationary:
            self.belief = self.stationary.copy()
        else:
            self.belief = np.ones(N_CELLS, dtype=np.float64) / N_CELLS

    def _compute_stationary(self) -> np.ndarray:
        """Power iteration to find the stationary distribution of T."""
        v = np.ones(N_CELLS, dtype=np.float64) / N_CELLS
        for _ in range(500):
            v_new = v @ self.T
            v_new /= v_new.sum()
            if np.max(np.abs(v_new - v)) < 1e-12:
                break
            v = v_new
        return v_new

    def _get_cell_types(self, board) -> np.ndarray:
        """
        Extract cell-type array (0=SPACE, 1=PRIMED, 2=CARPET, 3=BLOCKED)
        from the board's bitmasks.  O(64) bit tests.
        """
        types = np.zeros(N_CELLS, dtype=np.int32)
        pm = board._primed_mask
        cm = board._carpet_mask
        bm = board._blocked_mask
        for i in range(N_CELLS):
            bit = 1 << i
            if pm & bit:
                types[i] = 1
            elif cm & bit:
                types[i] = 2
            elif bm & bit:
                types[i] = 3
        return types

    def _distance_likelihood(self, worker_pos: Tuple[int, int],
                              observed_dist: int) -> np.ndarray:
        """
        P(observed_dist | rat_at_cell_i) for all 64 cells.

        The reported distance is  max(0, actual_dist + error_offset)
        where error_offset is drawn from DIST_ERROR_PROBS.
        """
        wx, wy = worker_pos
        actual = np.abs(_CELL_XS - wx) + np.abs(_CELL_YS - wy)  # (64,)
        lk = np.zeros(N_CELLS, dtype=np.float64)
        for offset, prob in zip(DIST_ERROR_OFFSETS, DIST_ERROR_PROBS):
            reported = np.maximum(0, actual + int(offset))
            lk += prob * (reported == int(observed_dist)).astype(np.float64)
        return lk

    # ------------------------------------------------------------------
    # Copying (for MCTS tree nodes / simulator)
    # ------------------------------------------------------------------

    def copy(self) -> "RatBelief":
        new = RatBelief.__new__(RatBelief)
        new.T          = self.T            # shared reference (read-only)
        new.stationary = self.stationary   # shared reference
        new.belief     = self.belief.copy()
        return new
