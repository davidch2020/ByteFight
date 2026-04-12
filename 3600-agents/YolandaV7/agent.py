from collections.abc import Callable
from typing import List, Tuple, Optional
import random
import time
import numpy as np

from game import board, move, enums, rat
from game.move import Move

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAT_FIND_PTS = 4
RAT_MISS_PTS = -2
SEARCH_PROB_FLOOR = 0.34
SEARCH_EV_FLOOR = 0.5

TIME_HARD_FLOOR = 2.5
MAX_DEPTH = 7
MIN_DEPTH = 3

# Version 7

class PlayerAgent:
    """
    Expectiminimax agent targeting Carrie-level play.

    Architecture
    ------------
    1.  HMM belief tracking for the rat (predict + sensor update each turn).
    2.  Per-turn precomputation of a *cell potential map*: for every cell on
        the board, estimate how many carpet points it contributes to (if
        already primed) or could contribute to (if primed in the future),
        discounted by distance from the player.  This is what Carrie's
        "potential of each cell and its distance from the bot" refers to.
    3.  Iterative-deepening expectiminimax with alpha-beta pruning.
        - At max nodes the best rat-search move is a candidate alongside
          movement moves, evaluated as a flat EV from the current belief.
        - The heuristic uses the precomputed potential map for O(local)
          evaluation instead of scanning the whole board at every leaf.
    4.  Move ordering: carpet (by points) > prime > plain.
    """

    def __init__(self, board_obj, transition_matrix=None, time_left: Callable = None):
        self.T = np.array(transition_matrix, dtype=np.float64)
        self.SZ = enums.BOARD_SIZE  # 8

        # Rat spawn prior: placed at (0,0), 1000 free steps.
        self.initial_belief = np.zeros(64, dtype=np.float64)
        self.initial_belief[0] = 1.0
        for _ in range(1000):
            self.initial_belief = self.initial_belief @ self.T
        self.belief = self.initial_belief.copy()

        # Fast carpet-points lookup
        self._cpt = enums.CARPET_POINTS_TABLE

        # Per-turn caches (set in _precompute)
        self._cell_pot = None        # (SZ, SZ) float — cell potential map
        self._is_primed = None       # (SZ, SZ) bool grid
        self._grid = None            # (SZ, SZ) CellType grid

        self.nodes_visited = 0

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _idx(self, loc):
        return loc[1] * self.SZ + loc[0]

    def _loc(self, idx):
        return (idx % self.SZ, idx // self.SZ)

    def _normalize(self, b):
        s = b.sum()
        return b / s if s > 0 else self.initial_belief.copy()

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------
    def commentate(self):
        return ""

    def play(self, b: board.Board, sensor_data: Tuple, time_left: Callable):
        self.nodes_visited = 0
        t0 = time.time()

        # 1. Belief update
        self._update_belief(b, sensor_data)

        # 2. Precompute spatial data for this turn
        self._precompute(b)

        # 3. Iterative deepening search
        best = self._id_search(b, time_left, t0)
        if best is not None:
            return best

        return random.choice(b.get_valid_moves())

    # ------------------------------------------------------------------
    # HMM belief
    # ------------------------------------------------------------------
    def _update_belief(self, b: board.Board, sd: Tuple):
        ol, of_ = b.opponent_search
        pl, pf = b.player_search

        if pf:
            self.belief = self.initial_belief.copy()
        elif pl is not None:
            self.belief[self._idx(pl)] = 0.0
            self.belief = self._normalize(self.belief)

        if of_:
            self.belief = self.initial_belief.copy()
        elif ol is not None:
            self.belief[self._idx(ol)] = 0.0
            self.belief = self._normalize(self.belief)

        # Predict
        self.belief = self.belief @ self.T

        # Sensor update
        wx, wy = b.player_worker.get_location()
        for i in range(64):
            cx, cy = i % self.SZ, i // self.SZ
            cell = b.get_cell((cx, cy))
            np_ = rat.NOISE_PROBS[cell][sd[0]]
            md = abs(wx - cx) + abs(wy - cy)
            diff = sd[1] - md
            dp = rat.DISTANCE_ERROR_PROBS[diff + 1] if -1 <= diff <= 2 else 0.0
            self.belief[i] *= np_ * dp

        self.belief = self._normalize(self.belief)

    # ------------------------------------------------------------------
    # Precompute spatial data (once per turn, before search)
    # ------------------------------------------------------------------
    def _precompute(self, b: board.Board):
        """
        Build the cell potential map — Carrie's core advantage.

        For each cell we compute how many carpet points it participates in
        (or would participate in if primed), considering contiguous primed
        neighbours in all four cardinal directions.

        The potential is discounted by Manhattan distance from our worker,
        so nearby cells are more valuable.
        """
        SZ = self.SZ
        pp = b.player_worker.get_location()
        px, py = pp

        # Snapshot the board into fast-access arrays
        is_p = [[False]*SZ for _ in range(SZ)]
        for y in range(SZ):
            for x in range(SZ):
                if b.get_cell((x, y)) == enums.Cell.PRIMED:
                    is_p[y][x] = True
        self._is_primed = is_p

        # Precompute run lengths in each direction.
        # run_L[y][x] = # of contiguous primed cells to the left of (x,y),
        #               NOT including (x,y) itself.
        rL = [[0]*SZ for _ in range(SZ)]
        rR = [[0]*SZ for _ in range(SZ)]
        rU = [[0]*SZ for _ in range(SZ)]
        rD = [[0]*SZ for _ in range(SZ)]

        for y in range(SZ):
            for x in range(1, SZ):
                if is_p[y][x-1]:
                    rL[y][x] = rL[y][x-1] + 1
            for x in range(SZ-2, -1, -1):
                if is_p[y][x+1]:
                    rR[y][x] = rR[y][x+1] + 1
        for x in range(SZ):
            for y in range(1, SZ):
                if is_p[y-1][x]:
                    rU[y][x] = rU[y-1][x] + 1
            for y in range(SZ-2, -1, -1):
                if is_p[y+1][x]:
                    rD[y][x] = rD[y+1][x] + 1

        # Build potential map
        pot = np.zeros((SZ, SZ), dtype=np.float64)
        cpt = self._cpt

        for y in range(SZ):
            for x in range(SZ):
                ct = b.get_cell((x, y))
                d = abs(px - x) + abs(py - y)
                discount = 1.0 / (1.0 + 0.3 * d)

                if ct == enums.Cell.PRIMED:
                    # Best run through this cell
                    h = 1 + rL[y][x] + rR[y][x]
                    v = 1 + rU[y][x] + rD[y][x]
                    best = min(max(h, v), 7)
                    pts = cpt[best] if best >= 2 else 0
                    pot[y, x] = max(pts, 0) * discount

                elif ct == enums.Cell.SPACE:
                    # Hypothetical: if we primed this cell, what run would it join?
                    h = 1 + rL[y][x] + rR[y][x]
                    v = 1 + rU[y][x] + rD[y][x]
                    hp = cpt[min(h, 7)] if h >= 2 else 0
                    vp = cpt[min(v, 7)] if v >= 2 else 0
                    best_p = max(hp, vp, 0)
                    # Even isolated cells have small future value
                    if best_p == 0:
                        best_p = 0.25
                    pot[y, x] = best_p * discount

                # CARPET and BLOCKED → 0

        self._cell_pot = pot
        self._rL = rL
        self._rR = rR
        self._rU = rU
        self._rD = rD

    # ------------------------------------------------------------------
    # Iterative deepening
    # ------------------------------------------------------------------
    def _id_search(self, b, time_left, t0):
        best_move = None
        cands = b.get_valid_moves()
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]

        root_pts = b.player_worker.get_points()

        for depth in range(MIN_DEPTH, MAX_DEPTH + 1):
            if time_left() < TIME_HARD_FLOOR + 1.5:
                break

            it0 = time.time()
            mv, sc = self._root(b, cands, root_pts, depth, time_left)
            it_dur = time.time() - it0

            if mv is not None:
                best_move = mv

            rem = time_left()
            turns = max(b.player_worker.turns_left, 1)
            budget = (rem - TIME_HARD_FLOOR) / turns

            if rem < TIME_HARD_FLOOR + it_dur * 6:
                break
            if (time.time() - t0) > budget * 1.5 and depth >= MIN_DEPTH:
                break

        return best_move

    # ------------------------------------------------------------------
    # Root search
    # ------------------------------------------------------------------
    def _root(self, b, cands, root_pts, depth, time_left):
        ordered = self._order(cands)
        best_move = None
        best_sc = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        # Rat search candidate
        sm, sev = self._rat_search_ev()
        if sm is not None and sev > best_sc:
            best_sc = sev
            best_move = sm
            alpha = max(alpha, best_sc)

        for m in ordered:
            if time_left() < TIME_HARD_FLOOR:
                break
            nb = b.forecast_move(m)
            if nb is None:
                continue
            nb.reverse_perspective()
            sc = self._emm(nb, alpha, beta, root_pts, depth - 1, False, time_left)
            if sc > best_sc:
                best_sc = sc
                best_move = m
            alpha = max(alpha, best_sc)

        return best_move, best_sc

    def _rat_search_ev(self):
        idx = int(np.argmax(self.belief))
        p = self.belief[idx]
        ev = RAT_FIND_PTS * p + RAT_MISS_PTS * (1.0 - p)
        if p >= SEARCH_PROB_FLOOR and ev >= SEARCH_EV_FLOOR:
            return Move.search(self._loc(idx)), ev
        return None, float('-inf')

    # ------------------------------------------------------------------
    # Expectiminimax
    # ------------------------------------------------------------------
    def _emm(self, b, alpha, beta, root_pts, depth, maximizing, time_left):
        self.nodes_visited += 1

        if depth <= 0 or time_left() < TIME_HARD_FLOOR:
            if not maximizing:
                b.reverse_perspective()
            return self._eval(b, root_pts)

        moves = b.get_valid_moves()
        if not moves:
            if not maximizing:
                b.reverse_perspective()
            return self._eval(b, root_pts)

        ordered = self._order(moves)

        if maximizing:
            best = float('-inf')

            # Rat-search chance node at max nodes
            idx = int(np.argmax(self.belief))
            p = self.belief[idx]
            sev = RAT_FIND_PTS * p + RAT_MISS_PTS * (1.0 - p)
            if p >= SEARCH_PROB_FLOOR and sev >= SEARCH_EV_FLOOR:
                # Search doesn't change the board. The value is:
                # the expected points from the search (sev) plus
                # the continuation value where the opponent plays next
                # on the unchanged board.
                # To avoid the cost of a recursive call for the search
                # branch (which would be on the same board), we use the
                # raw sev as the search value. This is a sound approximation
                # because the board state is unchanged — only the score shifts.
                best = max(best, sev)
                if best >= beta:
                    return best
                alpha = max(alpha, best)

            for m in ordered:
                if time_left() < TIME_HARD_FLOOR:
                    break
                nb = b.forecast_move(m)
                if nb is None:
                    continue
                nb.reverse_perspective()
                val = self._emm(nb, alpha, beta, root_pts, depth - 1, False, time_left)
                best = max(best, val)
                if best >= beta:
                    return best
                alpha = max(alpha, best)
            return best
        else:
            best = float('inf')
            for m in ordered:
                if time_left() < TIME_HARD_FLOOR:
                    break
                nb = b.forecast_move(m)
                if nb is None:
                    continue
                nb.reverse_perspective()
                val = self._emm(nb, alpha, beta, root_pts, depth - 1, True, time_left)
                best = min(best, val)
                if best <= alpha:
                    return best
                beta = min(beta, best)
            return best

    # ------------------------------------------------------------------
    # Move ordering
    # ------------------------------------------------------------------
    def _order(self, moves):
        def key(m):
            if m.move_type == enums.MoveType.CARPET:
                return (3, self._cpt.get(m.roll_length, 21))
            if m.move_type == enums.MoveType.PRIME:
                # Use precomputed cell potential to order primes.
                # A prime move departs from the current cell (priming it)
                # and lands on the destination. We want to prime cells
                # that extend long runs, so score by the destination's
                # potential — but the primed cell is actually the ORIGIN,
                # not the destination. Since we don't have origin info on
                # the Move object, use a flat priority. The tree will sort
                # out the best prime.
                return (2, 0)
            if m.move_type == enums.MoveType.PLAIN:
                return (1, 0)
            return (0, 0)
        return sorted(moves, key=key, reverse=True)

    # ------------------------------------------------------------------
    # Heuristic evaluation
    # ------------------------------------------------------------------
    def _eval(self, b: board.Board, root_pts: int) -> float:
        pp = b.player_worker.get_points()
        op = b.opponent_worker.get_points()
        ploc = b.player_worker.get_location()
        oloc = b.opponent_worker.get_location()
        turns = b.player_worker.turns_left

        score = 0.0

        # --- 1. Point margin (most important) ---
        margin = pp - op
        score += margin

        # --- 2. Available carpet + mobility from current moves ---
        moves = b.get_valid_moves()
        best_carpet = 0.0
        carpet_sum = 0.0
        n_prime = 0
        n_plain = 0
        n_carpet = 0

        for m in moves:
            if m.move_type == enums.MoveType.CARPET:
                pts = self._cpt[m.roll_length]
                if pts > best_carpet:
                    best_carpet = pts
                if pts > 0:
                    carpet_sum += pts
                    n_carpet += 1
            elif m.move_type == enums.MoveType.PRIME:
                n_prime += 1
            elif m.move_type == enums.MoveType.PLAIN:
                n_plain += 1

        score += best_carpet * 0.45
        score += carpet_sum * 0.06
        score += n_carpet * 0.10
        score += n_prime * 0.25
        score += n_plain * 0.04

        # --- 3. Cell potential (Carrie-style spatial heuristic) ---
        # The precomputed potential map is from the root board state, so it's
        # slightly stale at deeper leaves. But the board changes only a few
        # cells per ply, so it's a good approximation. We use it as a light
        # directional signal, not a dominant term.
        if self._cell_pot is not None:
            px, py = ploc
            SZ = self.SZ
            # Player's local potential (within Manhattan distance 4)
            my_pot = 0.0
            for dy in range(-4, 5):
                ny = py + dy
                if ny < 0 or ny >= SZ:
                    continue
                ady = abs(dy)
                for dx in range(-(4 - ady), 5 - ady):
                    nx = px + dx
                    if nx < 0 or nx >= SZ:
                        continue
                    my_pot += self._cell_pot[ny, nx]
            score += my_pot * 0.05

        # --- 4. Opponent direct carpet threat ---
        # Compute from the ACTUAL leaf board, not the stale cache.
        opp_threat = self._opp_threat_live(b, oloc)
        score -= opp_threat * 0.22

        # --- 5. Endgame amplification ---
        if turns <= 10:
            u = (11 - turns) / 10.0
            score += margin * 0.2 * u
            score += best_carpet * 0.35 * u

        # --- 6. Rat search opportunity ---
        rp = float(np.max(self.belief))
        rev = RAT_FIND_PTS * rp + RAT_MISS_PTS * (1.0 - rp)
        if rev > 0:
            score += rev * 0.12

        return score

    def _opp_threat_live(self, b: board.Board, oloc: Tuple[int, int]) -> float:
        """
        Max carpet run the opponent could roll right now.
        Reads directly from the board (not stale cache) for accuracy.
        """
        ox, oy = oloc
        best = 0.0
        SZ = self.SZ
        cpt = self._cpt
        PRIMED = enums.Cell.PRIMED
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = ox + dx, oy + dy
            run = 0
            while 0 <= nx < SZ and 0 <= ny < SZ and b.get_cell((nx, ny)) == PRIMED:
                run += 1
                nx += dx
                ny += dy
            if run >= 2:
                pts = cpt[min(run, 7)]
                if pts > best:
                    best = pts
        return best