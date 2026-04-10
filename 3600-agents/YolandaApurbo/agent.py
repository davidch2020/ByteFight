from collections.abc import Callable
from typing import List, Tuple, Optional
import random
import time
import numpy as np

from game import board, move, enums, rat
from game.move import Move
from game.enums import loc_after_direction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAT_FIND_PTS = 4
RAT_MISS_PTS = -2
SEARCH_PROB_FLOOR = 0.34
SEARCH_EV_FLOOR = 0.5

TIME_HARD_FLOOR = 2.5
MAX_DEPTH = 9
MIN_DEPTH = 3

# Carpet points (avoid dict lookup in hot loop)
CPT = (0, -1, 2, 4, 6, 10, 15, 21)  # index by roll_length, 0 unused


class PlayerAgent:
    """
    Built on V4's proven core — clean minimax/alpha-beta with flat-EV rat search.

    Targeted additions over V4 (each one justified by what it costs vs gains):

    1. Deeper search (MAX_DEPTH=9) with aggressive time usage.
       V4 was capped at 7 and used conservative time budgets. With 4 minutes
       total and ~40 turns, we can afford 6s/turn average.

    2. Full chance-node expansion for rat search AT ROOT ONLY.
       Properly evaluates hit/miss branches where it matters most (the actual
       decision), zero cost inside the tree.

    3. Dynamic search thresholds based on score margin.
       When winning big, be conservative. When losing, take risks.

    4. Precomputed belief propagation (no matrix multiply in tree).
       belief @ T is expensive. Precompute once.

    5. Cell potential used for prime ordering at root.
       Help alpha-beta find the best prime first.

    Everything else stays V4-simple. No TT, no killers, no history table,
    no quiescence, no Zobrist — the overhead of these in Python costs more
    than they save at the depths we actually reach.
    """

    def __init__(self, board_obj, transition_matrix=None, time_left: Callable = None):
        self.T = np.array(transition_matrix, dtype=np.float64)
        self.SZ = enums.BOARD_SIZE

        self.initial_belief = np.zeros(64, dtype=np.float64)
        self.initial_belief[0] = 1.0
        for _ in range(1000):
            self.initial_belief = self.initial_belief @ self.T
        self.belief = self.initial_belief.copy()

        self._cpt = enums.CARPET_POINTS_TABLE

        # Per-turn caches
        self._cell_pot = None
        self._precomputed_beliefs = None

        self.nodes_visited = 0

    # ------------------------------------------------------------------
    def _idx(self, loc):
        return loc[1] * self.SZ + loc[0]

    def _loc(self, idx):
        return (idx % self.SZ, idx // self.SZ)

    def _normalize(self, b):
        s = b.sum()
        return b / s if s > 0 else self.initial_belief.copy()

    # ------------------------------------------------------------------
    def commentate(self):
        return ""

    def play(self, b: board.Board, sensor_data: Tuple, time_left: Callable):
        self.nodes_visited = 0
        t0 = time.time()

        self._update_belief(b, sensor_data)
        self._precompute(b)

        # Precompute beliefs
        max_d = MAX_DEPTH + 2
        self._precomputed_beliefs = [None] * max_d
        self._precomputed_beliefs[0] = self.belief.copy()
        for d in range(1, max_d):
            self._precomputed_beliefs[d] = self._precomputed_beliefs[d - 1] @ self.T

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

        self.belief = self.belief @ self.T

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
    # Precompute
    # ------------------------------------------------------------------
    def _precompute(self, b: board.Board):
        SZ = self.SZ
        px, py = b.player_worker.get_location()

        is_p = [[False]*SZ for _ in range(SZ)]
        for y in range(SZ):
            for x in range(SZ):
                if b.get_cell((x, y)) == enums.Cell.PRIMED:
                    is_p[y][x] = True

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

        pot = [[0.0]*SZ for _ in range(SZ)]  # plain list, not numpy
        cpt = self._cpt

        for y in range(SZ):
            for x in range(SZ):
                ct = b.get_cell((x, y))
                d = abs(px - x) + abs(py - y)
                discount = 1.0 / (1.0 + 0.3 * d)

                if ct == enums.Cell.PRIMED:
                    h = 1 + rL[y][x] + rR[y][x]
                    v = 1 + rU[y][x] + rD[y][x]
                    best = min(max(h, v), 7)
                    pts = cpt[best] if best >= 2 else 0
                    pot[y][x] = max(pts, 0) * discount
                elif ct == enums.Cell.SPACE:
                    h = 1 + rL[y][x] + rR[y][x]
                    v = 1 + rU[y][x] + rD[y][x]
                    hp = cpt[min(h, 7)] if h >= 2 else 0
                    vp = cpt[min(v, 7)] if v >= 2 else 0
                    best_p = max(hp, vp, 0)
                    if best_p == 0:
                        best_p = 0.25
                    pot[y][x] = best_p * discount

        self._cell_pot = pot

    # ------------------------------------------------------------------
    # Iterative deepening — use the time budget aggressively
    # ------------------------------------------------------------------
    def _id_search(self, b, time_left, t0):
        best_move = None
        cands = b.get_valid_moves()
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]

        turns_left = b.player_worker.turns_left

        for depth in range(MIN_DEPTH, MAX_DEPTH + 1):
            rem = time_left()
            if rem < TIME_HARD_FLOOR + 0.5:
                break

            it0 = time.time()
            mv, sc = self._root(b, cands, depth, time_left)
            it_dur = time.time() - it0

            if mv is not None:
                best_move = mv

            rem = time_left()

            # Non-linear time budget
            if turns_left <= 5:
                budget_frac = 0.30
            elif turns_left <= 15:
                budget_frac = 0.07
            elif turns_left <= 25:
                budget_frac = 0.05
            else:
                budget_frac = 0.03

            turn_budget = (rem - TIME_HARD_FLOOR) * budget_frac

            # Will next iteration finish in time?
            if it_dur * 4 > (rem - TIME_HARD_FLOOR):
                break
            if (time.time() - t0) > turn_budget and depth >= MIN_DEPTH:
                break

        return best_move

    # ------------------------------------------------------------------
    # Root — full chance node for search, smart prime ordering
    # ------------------------------------------------------------------
    def _root(self, b, cands, depth, time_left):
        ordered = self._order_root(b, cands)
        best_move = None
        best_sc = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        belief = self._precomputed_beliefs[0]
        margin = b.player_worker.points - b.opponent_worker.points

        # Dynamic search threshold
        if margin >= 8:
            prob_floor = 0.45
        elif margin <= -8:
            prob_floor = 0.20
        else:
            prob_floor = 0.34

        best_idx = int(belief.argmax())
        p = float(belief[best_idx])
        raw_ev = RAT_FIND_PTS * p + RAT_MISS_PTS * (1.0 - p)

        if p >= prob_floor and raw_ev > 0 and depth >= 2:
            search_move = Move.search(self._loc(best_idx))

            # HIT branch
            nb_hit = b.forecast_move(search_move)
            if nb_hit is not None:
                nb_hit.reverse_perspective()
                hit_val = self._emm(nb_hit, float('-inf'), float('inf'),
                                    depth - 1, False, time_left) + RAT_FIND_PTS
            else:
                hit_val = RAT_FIND_PTS

            # MISS branch
            nb_miss = b.forecast_move(search_move)
            if nb_miss is not None:
                nb_miss.reverse_perspective()
                miss_val = self._emm(nb_miss, float('-inf'), float('inf'),
                                     depth - 1, False, time_left) + RAT_MISS_PTS
            else:
                miss_val = RAT_MISS_PTS

            search_sc = p * hit_val + (1.0 - p) * miss_val
            if search_sc > best_sc:
                best_sc = search_sc
                best_move = search_move
                alpha = max(alpha, best_sc)

        for m in ordered:
            if time_left() < TIME_HARD_FLOOR:
                break
            nb = b.forecast_move(m)
            if nb is None:
                continue
            nb.reverse_perspective()
            sc = self._emm(nb, alpha, beta, depth - 1, False, time_left)
            if sc > best_sc:
                best_sc = sc
                best_move = m
            alpha = max(alpha, best_sc)

        return best_move, best_sc

    # ------------------------------------------------------------------
    # EMM — V4-clean, no TT, no killers, no overhead
    # ------------------------------------------------------------------
    def _emm(self, b, alpha, beta, depth, maximizing, time_left):
        self.nodes_visited += 1

        if depth <= 0 or time_left() < TIME_HARD_FLOOR:
            if not maximizing:
                b.reverse_perspective()
            return self._eval(b)

        moves = b.get_valid_moves()
        if not moves:
            if not maximizing:
                b.reverse_perspective()
            return self._eval(b)

        ordered = self._order_fast(moves)

        if maximizing:
            best = float('-inf')

            # Flat EV for rat search (same as V4 — cheap, proven)
            idx = int(np.argmax(self.belief))
            p = self.belief[idx]
            sev = RAT_FIND_PTS * p + RAT_MISS_PTS * (1.0 - p)
            margin = b.player_worker.points - b.opponent_worker.points
            pf = 0.45 if margin >= 8 else (0.20 if margin <= -8 else 0.34)
            if p >= pf and sev > SEARCH_EV_FLOOR:
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
                val = self._emm(nb, alpha, beta, depth - 1, False, time_left)
                if val > best:
                    best = val
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
                val = self._emm(nb, alpha, beta, depth - 1, True, time_left)
                if val < best:
                    best = val
                if best <= alpha:
                    return best
                beta = min(beta, best)
            return best

    # ------------------------------------------------------------------
    # Move ordering
    # ------------------------------------------------------------------
    def _order_root(self, b, moves):
        """Root: use cell potential to rank primes. Only called once."""
        pot = self._cell_pot
        ploc = b.player_worker.position
        cpt = self._cpt

        def key(m):
            if m.move_type == enums.MoveType.CARPET:
                return (3, cpt.get(m.roll_length, 0))
            if m.move_type == enums.MoveType.PRIME:
                dest = loc_after_direction(ploc, m.direction)
                return (2, pot[dest[1]][dest[0]])
            if m.move_type == enums.MoveType.PLAIN:
                return (1, 0)
            return (0, 0)
        return sorted(moves, key=key, reverse=True)

    def _order_fast(self, moves):
        """In-tree: minimal overhead. Carpet by value > prime > plain."""
        cpt = self._cpt
        def key(m):
            mt = m.move_type
            if mt == enums.MoveType.CARPET:
                return (3, cpt.get(m.roll_length, 0))
            if mt == enums.MoveType.PRIME:
                return (2, 0)
            if mt == enums.MoveType.PLAIN:
                return (1, 0)
            return (0, 0)
        return sorted(moves, key=key, reverse=True)

    # ------------------------------------------------------------------
    # Heuristic — V4 style, proven, fast
    # ------------------------------------------------------------------
    def _eval(self, b: board.Board) -> float:
        pp = b.player_worker.points
        op = b.opponent_worker.points
        ploc = b.player_worker.get_location()
        oloc = b.opponent_worker.get_location()
        turns = b.player_worker.turns_left

        margin = pp - op
        score = float(margin)

        # Available carpet + mobility
        moves = b.get_valid_moves()
        best_carpet = 0.0
        carpet_sum = 0.0
        n_prime = 0
        n_plain = 0
        n_carpet = 0
        cpt = self._cpt

        for m in moves:
            mt = m.move_type
            if mt == enums.MoveType.CARPET:
                pts = cpt[m.roll_length]
                if pts > best_carpet:
                    best_carpet = pts
                if pts > 0:
                    carpet_sum += pts
                    n_carpet += 1
            elif mt == enums.MoveType.PRIME:
                n_prime += 1
            elif mt == enums.MoveType.PLAIN:
                n_plain += 1

        score += best_carpet * 0.45
        score += carpet_sum * 0.06
        score += n_carpet * 0.10
        score += n_prime * 0.25
        score += n_plain * 0.04

        # Cell potential (local, distance <= 4)
        pot = self._cell_pot
        if pot is not None:
            px, py = ploc
            SZ = self.SZ
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
                    my_pot += pot[ny][nx]
            score += my_pot * 0.05

        # Opponent carpet threat (immediate only — fast)
        opp_threat = self._opp_threat(b, oloc)
        score -= opp_threat * 0.22

        # Endgame amplification
        if turns <= 10:
            u = (11 - turns) / 10.0
            score += margin * 0.2 * u
            score += best_carpet * 0.35 * u

        # Rat signal
        rp = float(np.max(self.belief))
        rev = RAT_FIND_PTS * rp + RAT_MISS_PTS * (1.0 - rp)
        if rev > 0:
            score += rev * 0.12

        return score

    def _opp_threat(self, b, oloc):
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