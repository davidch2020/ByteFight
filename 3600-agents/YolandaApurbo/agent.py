from collections.abc import Callable
from typing import List, Tuple
import random
import time as time_module

import numpy as np

from game import board, move, enums, rat
from game.move import Move

# ---------------------------------------------------------------------------
# Constants — localized to avoid module attribute lookups in hot paths
# ---------------------------------------------------------------------------
SEARCH_PROB_THRESHOLD = 0.50
RAT_FIND_PTS = 4
RAT_MISS_PTS = 2

# Time: exponential decay scaled from StockChicken.
# Sum of 17.5 * 0.93^(t-1) for t=1..40 ≈ 236s, leaving ~4s safety margin.
# Zero turns get clamped — every turn uses its full budget.
# Turn 1: 17.5s, Turn 10: 9.1s, Turn 20: 4.4s, Turn 40: 1.0s
TIME_BASE = 17.5
TIME_DECAY = 0.93
TIME_FLOOR = 1.5          # Never let total remaining drop below this

MAX_DEPTH = 40             # Iterative deepening stops on time, not this

# Carpet points as a list for O(1) index lookup (index 0 unused)
_CPT = [0, -1, 2, 4, 6, 10, 15, 21]  # index = roll_length

# Inline constants to avoid module lookups in tight loops
_CARPET_MT = int(enums.MoveType.CARPET)
_PRIME_MT = int(enums.MoveType.PRIME)
_PLAIN_MT = int(enums.MoveType.PLAIN)

# Noise probs as a flat lookup: _NOISE_LUT[(cell_type, noise_type)] = prob
_NOISE_LUT = {}
for _ct, _probs in rat.NOISE_PROBS.items():
    for _nt in range(3):
        _NOISE_LUT[(int(_ct), _nt)] = _probs[_nt]

_DIST_ERR = rat.DISTANCE_ERROR_PROBS  # (0.12, 0.7, 0.12, 0.06)


# ---------------------------------------------------------------------------
# Transposition Table
# ---------------------------------------------------------------------------
class TT:
    __slots__ = ('table', 'hits', 'misses')

    def __init__(self):
        self.table = {}
        self.hits = 0
        self.misses = 0

    def lookup(self, key, depth, alpha, beta):
        """Returns (value, best_move_key, found)."""
        entry = self.table.get(key)
        if entry is None:
            self.misses += 1
            return 0.0, None, False

        sd, sv, sm, nt = entry
        if sd < depth:
            self.misses += 1
            return 0.0, sm, False  # Move hint even on depth miss

        if nt == 0:  # EXACT
            self.hits += 1
            return sv, sm, True
        elif nt == 1 and sv >= beta:  # LOWER
            self.hits += 1
            return sv, sm, True
        elif nt == 2 and sv <= alpha:  # UPPER
            self.hits += 1
            return sv, sm, True

        self.misses += 1
        return 0.0, sm, False

    def store(self, key, depth, value, best_move_key, orig_alpha, beta):
        if value <= orig_alpha:
            nt = 2   # UPPER
        elif value >= beta:
            nt = 1   # LOWER
        else:
            nt = 0   # EXACT

        old = self.table.get(key)
        if old is None or depth >= old[0]:
            self.table[key] = (depth, value, best_move_key, nt)

    def maybe_clear(self, max_size=500000):
        """If table is too large, wipe it. O(1) via GC, much faster than partial eviction."""
        if len(self.table) > max_size:
            self.table = {}


# ---------------------------------------------------------------------------
# Fast helpers (module-level to avoid method lookup overhead)
# ---------------------------------------------------------------------------
def _best_carpet_run(primed_mask, px, py, ox, oy):
    """Best carpet points available from (px,py), blocked by (ox,oy).
    Pure bitboard — zero get_cell() calls."""
    best = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = px + dx, py + dy
        run = 0
        while 0 <= nx < 8 and 0 <= ny < 8:
            if nx == ox and ny == oy:
                break
            if not ((primed_mask >> (ny * 8 + nx)) & 1):
                break
            run += 1
            nx += dx
            ny += dy
        if run >= 1:
            pts = _CPT[min(run, 7)]
            if pts > best:
                best = pts
    return best


def _eval_board(b):
    """Ultra-lean heuristic. No get_cell(). No get_valid_moves().
    ~30 bit ops + 2 direction scans (max 8 steps each)."""
    pw = b.player_worker
    ow = b.opponent_worker
    pp = pw.points
    op_ = ow.points
    px, py = pw.position
    ox, oy = ow.position
    turns = pw.turns_left if pw.turns_left > 0 else 1
    primed_mask = b._primed_mask

    # 1. Score margin, weighted by remaining game length
    margin = pp - op_
    score = margin * (1.0 + turns * 0.12)

    # 2. Best carpet available to player RIGHT NOW
    score += _best_carpet_run(primed_mask, px, py, ox, oy) * 0.5

    # 3. Best carpet available to opponent (threat)
    score -= _best_carpet_run(primed_mask, ox, oy, px, py) * 0.3

    # 4. Primed cell count — proxy for future carpet infrastructure
    score += bin(primed_mask).count('1') * 0.12

    # 5. Can we prime from current position? (must be on SPACE)
    my_bit = 1 << (py * 8 + px)
    if not ((primed_mask | b._carpet_mask) & my_bit):
        score += 0.4

    # 6. Adjacent primed cells (potential carpet starts — 4 checks)
    adj = 0
    if px > 0 and (primed_mask >> (py * 8 + px - 1)) & 1: adj += 1
    if px < 7 and (primed_mask >> (py * 8 + px + 1)) & 1: adj += 1
    if py > 0 and (primed_mask >> ((py - 1) * 8 + px)) & 1: adj += 1
    if py < 7 and (primed_mask >> ((py + 1) * 8 + px)) & 1: adj += 1
    score += adj * 0.2

    return score


def _order_moves(moves, tt_hint, killers):
    """Bucket-sort moves. ~3x faster than sorted() with a closure in Python.
    TT hint → killers → carpets (by length desc) → primes → plains."""
    front = []
    carpets = []
    primes = []
    plains = []

    for m in moves:
        mt = m.move_type
        # Check if this matches the TT hint
        if tt_hint is not None and mt == tt_hint[0] and m.direction == tt_hint[2]:
            if mt != _CARPET_MT or m.roll_length == tt_hint[1]:
                front.insert(0, m)
                continue

        if mt == _CARPET_MT:
            carpets.append(m)
        elif mt == _PRIME_MT:
            primes.append(m)
        else:
            plains.append(m)

    # Sort carpets by roll_length descending (longer = more points)
    if len(carpets) > 1:
        carpets.sort(key=lambda m: m.roll_length, reverse=True)

    # Pull killer matches to front
    if killers:
        rest_c = []
        rest_p = []
        rest_pl = []
        for m in carpets:
            if _match_killer(m, killers):
                front.append(m)
            else:
                rest_c.append(m)
        for m in primes:
            if _match_killer(m, killers):
                front.append(m)
            else:
                rest_p.append(m)
        for m in plains:
            if _match_killer(m, killers):
                front.append(m)
            else:
                rest_pl.append(m)
        return front + rest_c + rest_p + rest_pl

    return front + carpets + primes + plains


def _match_killer(m, killers):
    """Check if move matches any killer. Killers are (mt, roll, dir) tuples."""
    mt = m.move_type
    d = m.direction
    for kt, kr, kd in killers:
        if mt == kt and d == kd:
            if mt != _CARPET_MT or m.roll_length == kr:
                return True
    return False


def _move_key(m):
    """Compact tuple for TT/killer storage. No Move object references held."""
    return (m.move_type,
            m.roll_length if m.move_type == _CARPET_MT else 0,
            m.direction)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class PlayerAgent:
    """
    V8: Speed-first minimax with TT, killers, lean eval.
    Designed to use the full 240s budget and search as deep as possible.
    """

    def __init__(self, board_obj, transition_matrix=None, time_left: Callable = None):
        self.T = np.array(transition_matrix, dtype=np.float64)
        self.initial_belief = np.zeros(64, dtype=np.float64)
        self.initial_belief[0] = 1.0
        for _ in range(1000):
            self.initial_belief = self.initial_belief @ self.T
        self.belief = self.initial_belief.copy()

        self.tt = TT()
        self.killer_moves = {}   # depth -> [(mt, roll, dir), ...]
        self.turn_number = 0

    def commentate(self):
        t = self.tt
        total = t.hits + t.misses
        r = t.hits / total if total > 0 else 0
        return f"TT: {t.hits}/{total} ({r:.0%}), size={len(t.table)}"

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------
    def play(self, b: board.Board, sensor_data: Tuple, time_left: Callable):
        self.turn_number += 1
        self.killer_moves.clear()
        self.tt.maybe_clear()

        # 1. Rat belief
        self._update_belief(b, sensor_data)

        # 2. Rat search candidate
        search_move = self._choose_search(b)

        # 3. Time budget for this turn
        budget = TIME_BASE * (TIME_DECAY ** (self.turn_number - 1))
        remaining = time_left() - TIME_FLOOR
        alloc = max(0.05, min(budget, remaining))
        deadline = time_module.time() + alloc

        # 4. Search
        best = self._id_search(b, search_move, deadline, time_left)
        if best is not None:
            return best
        moves = b.get_valid_moves()
        return random.choice(moves) if moves else Move.plain(enums.Direction.UP)

    # ------------------------------------------------------------------
    # Iterative deepening
    # ------------------------------------------------------------------
    def _id_search(self, b, search_move, deadline, time_left):
        cands = b.get_valid_moves()
        if not cands:
            return None
        if len(cands) == 1 and search_move is None:
            return cands[0]

        best_move = cands[0]
        _now = time_module.time  # Cache function reference

        for depth in range(1, MAX_DEPTH + 1):
            if _now() >= deadline:
                break

            mv, sc = self._root(b, cands, search_move, depth, deadline)
            if mv is not None:
                best_move = mv
            else:
                break  # Timed out mid-search

            # Safety: don't start next depth if globally low
            if time_left() < TIME_FLOOR + 0.3:
                break

        return best_move

    # ------------------------------------------------------------------
    # Root
    # ------------------------------------------------------------------
    def _root(self, b, cands, search_move, depth, deadline):
        _now = time_module.time
        rk = self._board_key(b, True)
        _, tt_hint, _ = self.tt.lookup(rk, depth, -999999.0, 999999.0)
        killers = self.killer_moves.get(depth, [])
        ordered = _order_moves(cands, tt_hint, killers)

        best_move = None
        best_sc = -999999.0
        alpha = -999999.0
        beta = 999999.0

        # Rat search as candidate
        if search_move is not None:
            bidx = int(np.argmax(self.belief))
            p = float(self.belief[bidx])
            sev = RAT_FIND_PTS * p - RAT_MISS_PTS * (1.0 - p)
            if sev > 0:
                best_sc = sev
                best_move = search_move
                alpha = sev

        for m in ordered:
            if _now() >= deadline:
                return None, None
            nb = b.forecast_move(m)
            if nb is None:
                continue
            nb.reverse_perspective()
            sc = self._ab(nb, alpha, beta, depth - 1, False, deadline)
            if sc is None:
                return None, None
            if sc > best_sc:
                best_sc = sc
                best_move = m
            if best_sc > alpha:
                alpha = best_sc

        return best_move, best_sc

    # ------------------------------------------------------------------
    # Alpha-Beta
    # ------------------------------------------------------------------
    def _ab(self, b, alpha, beta, depth, maximizing, deadline):
        if time_module.time() >= deadline:
            return None

        # Leaf
        if depth <= 0 or b.winner is not None:
            if not maximizing:
                b.reverse_perspective()
            return _eval_board(b)

        # TT probe
        key = self._board_key(b, maximizing)
        tt_val, tt_hint, tt_found = self.tt.lookup(key, depth, alpha, beta)
        if tt_found:
            return tt_val

        moves = b.get_valid_moves()
        if not moves:
            if not maximizing:
                b.reverse_perspective()
            return _eval_board(b)

        killers = self.killer_moves.get(depth, [])
        ordered = _order_moves(moves, tt_hint, killers)
        orig_alpha = alpha
        best_mk = None
        _now = time_module.time

        if maximizing:
            best = -999999.0
            for m in ordered:
                if _now() >= deadline:
                    return None
                nb = b.forecast_move(m)
                if nb is None:
                    continue
                nb.reverse_perspective()
                val = self._ab(nb, alpha, beta, depth - 1, False, deadline)
                if val is None:
                    return None
                if val > best:
                    best = val
                    best_mk = _move_key(m)
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    self._add_killer(depth, m)
                    break
        else:
            best = 999999.0
            for m in ordered:
                if _now() >= deadline:
                    return None
                nb = b.forecast_move(m)
                if nb is None:
                    continue
                nb.reverse_perspective()
                val = self._ab(nb, alpha, beta, depth - 1, True, deadline)
                if val is None:
                    return None
                if val < best:
                    best = val
                    best_mk = _move_key(m)
                if best < beta:
                    beta = best
                if alpha >= beta:
                    self._add_killer(depth, m)
                    break

        self.tt.store(key, depth, best, best_mk, orig_alpha, beta)
        return best

    # ------------------------------------------------------------------
    # Board key for TT
    # ------------------------------------------------------------------
    def _board_key(self, b, maximizing):
        return (
            b._primed_mask,
            b._carpet_mask,
            b.player_worker.position,
            b.opponent_worker.position,
            b.player_worker.points,
            b.opponent_worker.points,
            b.player_worker.turns_left,
            maximizing,
        )

    # ------------------------------------------------------------------
    # Killer moves — lightweight tuples
    # ------------------------------------------------------------------
    def _add_killer(self, depth, m):
        mk = _move_key(m)
        ks = self.killer_moves.get(depth)
        if ks is None:
            self.killer_moves[depth] = [mk]
            return
        if mk not in ks:
            if len(ks) >= 2:
                ks[1] = ks[0]
                ks[0] = mk
            else:
                ks.insert(0, mk)

    # ------------------------------------------------------------------
    # Rat HMM (V5 logic, optimized access)
    # ------------------------------------------------------------------
    def _update_belief(self, b, sd):
        ol, of_ = b.opponent_search
        pl, pf = b.player_search

        if pf:
            self.belief = self.initial_belief.copy()
        elif pl is not None:
            self.belief[pl[1] * 8 + pl[0]] = 0.0
            self._norm()

        if of_:
            self.belief = self.initial_belief.copy()
        elif ol is not None:
            self.belief[ol[1] * 8 + ol[0]] = 0.0
            self._norm()

        # Predict
        self.belief = self.belief @ self.T

        # Sensor update — inline bitboard, no get_cell()
        wx, wy = b.player_worker.position
        noise = sd[0]
        rdist = sd[1]
        pm = b._primed_mask
        cm = b._carpet_mask
        bm = b._blocked_mask

        belief = self.belief  # Local ref for speed
        for i in range(64):
            cx = i & 7       # i % 8
            cy = i >> 3      # i // 8
            bit = 1 << i

            # Cell type via bitboard
            if pm & bit:
                ct = 1
            elif cm & bit:
                ct = 2
            elif bm & bit:
                ct = 3
            else:
                ct = 0

            np_ = _NOISE_LUT[(ct, noise)]
            md = abs(wx - cx) + abs(wy - cy)
            diff = rdist - md
            if -1 <= diff <= 2:
                dp = _DIST_ERR[diff + 1]
            else:
                dp = 0.0

            belief[i] *= np_ * dp

        self._norm()

    def _norm(self):
        s = self.belief.sum()
        if s > 0:
            self.belief /= s
        else:
            self.belief = self.initial_belief.copy()

    def _choose_search(self, b):
        bidx = int(np.argmax(self.belief))
        p = float(self.belief[bidx])
        if p < SEARCH_PROB_THRESHOLD:
            return None

        ev = RAT_FIND_PTS * p - RAT_MISS_PTS * (1.0 - p)

        # Opportunity cost check
        px, py = b.player_worker.position
        ox, oy = b.opponent_worker.position
        bc = _best_carpet_run(b._primed_mask, px, py, ox, oy)
        if bc > ev + 1:
            return None

        if ev > 0.5:
            return Move.search((bidx & 7, bidx >> 3))
        return None