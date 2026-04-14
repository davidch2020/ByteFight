"""
Garry – Stockfish-inspired agent for the carpet/rat tournament.

Search features (on top of Yolanda's alpha-beta):
  - Transposition table (hash → depth/value/flag/best_move, 200k cap)
  - Quiescence search: after horizon, continue searching CARPET moves only
    (analogous to captures in chess) until quiet or qdepth limit
  - Late Move Reductions (LMR): PLAIN moves after the 2nd full-depth move
    are searched at depth-2; re-searched at full depth on fail-high
  - Aspiration windows: iterative deepening uses a ±5pt window from the
    previous depth's score; widens exponentially on fail-low/high

Rat handling: HMM belief + EV-based search decision (search when 6p*scale - 2 > 0).
"""

import os, sys, random, time
from collections.abc import Callable
from typing import Tuple

import numpy as np

_HERE   = os.path.dirname(os.path.abspath(__file__))
_AGENTS = os.path.dirname(_HERE)
_ENGINE = os.path.normpath(os.path.join(_HERE, "..", "..", "engine"))
for _p in (_AGENTS, _ENGINE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Fast C++ extension — bundled inside this package directory.
# Try loading by exact file path first (avoids any sys.path weirdness),
# then fall back to name-based import.  Catch all exceptions so a bad .so
# never kills the agent — pure Python fallback always works.
def _load_cpp():
    import importlib.util, importlib.machinery, glob as _glob

    if sys.platform == "win32":
        import ctypes
        _ort_dll = os.path.join(_HERE, "onnxruntime.dll")
        if os.path.exists(_ort_dll):
            try: ctypes.CDLL(_ort_dll)
            except OSError: pass

    # Collect every carpet_ext .so / .pyd in _HERE, sorted so the ABI-tagged
    # ones come before the generic .so fallback.
    patterns = [
        os.path.join(_HERE,   "carpet_ext.cpython-*.so"),
        os.path.join(_HERE,   "carpet_ext.cp*.pyd"),
        os.path.join(_HERE,   "carpet_ext.*.so"),
        os.path.join(_HERE,   "carpet_ext.so"),
        os.path.join(_AGENTS, "carpet_ext.cpython-*.so"),
        os.path.join(_AGENTS, "carpet_ext.cp*.pyd"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(_glob.glob(pat))

    # Prefer ABI-matching version; break ties by newest mtime so a freshly
    # rebuilt 3600-agents/ .pyd beats a stale Garry/ copy.
    import sys as _sys
    ver = f"cpython-{_sys.version_info.major}{_sys.version_info.minor}"
    candidates.sort(key=lambda p: (0 if ver in p else 1, -os.path.getmtime(p)))

    for path in candidates:
        try:
            spec = importlib.util.spec_from_file_location("carpet_ext", path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        except Exception:
            continue

    # Last resort: plain import (picks up whatever is on sys.path)
    sys.path.insert(0, _HERE)
    try:
        import carpet_ext as _m
        return _m
    except Exception:
        return None

_cpp = _load_cpp()
_HAS_CPP = _cpp is not None

from .rat_belief import RatBelief
from .heuristic  import evaluate_board, move_prior, search_threshold
from game.move  import Move
from game.enums import MoveType, Direction, CARPET_POINTS_TABLE, BOARD_SIZE


def _cpp_search(board, budget, belief_vec=None):
    pw = board.player_worker
    ow = board.opponent_worker
    px, py = pw.get_location()
    ox, oy = ow.get_location()
    belief_list = belief_vec.tolist() if belief_vec is not None else []
    result = _cpp.search(
        board._primed_mask, board._carpet_mask, board._blocked_mask,
        px, py, pw.get_points(), pw.turns_left,
        ox, oy, ow.get_points(), ow.turns_left,
        budget,
        belief_list,
    )
    if len(result) == 4:
        mtype, direction, roll, score = result
    else:
        mtype, direction, roll = result
        score = evaluate_board(board)   # old .so: fallback to static eval
    if mtype < 0: return None, float(score)
    if mtype == 3: return Move.search((direction, roll)), float(score)
    d = Direction(direction)
    if mtype == 0: return Move.plain(d), float(score)
    if mtype == 1: return Move.prime(d), float(score)
    if mtype == 2: return Move.carpet(d, roll), float(score)
    return None, float(score)

_INF       = float('inf')
_MAX_DEPTH = 20
_Q_DEPTH   = 6      # max quiescence search depth

# Transposition table flags
_TT_EXACT = 0
_TT_LOWER = 1   # fail-high: value >= beta
_TT_UPPER = 2   # fail-low:  value <= alpha

_TT_MAX   = 200_000


# ── Transposition table ───────────────────────────────────────────────────────

_tt: dict = {}

def _tt_key(board):
    pw, ow = board.player_worker, board.opponent_worker
    px, py = pw.get_location()
    ox, oy = ow.get_location()
    return (
        board._primed_mask,
        board._carpet_mask,
        px | (py << 4),
        ox | (oy << 4),
        pw.turns_left,
        ow.turns_left,
    )

def _tt_probe(key):
    return _tt.get(key)

def _tt_store(key, depth, value, flag, best_move):
    global _tt
    if len(_tt) >= _TT_MAX:
        _tt.clear()
    existing = _tt.get(key)
    if existing is None or existing[0] <= depth:
        _tt[key] = (depth, value, flag, best_move)


# ── Move helpers ──────────────────────────────────────────────────────────────

def _move_key(move):
    return (
        int(move.move_type),
        int(move.direction) if hasattr(move, 'direction') and move.direction is not None else -1,
        int(getattr(move, 'roll_length', 0)),
    )


def _get_moves(board):
    return [
        m for m in board.get_valid_moves(exclude_search=True)
        if not (m.move_type == MoveType.CARPET and m.roll_length == 1)
    ]


def _order_moves(moves, board, depth, killers, history, tt_move=None):
    tt_key_val = _move_key(tt_move) if tt_move is not None else None
    depth_killers = killers.get(depth, [])

    def key(m):
        mk = _move_key(m)
        if mk == tt_key_val:
            return (-2, 0, 0)   # TT move first
        if mk in depth_killers:
            return (-1, -history.get(mk, 0), 0)
        return (0, -history.get(mk, 0), -move_prior(m, board))

    return sorted(moves, key=key)


# ── Quiescence search ─────────────────────────────────────────────────────────

def _quiescence(board, alpha, beta, qdepth, deadline):
    """
    Search only CARPET moves (length >= 2) after the horizon.
    Stand-pat: evaluate statically and use as a lower bound.
    """
    if time.perf_counter() >= deadline:
        raise TimeoutError

    if board.is_game_over():
        return 10_000.0 * (board.player_worker.get_points()
                           - board.opponent_worker.get_points())

    stand_pat = evaluate_board(board)

    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
    if qdepth <= 0:
        return alpha

    carpet_moves = sorted(
        (m for m in board.get_valid_moves(exclude_search=True)
         if m.move_type == MoveType.CARPET and m.roll_length >= 2),
        key=lambda m: -CARPET_POINTS_TABLE[m.roll_length],
    )

    for move in carpet_moves:
        child = board.get_copy()
        if not child.apply_move(move):
            continue
        child.reverse_perspective()

        score = -_quiescence(child, -beta, -alpha, qdepth - 1, deadline)

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


# ── Core negamax ──────────────────────────────────────────────────────────────

def _negamax(board, depth, alpha, beta, killers, history, deadline):
    """Negamax with alpha-beta, TT, quiescence, and LMR."""
    if time.perf_counter() >= deadline:
        raise TimeoutError

    orig_alpha = alpha

    # Transposition table probe
    key      = _tt_key(board)
    tt_entry = _tt_probe(key)
    tt_move  = None
    if tt_entry is not None:
        tt_depth, tt_val, tt_flag, tt_move = tt_entry
        if tt_depth >= depth:
            if tt_flag == _TT_EXACT:
                return tt_val
            elif tt_flag == _TT_LOWER:
                alpha = max(alpha, tt_val)
            elif tt_flag == _TT_UPPER:
                beta = min(beta, tt_val)
            if alpha >= beta:
                return tt_val

    # Terminal / leaf
    if board.is_game_over():
        return 10_000.0 * (board.player_worker.get_points()
                           - board.opponent_worker.get_points())

    if depth == 0:
        return _quiescence(board, alpha, beta, _Q_DEPTH, deadline)

    moves = _get_moves(board)
    if not moves:
        return _quiescence(board, alpha, beta, _Q_DEPTH, deadline)

    moves = _order_moves(moves, board, depth, killers, history, tt_move)

    best      = -_INF
    best_move = None
    n_searched = 0

    for move in moves:
        child = board.get_copy()
        if not child.apply_move(move):
            continue
        child.reverse_perspective()

        # Late Move Reductions: reduce PLAIN moves after 2 full-depth searches
        use_lmr = (
            depth >= 3
            and n_searched >= 2
            and move.move_type == MoveType.PLAIN
        )

        if use_lmr:
            # Search at reduced depth with a null window
            score = -_negamax(child, depth - 2, -(alpha + 1), -alpha,
                              killers, history, deadline)
            if score > alpha:
                # Fail-high: re-search at full depth
                score = -_negamax(child, depth - 1, -beta, -alpha,
                                  killers, history, deadline)
        else:
            score = -_negamax(child, depth - 1, -beta, -alpha,
                              killers, history, deadline)

        n_searched += 1

        if score > best:
            best      = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            mk = _move_key(move)
            k  = killers.setdefault(depth, [])
            if mk not in k:
                k.insert(0, mk)
                if len(k) > 2:
                    k.pop()
            history[mk] = history.get(mk, 0) + depth * depth
            break

    # Store in TT
    if best_move is not None:
        flag = (_TT_EXACT  if orig_alpha < best <= beta - 1 else
                _TT_LOWER  if best >= beta else
                _TT_UPPER)
        _tt_store(key, depth, best, flag, best_move)

    return best


# ── Iterative deepening with aspiration windows ───────────────────────────────

def _iterative_deepening(board, time_budget):
    """Returns (best_move, best_score)."""
    global _tt
    t0       = time.perf_counter()
    deadline = t0 + time_budget * 0.90
    killers  = {}
    history  = {}
    _tt      = {}  # fresh TT per turn (avoids stale entries from different game states)

    moves = _get_moves(board)
    if not moves:
        return None, 0.0
    if len(moves) == 1:
        return moves[0], evaluate_board(board)

    best_move  = max(moves, key=lambda m: move_prior(m, board))
    best_score = evaluate_board(board)
    prev_score = best_score

    for depth in range(1, _MAX_DEPTH + 1):
        if (time.perf_counter() - t0) / time_budget > 0.65:
            break

        # Aspiration window: tight on depth >= 2, full on depth 1
        if depth == 1:
            asp_alpha, asp_beta = -_INF, _INF
        else:
            delta     = 5.0
            asp_alpha = prev_score - delta
            asp_beta  = prev_score + delta

        while True:
            ordered     = _order_moves(moves, board, depth, killers, history)
            alpha       = asp_alpha
            depth_best  = None
            depth_score = -_INF
            failed      = False

            try:
                for move in ordered:
                    child = board.get_copy()
                    if not child.apply_move(move):
                        continue
                    child.reverse_perspective()

                    score = -_negamax(child, depth - 1, -asp_beta, -alpha,
                                      killers, history, deadline)

                    if score > depth_score:
                        depth_score = score
                        depth_best  = move
                    if score > alpha:
                        alpha = score
                    if alpha >= asp_beta:
                        break

            except TimeoutError:
                failed = True
                break

            if failed:
                break

            # Check aspiration window
            if depth_score <= asp_alpha:
                # Fail-low: widen lower bound
                delta    *= 2
                asp_alpha = max(prev_score - delta, -_INF)
            elif depth_score >= asp_beta:
                # Fail-high: widen upper bound
                delta   *= 2
                asp_beta = min(prev_score + delta, _INF)
            else:
                # Within window
                break

        if depth_best is not None and not failed:
            best_move  = depth_best
            best_score = depth_score
            prev_score = depth_score

    return best_move, best_score


# ── PlayerAgent ───────────────────────────────────────────────────────────────

class PlayerAgent:

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.T = (np.array(transition_matrix, dtype=np.float64)
                  if transition_matrix is not None else _fallback_T())
        self.rat_belief = RatBelief(self.T)
        self.first_turn = True
        self.search_penalty = 0.0   # decays slowly; spikes on miss
        self._opp_just_caught = False  # set when opponent catches rat → belief reset
        self.consecutive_misses = 0   # reset belief after too many consecutive misses
        self._last_decision: dict = {}  # populated each turn for external logging

        # Load neural value model if available (built with GARRY_ONNX)
        _model_path = os.path.join(_HERE, "garry_value.onnx")
        if _HAS_CPP and os.path.exists(_model_path):
            try:
                _cpp.load_value_model(_model_path)
            except AttributeError:
                pass  # .pyd built without ONNX support

    def commentate(self) -> str:
        mode = "C++" if _HAS_CPP else "Python-TT+LMR+quiescence"
        return f"Garry (Stockfish-style/{mode}/HMM)"

    def play(self, board, sensor_data: Tuple, time_left: Callable) -> Move:
        noise, dist  = sensor_data
        t_available  = time_left()

        # Decaying search penalty: spikes on miss, decays on board moves.
        # Prevents both consecutive-miss spirals AND miss→boardmove→miss patterns.
        if not self.first_turn:
            my_loc, my_found = board.player_search
            if my_loc is not None:   # we searched last turn
                if my_found:
                    self.search_penalty     = 0.0
                    self.consecutive_misses = 0     # reset on hit
                else:
                    self.search_penalty     = min(self.search_penalty + 3.0, 8.0)
                    self.consecutive_misses += 1    # accumulate across board moves
            else:                    # board move last turn — decay slowly
                self.search_penalty = max(0.0, self.search_penalty - 0.5)
                # Do NOT reset consecutive_misses on board moves — count total misses
                # since last hit so a miss→board→miss→board→miss pattern still triggers

        # After opponent catches rat, belief resets to stationary (≈uniform).
        # No single cell exceeds ~0.03-0.05 — all searches are negative EV.
        # Spike penalty to maximum so C++ won't recommend searching.
        if self._opp_just_caught:
            self.search_penalty = 8.0
            self._opp_just_caught = False

        self._update_belief(board, noise, dist)

        turns_left = board.player_worker.turns_left
        budget     = min(t_available / max(turns_left, 1) * 0.80, 10.0)
        budget     = max(budget, 0.05)
        budget     = min(budget, t_available - 0.20)

        if budget <= 0.02:
            return self._emergency(board)

        # After 3 consecutive misses, we've lost the rat — reset belief so we stop
        # chasing the wrong concentration. Mirrors conservative search policy.
        if self.consecutive_misses >= 3:
            self.rat_belief.reset_to_stationary()
            self.consecutive_misses = 0
            self.search_penalty     = 8.0   # prevent immediate re-search after reset

        raw_belief = self.rat_belief.belief
        raw_max_p  = float(np.max(raw_belief))
        _, best_search_ev = self.rat_belief.best_search()

        # Search should compete against the best board move available, not a
        # fixed probability threshold. A miss penalty still makes us more
        # conservative after bad searches.
        search_bar = search_threshold(board, self.rat_belief)
        effective_search_ev = best_search_ev - self.search_penalty

        # When clearly behind late, allow a little more rat variance.
        my_pts  = board.player_worker.get_points()
        opp_pts = board.opponent_worker.get_points()
        behind_by = opp_pts - my_pts
        if turns_left <= 4 and behind_by >= 4:
            search_bar = max(0.5, search_bar - 0.75)

        scale = max(0.0, 1.0 - self.search_penalty / 8.0)
        search_allowed = (
            raw_max_p > (1.0 / 3.0)
            and effective_search_ev >= search_bar
            and scale > 0.0
        )
        if search_allowed:
            belief_vec = raw_belief.copy() * scale
        else:
            belief_vec = np.zeros(len(raw_belief), dtype=np.float32)

        gate_blocked = not search_allowed

        if _HAS_CPP:
            move, cpp_score = _cpp_search(board, budget, belief_vec)
        else:
            move, cpp_score = _iterative_deepening(board, budget)
            cpp_score = cpp_score or 0.0

        # Endgame search guard: at t<=2, PRIME guarantees +1.
        endgame_override = False
        if (move is not None and move.move_type == MoveType.SEARCH
                and turns_left <= 2):
            max_p     = float(np.max(self.rat_belief.belief))
            if behind_by <= 4 and max_p < 0.5:
                endgame_override = True
                move, cpp_score = _cpp_search(board, min(budget * 0.3, 0.5),
                                              np.zeros(len(self.rat_belief.belief), dtype=np.float32))

        # Fall back to emergency move if C++ found no board moves
        forced_search = False
        if move is None:
            move = self._emergency(board)
            if move is not None and move.move_type == MoveType.SEARCH:
                forced_search = True   # no board moves available — search is the only option

        # Log decision details for external analysis tools (e.g. analyze_games.py)
        best_idx = int(np.argmax(self.rat_belief.belief))
        self._last_decision = {
            'raw_max_p':          raw_max_p,
            'best_cell':          (best_idx % 8, best_idx // 8),
            'best_search_ev':     best_search_ev,
            'search_threshold':   search_bar,
            'effective_search_ev': effective_search_ev,
            'penalty':            self.search_penalty,
            'scale':              scale,
            'consecutive_misses': self.consecutive_misses,
            'gate_blocked':       gate_blocked,
            'endgame_override':   endgame_override,
            'forced_search':      forced_search,
            'cpp_score':          cpp_score,
            'chose_search':       (move is not None and move.move_type == MoveType.SEARCH),
        }

        return move

    def _update_belief(self, board, noise: int, dist: int):
        if self.first_turn:
            self.rat_belief.predict()
            self.first_turn = False
        else:
            my_loc, my_found = board.player_search
            if my_loc is not None and not my_found:
                self.rat_belief.zero_cell(my_loc)
            self.rat_belief.predict()
            opp_loc, opp_found = board.opponent_search
            if opp_loc is not None:
                if opp_found:
                    self.rat_belief.reset_to_stationary()
                    self._opp_just_caught = True  # spike penalty next turn
                else:
                    self.rat_belief.zero_cell(opp_loc)
            self.rat_belief.predict()
        self.rat_belief.update(int(noise), int(dist),
                               board.player_worker.get_location(), board)

    def _emergency(self, board) -> Move:
        moves = board.get_valid_moves(exclude_search=False)
        if not moves:
            return None
        carpets = [m for m in moves
                   if m.move_type == MoveType.CARPET and m.roll_length >= 2]
        if carpets:
            return max(carpets, key=lambda m: m.roll_length)
        # Never play CARPET(1) — it costs -1 point. Prefer PRIME or PLAIN.
        non_search = [m for m in moves
                      if m.move_type not in (MoveType.SEARCH,)
                      and not (m.move_type == MoveType.CARPET and m.roll_length == 1)]
        if non_search:
            return random.choice(non_search)
        return random.choice(moves)


def _fallback_T():
    n = BOARD_SIZE * BOARD_SIZE
    T = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        y, x = divmod(i, BOARD_SIZE)
        nbrs = [i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                nbrs.append(ny * BOARD_SIZE + nx)
        for j in nbrs:
            T[i, j] = 1.0 / len(nbrs)
    return T
