"""
Board evaluation heuristic.

Game-theory analysis:
  Points per turn for carpet-of-n:
    n : 1     2     3     4     5     6     7
    pts/turn : 0.0  1.33  1.75  2.0   2.5   3.0   3.5   (including prime points)

  Key insight: NEVER do single-carpet (-1 pts).
               Maximise run length; 7-carpet is 3.5 pts/turn.

  Threat model:
    - Opponent can STEAL your primed squares by carpet-rolling over them.
    - A long primed run that the opponent can reach is a HUGE liability.
    - Blocking the end of opponent's primed line (by standing there) is very powerful.

  Rat strategy:
    EV(search at cell c) = 6 * P(rat at c) - 2.
    Break-even at P = 1/3.
    Opportunity cost of searching = best available carpet value per turn.
"""

import numpy as np
from typing import Tuple
from game.enums import BOARD_SIZE, Direction, MoveType, CARPET_POINTS_TABLE, loc_after_direction

N_CELLS = 64

# Scoring weights for the linear value function
W_SCORE_DIFF    =  1.00
W_MY_CARPET     =  0.70   # best immediate carpet I can make
W_MY_PRIMED     =  0.40   # future value of primed runs I can access
W_OPP_CARPET    = -0.60   # opponent's immediate carpet opportunity
W_OPP_PRIMED    = -0.35   # future value of primed runs opponent can access
W_RAT_EV        =  0.00   # handled outside MCTS tree (set to 0 here)


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def evaluate_board(board, rat_belief=None) -> float:
    """
    Static evaluation of `board` from the CURRENT PLAYER's perspective.
    Returns estimated point differential (positive = we are ahead).

    NOTE: rat_belief is intentionally NOT used here because:
    1. In the MCTS tree, child nodes use the opponent's perspective but
       our belief – mixing perspectives causes sign errors.
    2. Search decisions are made in agent.py using current belief directly.
    """
    my_pts  = board.player_worker.get_points()
    opp_pts = board.opponent_worker.get_points()
    score_diff = float(my_pts - opp_pts)

    turns_me  = board.player_worker.turns_left
    turns_opp = board.opponent_worker.turns_left

    if turns_me == 0 and turns_opp == 0:
        return score_diff   # terminal

    my_pos  = board.player_worker.get_location()
    opp_pos = board.opponent_worker.get_location()
    pm = board._primed_mask
    workers_mask = (_loc_bit(my_pos) | _loc_bit(opp_pos))
    carpetable = pm & ~workers_mask

    # Best carpet immediately available to each player from their position
    my_best_carpet  = _best_carpet_from(my_pos,  carpetable)
    opp_best_carpet = _best_carpet_from(opp_pos, carpetable)

    # Future primed-run potential ACCESSIBLE to each player
    # (how much carpet value can each player eventually make from primed runs?)
    my_primed  = _accessible_primed_value(my_pos,  pm, workers_mask)
    opp_primed = _accessible_primed_value(opp_pos, pm, workers_mask)

    val = (
        W_SCORE_DIFF  * score_diff
      + W_MY_CARPET   * my_best_carpet
      + W_MY_PRIMED   * my_primed
      + W_OPP_CARPET  * opp_best_carpet
      + W_OPP_PRIMED  * opp_primed
    )

    return val


def best_carpet_available(board, for_player: bool = True) -> float:
    """
    Return the maximum CARPET_POINTS_TABLE value that `for_player` can
    achieve in one carpet move from their current position.
    """
    if for_player:
        pos = board.player_worker.get_location()
        other_pos = board.opponent_worker.get_location()
    else:
        pos = board.opponent_worker.get_location()
        other_pos = board.player_worker.get_location()

    workers_mask = _loc_bit(pos) | _loc_bit(other_pos)
    carpetable = board._primed_mask & ~workers_mask
    return _best_carpet_from(pos, carpetable)


def search_threshold(board, rat_belief) -> float:
    """
    Adaptive threshold: rat search must beat the carpet opportunity cost.

    Core idea: carpet is the primary win condition. Only search when:
      a) carpet EV is genuinely low (no runs to exploit), OR
      b) rat EV clearly exceeds what carpeting would earn.

    Opponent position matters:
      - If opponent threatens to steal our best run (adjacent to far end),
        our run is "use it now or lose it" — carpet even more urgently.
      - If opponent just stole our runs (our carpet EV collapsed),
        rat search becomes relatively more attractive.
    """
    my_pos  = board.player_worker.get_location()
    opp_pos = board.opponent_worker.get_location()

    my_carpet = best_carpet_available(board, for_player=True)

    # Base threshold: rat EV must match full carpet opportunity
    # Below 1.0 carpet EV → no real carpet to give up, floor at 0.5
    if my_carpet < 1.0:
        threshold = 0.5
    else:
        threshold = my_carpet * 1.0   # direct EV comparison, not 0.6

    # Opponent threat: check if opponent is adjacent to the far end of
    # our best primed run (can block/steal next turn)
    steal_threat = _opponent_steal_threat(my_pos, opp_pos, board)
    if steal_threat > 0:
        # Run is about to be lost — make carpeting even more urgent
        threshold += steal_threat * 0.5

    # Endgame: willing to search more when turns are short
    turns_left = board.player_worker.turns_left
    if turns_left <= 5:
        threshold = max(0.5, threshold - 1.0)

    return threshold


def _opponent_steal_threat(my_pos, opp_pos, board) -> float:
    """
    Returns the carpet value at risk from opponent stealing our best run.
    Checks if opponent is ≤1 plain move from the far end of any primed run
    that we could otherwise carpet.
    """
    pm = board._primed_mask
    if pm == 0:
        return 0.0

    workers_mask = _loc_bit(my_pos) | _loc_bit(opp_pos)
    ox, oy = opp_pos

    best_threatened = 0.0
    for direction in (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT):
        # Walk the run from my position
        run = 0
        cur = my_pos
        far_end = my_pos
        while run < 7:
            nxt = loc_after_direction(cur, direction)
            if not (0 <= nxt[0] < BOARD_SIZE and 0 <= nxt[1] < BOARD_SIZE):
                break
            bit = 1 << (nxt[1] * BOARD_SIZE + nxt[0])
            if not (pm & bit) or (workers_mask & bit):
                break
            run += 1
            far_end = nxt
            cur = nxt

        if run < 2:
            continue

        val = float(CARPET_POINTS_TABLE[run])
        if val <= 0:
            continue

        # Is opponent adjacent to far_end (or one plain step away)?
        dist = abs(ox - far_end[0]) + abs(oy - far_end[1])
        if dist <= 1:
            best_threatened = max(best_threatened, val)

    return best_threatened


# -----------------------------------------------------------------------
# Fast prior for MCTS node expansion (no full board eval per child)
# -----------------------------------------------------------------------

def move_prior(move, board) -> float:
    """
    Heuristic prior probability for MCTS (unnormalized).
    Reflects game-theoretic intuition about move quality.
    """
    mt = move.move_type
    if mt == MoveType.CARPET:
        pts = CARPET_POINTS_TABLE[move.roll_length]
        if pts <= 0:
            return 0.05   # never do 1-carpets
        # Strongly prefer longer carpets (3.5x bonus for length 7)
        return 0.5 + float(pts) * 0.5
    elif mt == MoveType.PRIME:
        pos = board.player_worker.get_location()
        # The cell we're priming is pos (current cell)
        # The run direction: how many primed cells exist in move.direction from pos?
        # Priming along an existing run is great (extending it)
        run_forward = _run_length_from(
            loc_after_direction(pos, move.direction),
            move.direction, board._primed_mask, 0
        )
        # Also check if next cell has space to build (non-blocked)
        next_pos = loc_after_direction(pos, move.direction)
        if not (0 <= next_pos[0] < BOARD_SIZE and 0 <= next_pos[1] < BOARD_SIZE):
            return 0.1
        # Give strong prior for priming that extends toward a potential long run
        # Check how many cells are free in the prime direction (future run potential)
        free_cells = 0
        cur = next_pos
        for _ in range(7):
            nxt = loc_after_direction(cur, move.direction)
            if not (0 <= nxt[0] < BOARD_SIZE and 0 <= nxt[1] < BOARD_SIZE):
                break
            bit = 1 << (nxt[1] * BOARD_SIZE + nxt[0])
            if board._blocked_mask & bit or board._carpet_mask & bit:
                break
            free_cells += 1
            cur = nxt
        # Total potential run = 1 (this prime) + run_forward + free_cells
        potential = 1 + run_forward + free_cells
        return 1.0 + potential * 0.5  # 1.0 to 4.5 depending on potential
    elif mt == MoveType.PLAIN:
        # Prefer moves that position us adjacent to a long carpetable run.
        # A PLAIN step to a cell from which you can immediately carpet a big run
        # is nearly as good as carpet itself (one turn away) — give it high priority.
        nxt = loc_after_direction(board.player_worker.get_location(), move.direction)
        workers_mask = (_loc_bit(board.player_worker.get_location()) |
                        _loc_bit(board.opponent_worker.get_location()))
        carpetable = board._primed_mask & ~workers_mask
        carpet_val = _best_carpet_from(nxt, carpetable)
        return 0.5 + carpet_val * 0.1
    elif mt == MoveType.SEARCH:
        return 0.3
    return 1.0


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _loc_bit(pos: Tuple[int, int]) -> int:
    """Position (x,y) → bit mask."""
    return 1 << (pos[1] * BOARD_SIZE + pos[0])


def _best_carpet_from(pos: Tuple[int, int], carpetable_mask: int) -> float:
    """
    Best carpet points value reachable from `pos` in any direction,
    given the set of carpetable cells (primed & not occupied).
    """
    best = 0.0
    for direction in (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT):
        run = 0
        cur = pos
        while run < 7:
            cur = loc_after_direction(cur, direction)
            if not (0 <= cur[0] < BOARD_SIZE and 0 <= cur[1] < BOARD_SIZE):
                break
            bit = 1 << (cur[1] * BOARD_SIZE + cur[0])
            if not (carpetable_mask & bit):
                break
            run += 1
        if run > 0:
            val = float(CARPET_POINTS_TABLE[run])
            if val > best:
                best = val
    return best


def _run_length_from(pos: Tuple[int, int], direction: Direction,
                     primed_mask: int, start_run: int) -> int:
    """Count primed squares in `direction` starting from `pos`."""
    run = start_run
    cur = pos
    while run < 7:
        if not (0 <= cur[0] < BOARD_SIZE and 0 <= cur[1] < BOARD_SIZE):
            break
        bit = 1 << (cur[1] * BOARD_SIZE + cur[0])
        if not (primed_mask & bit):
            break
        run += 1
        cur = loc_after_direction(cur, direction)
    return run


def _accessible_primed_value(pos: Tuple[int, int], primed_mask: int,
                              workers_mask: int) -> float:
    """
    Best carpet value reachable by a player at `pos` considering:
    1. Direct carpet from current position (immediate).
    2. Carpet accessible within 2 moves (near-future).
    """
    if primed_mask == 0:
        return 0.0

    carpetable = primed_mask & ~workers_mask

    # Immediate carpet
    immediate = _best_carpet_from(pos, carpetable)

    # Near-future: cells reachable in 1-2 plain steps (through non-primed space)
    best_future = 0.0
    visited = {pos}
    frontier = [pos]
    for _depth in range(2):
        next_f = []
        for cur in frontier:
            for direction in (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT):
                nxt = loc_after_direction(cur, direction)
                if not (0 <= nxt[0] < BOARD_SIZE and 0 <= nxt[1] < BOARD_SIZE):
                    continue
                if nxt in visited:
                    continue
                bit = 1 << (nxt[1] * BOARD_SIZE + nxt[0])
                # Can only plain-step to non-primed, non-worker cells
                if primed_mask & bit or workers_mask & bit:
                    continue
                visited.add(nxt)
                next_f.append(nxt)
                v = _best_carpet_from(nxt, carpetable)
                if v > best_future:
                    best_future = v
        frontier = next_f

    return max(immediate, best_future * 0.6)   # discount future by 40%


def _total_primed_run_value(primed_mask: int) -> float:
    """
    Sum of CARPET_POINTS_TABLE values for all horizontal and vertical
    maximal runs of primed squares.  Used as a proxy for future carpet
    potential on the board.
    """
    if primed_mask == 0:
        return 0.0

    total = 0.0
    seen_h = 0  # visited bits for horizontal scan
    seen_v = 0  # visited bits for vertical scan

    for i in range(N_CELLS):
        if not ((primed_mask >> i) & 1):
            continue

        x, y = i % BOARD_SIZE, i // BOARD_SIZE

        # --- Horizontal run (scan right from left edge of run) ---
        if not ((seen_h >> i) & 1):
            # Find leftmost cell of this horizontal run
            start_x = x
            while start_x > 0 and (primed_mask >> (y * BOARD_SIZE + start_x - 1)) & 1:
                start_x -= 1
            # Count run length
            run = 0
            cx = start_x
            while cx < BOARD_SIZE and (primed_mask >> (y * BOARD_SIZE + cx)) & 1:
                seen_h |= (1 << (y * BOARD_SIZE + cx))
                run += 1
                cx += 1
            run = min(run, 7)
            if run >= 2:  # runs of 1 give -1 (ignore as a positive)
                total += CARPET_POINTS_TABLE[run]

        # --- Vertical run ---
        if not ((seen_v >> i) & 1):
            # Find topmost cell
            start_y = y
            while start_y > 0 and (primed_mask >> ((start_y - 1) * BOARD_SIZE + x)) & 1:
                start_y -= 1
            run = 0
            cy = start_y
            while cy < BOARD_SIZE and (primed_mask >> (cy * BOARD_SIZE + x)) & 1:
                seen_v |= (1 << (cy * BOARD_SIZE + x))
                run += 1
                cy += 1
            run = min(run, 7)
            if run >= 2:
                total += CARPET_POINTS_TABLE[run]

    return total
