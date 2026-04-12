"""
YolandaAgent3 — Expectiminimax agent.

Architecture:
  1. HMM belief tracking for the rat (predict + sensor update each turn).
  2. Fixed-depth expectiminimax with alpha-beta pruning.
     - Max nodes: our turn — pick the best move (including rat search).
     - Min nodes: opponent's turn — assume they play optimally.
     - Chance nodes: rat search outcomes are probabilistic (hit/miss),
       so we take the weighted average over both branches.
  3. Simple heuristic: point margin + carpet opportunity + cell potential.
"""

from collections.abc import Callable
from typing import Tuple
import time
import numpy as np

from game import board, enums, rat
from game.move import Move

# --- Constants ---
BOARD_SIZE = enums.BOARD_SIZE          # 8
CPT = enums.CARPET_POINTS_TABLE       # {1: -1, 2: 2, 3: 4, 4: 6, 5: 10, 6: 15, 7: 21}
RAT_BONUS = enums.RAT_BONUS           # +4 for correct guess
RAT_PENALTY = enums.RAT_PENALTY       # -2 for wrong guess
RAT_NET_SWING = RAT_BONUS + RAT_PENALTY  # 6 (we gain 4, opponent loses the -2 threat)

SEARCH_DEPTH = 8                      # fixed expectiminimax depth
SEARCH_PROB_FLOOR = 0.1        # consider search at 10%+ belief (opponents search aggressively)
TIME_SAFETY = 1.5                     # stop searching if less than this many seconds left
SEARCH_INFO_BONUS = 3              # bonus for the strategic value of searching that the tree
                                      # can't model: even a miss concentrates belief for future turns,
                                      # and keeping pace with opponent searches is critical


class PlayerAgent:

    def __init__(self, board_obj, transition_matrix=None, time_left: Callable = None):
        self.T = np.array(transition_matrix, dtype=np.float64)

        # Rat spawn prior: placed at (0,0), runs 1000 steps via transition matrix
        self.belief = np.zeros(64, dtype=np.float64)
        self.belief[0] = 1.0
        for _ in range(1000):
            self.belief = self.belief @ self.T
        self.initial_belief = self.belief.copy()

    # ------------------------------------------------------------------
    # Main entry point — called once per turn
    # ------------------------------------------------------------------
    def play(self, b: board.Board, sensor_data: Tuple, time_left: Callable):
        # 1. Update rat belief with new sensor data
        self._update_belief(b, sensor_data)

        if self._should_search_mode(b):
            best_idx = int(np.argmax(self.belief))
            return Move.search(self._loc(best_idx))

        # 2. Run expectiminimax to pick the best move (movement OR search).
        #    Search moves are evaluated as chance nodes inside the tree,
        #    so there's no separate pre-tree search decision.
        best_move = self._root_search(b, time_left)
        if best_move is not None:
            return best_move

        # 3. Fallback: pick first valid move
        import random
        return random.choice(b.get_valid_moves())

    def commentate(self):
        return ""

    # ------------------------------------------------------------------
    # HMM belief tracking
    # ------------------------------------------------------------------
    def _update_belief(self, b: board.Board, sensor_data: Tuple):
        """Update rat probability distribution using search results + sensor data."""
        # Handle opponent's search result
        ol, of_ = b.opponent_search
        if of_:
            # Opponent caught the rat — rat respawned, reset to prior
            self.belief = self.initial_belief.copy()
        elif ol is not None:
            # Opponent searched but missed — rat is NOT at that location
            self.belief[self._idx(ol)] = 0.0
            self.belief = self._normalize(self.belief)

        # Handle our own search result
        pl, pf = b.player_search
        if pf:
            self.belief = self.initial_belief.copy()
        elif pl is not None:
            self.belief[self._idx(pl)] = 0.0
            self.belief = self._normalize(self.belief)

        # Predict: advance belief one step using transition matrix
        self.belief = self.belief @ self.T

        # Update: incorporate sensor data (noise type + distance estimate)
        wx, wy = b.player_worker.get_location()
        noise_type, dist_est = sensor_data

        for i in range(64):
            cx, cy = i % BOARD_SIZE, i // BOARD_SIZE
            cell = b.get_cell((cx, cy))

            # P(noise | cell type at rat's position)
            noise_prob = rat.NOISE_PROBS[cell][noise_type]

            # P(distance estimate | actual manhattan distance)
            actual_dist = abs(wx - cx) + abs(wy - cy)
            diff = dist_est - actual_dist  # should be in range [-1, 2]
            if -1 <= diff <= 2:
                dist_prob = rat.DISTANCE_ERROR_PROBS[diff + 1]
            else:
                dist_prob = 0.0

            self.belief[i] *= noise_prob * dist_prob

        self.belief = self._normalize(self.belief)

    # ------------------------------------------------------------------
    # Expectiminimax search
    # ------------------------------------------------------------------
    def _should_search_mode(self, b: board.Board) -> bool:
        prob = float(np.max(self.belief))
        turns = b.player_worker.turns_left

        # Aggressive search policy:
        # if the game is not almost over and we have even modest belief,
        # treat this as a search turn.
        if turns > 8 and prob >= 0.3:
            return True

        return False
    
    def _root_search(self, b: board.Board, time_left: Callable):
        """
        Root of the expectiminimax tree. Evaluates all movement moves AND
        a rat-search chance node, returning whichever Move scores highest.
        """
        moves = b.get_valid_moves()
        if not moves:
            return None

        moves = self._order_moves(moves)

        best_move = moves[0]
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        """
        # --- Chance node: evaluate searching for the rat ---
        # This is the "expecti" part — search has a probabilistic outcome,
        # so we branch into hit/miss and take the weighted average.
        # We check this even when only 1 movement move exists — searching
        # might be better than the only available board move.
        search_score = self._chance_node(b, SEARCH_DEPTH, alpha, beta, time_left)
        if search_score is not None:
            best_idx = int(np.argmax(self.belief))
            best_move = Move.search(self._loc(best_idx))
            best_score = search_score
            alpha = max(alpha, best_score)
        """

        # --- Max node: evaluate each deterministic movement move ---
        for m in moves:
            if time_left() < TIME_SAFETY:
                break
            nb = b.forecast_move(m)
            if nb is None:
                continue
            nb.reverse_perspective()
            score = self._expectiminimax(nb, SEARCH_DEPTH - 1, alpha, beta, False, time_left)
            if score > best_score:
                best_score = score
                best_move = m
            alpha = max(alpha, best_score)

        return best_move

    def _expectiminimax(self, b, depth, alpha, beta, maximizing, time_left):
        """
        Recursive expectiminimax with alpha-beta pruning.

        Node types:
          - Max (our turn):    pick the move with the highest value,
                               including a chance node for rat search.
          - Min (opponent):    assume opponent picks the move that
                               minimizes our value.
          - Chance (implicit): when a max node considers searching,
                               _chance_node() returns the expected value
                               over the hit/miss branches.
        """
        # Base case: leaf node — evaluate the board
        if depth <= 0 or time_left() < TIME_SAFETY:
            if not maximizing:
                b.reverse_perspective()
            return self._evaluate(b)

        moves = b.get_valid_moves()
        if not moves:
            if not maximizing:
                b.reverse_perspective()
            return self._evaluate(b)

        moves = self._order_moves(moves)

        if maximizing:
            best = float('-inf')

            # Chance node: consider searching for the rat
            """
            search_val = self._chance_node(b, depth, alpha, beta, time_left)
            if search_val is not None:
                best = max(best, search_val)
                if best >= beta:
                    return best
                alpha = max(alpha, best)
            """

            # Deterministic children: movement moves
            for m in moves:
                if time_left() < TIME_SAFETY:
                    break
                nb = b.forecast_move(m)
                if nb is None:
                    continue
                nb.reverse_perspective()
                val = self._expectiminimax(nb, depth - 1, alpha, beta, False, time_left)
                best = max(best, val)
                if best >= beta:
                    return best
                alpha = max(alpha, best)
            return best
        else:
            # Min node: opponent picks the worst move for us.
            # We don't model the opponent's rat search (we don't know their belief).
            best = float('inf')
            for m in moves:
                if time_left() < TIME_SAFETY:
                    break
                nb = b.forecast_move(m)
                if nb is None:
                    continue
                nb.reverse_perspective()
                val = self._expectiminimax(nb, depth - 1, alpha, beta, True, time_left)
                best = min(best, val)
                if best <= alpha:
                    return best
                beta = min(beta, best)
            return best

    def _chance_node(self, b, depth, alpha, beta, time_left):
        """
        Chance node for rat search — the core of expectiminimax.

        A search move has two possible outcomes:
          - Hit  (prob p):   we gain RAT_BONUS (+4) points, rat respawns
          - Miss (prob 1-p): we lose RAT_PENALTY (-2) points

        Returns the expected value: p * V(hit) + (1-p) * V(miss) + info_bonus,
        or None if the belief is too low to bother searching.

        The info bonus accounts for strategic value the tree can't model:
        even a miss eliminates a cell from the belief, concentrating probability
        and making future searches more accurate. Opponents that search
        aggressively (~10/game) consistently outperform passive searchers.
        """
        best_idx = int(np.argmax(self.belief))
        prob = self.belief[best_idx]

        # Don't waste tree search time on very low probability searches
        if prob < SEARCH_PROB_FLOOR:
            return None

        if time_left() < TIME_SAFETY:
            return None

        # --- Hit branch: we found the rat, gain +4 points ---
        hit_board = b.get_copy()
        hit_board.player_worker.increment_points(RAT_BONUS)
        hit_board.end_turn()              # uses our turn
        hit_board.reverse_perspective()   # now it's opponent's turn
        hit_val = self._expectiminimax(hit_board, depth - 1, alpha, beta, False, time_left)

        # --- Miss branch: wrong guess, lose -2 points ---
        miss_board = b.get_copy()
        miss_board.player_worker.increment_points(-RAT_PENALTY)
        miss_board.end_turn()
        miss_board.reverse_perspective()
        miss_val = self._expectiminimax(miss_board, depth - 1, alpha, beta, False, time_left)

        # --- Expected value: weighted average + information bonus ---
        # The bonus compensates for the tree's inability to model how
        # searching concentrates belief for future turns (information gain).
        # even a miss is valuable for that reason, and opponents that search aggressively consistently outperform passive searchers.
        return prob * hit_val + (1.0 - prob) * miss_val + SEARCH_INFO_BONUS

    # ------------------------------------------------------------------
    # Move ordering (helps alpha-beta prune faster)
    # ------------------------------------------------------------------
    def _order_moves(self, moves):
        """Sort: carpet by points (desc) > prime > plain."""
        def key(m):
            if m.move_type == enums.MoveType.CARPET:
                return (3, CPT.get(m.roll_length, 0))
            if m.move_type == enums.MoveType.PRIME:
                return (2, 0)
            if m.move_type == enums.MoveType.PLAIN:
                return (1, 0)
            return (0, 0)
        return sorted(moves, key=key, reverse=True)

    # ------------------------------------------------------------------
    # Heuristic evaluation
    # ------------------------------------------------------------------
    def _evaluate(self, b: board.Board) -> float:
        """
        Score a board state from the player's perspective.

        Components:
          1. Point margin (most important — this is what wins)
          2. Best available carpet (immediate scoring opportunity)
          3. Cell potential (Carrie-style: value of nearby primed cells)
          4. Opponent threat (penalize if opponent has big carpets available)
          5. Game-phase scaling (value points more and potential less as game ends)
        """
        my_pts = b.player_worker.get_points()
        opp_pts = b.opponent_worker.get_points()
        my_loc = b.player_worker.get_location()
        opp_loc = b.opponent_worker.get_location()
        turns = b.player_worker.turns_left

        # Game phase: 0.0 at start, 1.0 at end
        urgency = 1.0 - turns / enums.MAX_TURNS_PER_PLAYER

        score = 0.0

        # --- 1. Point margin ---
        margin = my_pts - opp_pts
        # Weight margin more heavily as game progresses
        score += margin * (1.0 + 0.5 * urgency)

        # --- 2. Available carpet moves ---
        moves = b.get_valid_moves()
        best_carpet = 0
        carpet_sum = 0
        n_prime = 0

        for m in moves:
            if m.move_type == enums.MoveType.CARPET:
                pts = CPT[m.roll_length]
                if pts > best_carpet:
                    best_carpet = pts
                if pts > 0:
                    carpet_sum += pts
            elif m.move_type == enums.MoveType.PRIME:
                n_prime += 1

        # Immediate scoring opportunity
        score += best_carpet * 0.6
        # Total carpet value available
        score += carpet_sum * 0.1
        # Having prime moves available = future potential (less valuable late)
        score += n_prime * 0.1 * (1.0 - 0.5 * urgency)

        # --- 3. Cell potential (nearby primed cells = future carpet value) ---
        # For each primed cell near us, estimate what carpet run it belongs to
        pot = self._local_potential(b, my_loc)
        # Discount future potential as game ends
        score += pot * 0.08 * (1.0 - 0.6 * urgency)

        # --- 4. Opponent carpet threat ---
        opp_best = self._best_carpet_from(b, opp_loc)
        score -= opp_best * 0.2

        # --- 5. Endgame: strongly favor converting carpets ---
        if turns <= 15:
            t = (16 - turns) / 15.0
            score += best_carpet * 0.4 * t
            score += margin * 0.25 * t

        # Note: no rat-search bonus here — the tree evaluates search moves
        # as chance nodes directly, so adding EV here would double-count.

        return score

    # ------------------------------------------------------------------
    # Heuristic helpers
    # ------------------------------------------------------------------
    def _local_potential(self, b: board.Board, loc: Tuple[int, int]) -> float:
        """
        Sum the carpet-run potential of primed cells within manhattan distance 3
        of the given location. Each primed cell's value = length of the longest
        contiguous primed run through it (in any cardinal direction).
        """
        px, py = loc
        total = 0.0

        for dy in range(-3, 4):
            ny = py + dy
            if ny < 0 or ny >= BOARD_SIZE:
                continue
            for dx in range(-(3 - abs(dy)), 4 - abs(dy)):
                nx = px + dx
                if nx < 0 or nx >= BOARD_SIZE:
                    continue
                if b.get_cell((nx, ny)) != enums.Cell.PRIMED:
                    continue
                # Count longest run through (nx, ny) in cardinal directions
                best_run = 1
                for ddx, ddy in ((1, 0), (0, 1)):
                    run = 1
                    # Extend forward
                    fx, fy = nx + ddx, ny + ddy
                    while 0 <= fx < BOARD_SIZE and 0 <= fy < BOARD_SIZE and b.get_cell((fx, fy)) == enums.Cell.PRIMED:
                        run += 1
                        fx += ddx
                        fy += ddy
                    # Extend backward
                    fx, fy = nx - ddx, ny - ddy
                    while 0 <= fx < BOARD_SIZE and 0 <= fy < BOARD_SIZE and b.get_cell((fx, fy)) == enums.Cell.PRIMED:
                        run += 1
                        fx -= ddx
                        fy -= ddy
                    if run > best_run:
                        best_run = run

                # Look up carpet points for this run length
                pts = CPT.get(min(best_run, 7), 0)
                if pts > 0:
                    dist = abs(dy) + abs(dx)
                    total += pts / (1.0 + 0.4 * dist)

        return total

    def _best_carpet_from(self, b: board.Board, loc: Tuple[int, int]) -> float:
        """Max carpet run a player at `loc` could roll right now."""
        ox, oy = loc
        best = 0.0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = ox + dx, oy + dy
            run = 0
            while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and b.get_cell((nx, ny)) == enums.Cell.PRIMED:
                run += 1
                nx += dx
                ny += dy
            if run >= 2:
                pts = CPT[min(run, 7)]
                if pts > best:
                    best = pts
        return best

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _idx(self, loc):
        """(x, y) -> flat index 0-63."""
        return loc[1] * BOARD_SIZE + loc[0]

    def _loc(self, idx):
        """Flat index 0-63 -> (x, y)."""
        return (idx % BOARD_SIZE, idx // BOARD_SIZE)

    def _normalize(self, b):
        """Normalize belief to sum to 1, or reset to prior if all zeros."""
        s = b.sum()
        return b / s if s > 0 else self.initial_belief.copy()
