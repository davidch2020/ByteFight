from collections.abc import Callable
from typing import List, Set, Tuple
import random
import time
import numpy as np

from game import board, move, enums, rat
from game.move import Move

SEARCH_PROB_THRESHOLD = 0.55
MINIMAX_DEPTH = 3


class PlayerAgent:
    """
    /you may add and modify functions, however, __init__, commentate and play are the entry points for
    your program and should not be changed.
    """

    # ----- Entry Points -----

    """
    Set up the rat transition model and our starting belief over rat locations.
    """
    def __init__(self, board, transition_matrix=None, time_left: Callable = None):

        """
        TODO: Your initialization code below. Should be used to do any setup you want
        before the game begins (i.e. calculating priors.)
        """
        self.transition_matrix = np.array(transition_matrix)
        self.initial_belief = np.zeros(64)  # Start with the assumption that the opponent is at position 0 (or any other starting position based on the game rules)
        self.initial_belief[0] = 1

        for _ in range(1000):
            self.initial_belief = self.initial_belief @ self.transition_matrix

        self.belief = self.initial_belief.copy()
        self.decision_nodes_visited = 0
        self.chance_nodes_visited = 0

        self.detected_time_budget = None

        pass

    """
    Optional hook for end-of-game text. We leave it empty.
    """
    def commentate(self):
        """
        Optional: You can use this function to print out any commentary you want at the end of the game.
        """
        return ""

    """
    Main turn function: update belief, search if confidence is high enough,
    otherwise choose the best movement move.
    """
    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        """
        TODO: Below is random mover code. Replace it with your own.
        You may do so however you like, including adding extra functions,
        variables. Return a valid move from this function.
        """
        turn_start = time.perf_counter()
        start_time_left = time_left()
        self.reset_search_counters()

        # First refresh our belief over rat locations from search history and sensor data.
        self.update_rat_belief(board, sensor_data)

        search_move = self.choose_search_move()
        if search_move is not None:
            elapsed = time.perf_counter() - turn_start
            end_time_left = time_left()
            print(
                f"[timing] turn action=SEARCH elapsed={elapsed:.3f}s "
                f"start_left={start_time_left:.2f}s end_left={end_time_left:.2f}s "
                f"top_belief={float(np.max(self.belief)):.3f}"
            )
            return search_move

        # If search is not confident enough, fall back to board-play heuristics.
        search_depth = self.choose_minimax_depth(time_left)
        print(f"[debug] minimax search depth={search_depth}")

        move = self.choose_best_movement_move(board, search_depth)
        elapsed = time.perf_counter() - turn_start
        end_time_left = time_left()
        print(
            f"[timing] turn action=MOVE depth={search_depth} elapsed={elapsed:.3f}s "
            f"start_left={start_time_left:.2f}s end_left={end_time_left:.2f}s "
            f"top_belief={float(np.max(self.belief)):.3f} "
            f"decision_nodes={self.decision_nodes_visited} chance_nodes={self.chance_nodes_visited}"
        )
        return move

    # ----- Belief Updates -----

    """
    Renormalize the belief vector so all probabilities add up to 1.
    """
    def normalize_belief(self):
        total = np.sum(self.belief)
        if total > 0:
            self.belief /= total

    """
    Update belief using recent search results, one rat transition step,
    and the latest sensor reading.
    """
    def update_rat_belief(self, board: board.Board, sensor_data: Tuple):
        opponent_loc, opponent_found = board.opponent_search
        player_loc, player_found = board.player_search

        # Any successful search respawns the rat, so belief resets to the spawn prior.
        if player_found:
            self.belief = self.initial_belief.copy()
        elif player_loc is not None:
            player_index = self.loc_to_index(player_loc)
            self.belief[player_index] = 0
            self.normalize_belief()

        if opponent_found:
            self.belief = self.initial_belief.copy()
        elif opponent_loc is not None:
            opponent_index = self.loc_to_index(opponent_loc)
            self.belief[opponent_index] = 0
            self.normalize_belief()

        # Predict one rat movement step before applying the new observation.
        if self.transition_matrix is not None:
            self.belief = self.belief @ self.transition_matrix

        self.update_belief_with_sensor_data(sensor_data, board)

    """
    Reweight each possible rat cell by how well it matches the current sensor data.
    """
    def update_belief_with_sensor_data(self, sensor_data: Tuple, board: board.Board):
        # Reweight each possible rat cell by how well it matches the observed noise and distance.
        worker_pos = board.player_worker.get_location()

        for i in range(len(self.belief)):
            cell_type = board.get_cell(self.index_to_loc(i))
            noise_prob = rat.NOISE_PROBS[cell_type][sensor_data[0]]  # Get the probability of observing the sensor data given the opponent is at position i

            manhattan_distance = abs(worker_pos[0] - self.index_to_loc(i)[0]) + abs(worker_pos[1] - self.index_to_loc(i)[1]) # Calculate the Manhattan distance from the worker to position i

            diff = sensor_data[1] - manhattan_distance  # Calculate the difference between the observed distance and the expected distance
            if diff < -1 or diff > 2:
                distance_prob = 0  # If the observed distance is not within the expected range, set the probability to 0
            else:
                distance_prob = rat.DISTANCE_ERROR_PROBS[diff + 1]  # Get the probability of observing the sensor data given the distance difference

            self.belief[i] = self.belief[i] * noise_prob * distance_prob

        self.normalize_belief()

    """
    Return a SEARCH move when one cell looks likely enough to justify it.
    """
    def choose_search_move(self):
        best_index = np.argmax(self.belief)
        best_prob = self.belief[best_index]
        best_loc = self.index_to_loc(best_index)

        # Search only when we are expected to gain enough points on average, and when we are reasonably confident about the rat's location.
        expected_value_search = (
            best_prob * enums.RAT_BONUS 
            + (1 - best_prob) * (-enums.RAT_PENALTY)
        )

        if expected_value_search > 0 and best_prob > 0.5:
            return Move.search(best_loc)

        return None

    # ----- Time And Top-Level Decisions -----

    """
    Lower the search depth when time is getting low.
    """
    def choose_minimax_depth(self, time_left: Callable[[], float]):
        # TODO: Use the remaining time budget to choose a safe search depth.
        seconds_left = time_left()
        if self.detected_time_budget is None:
            self.detected_time_budget = seconds_left

        total_time = self.detected_time_budget

        if seconds_left > total_time * 0.7:
            return MINIMAX_DEPTH
        elif seconds_left > total_time * 0.3:
            return max(1, MINIMAX_DEPTH - 1)
        else:
            return max(1, MINIMAX_DEPTH - 2)

    """
    Evaluate each legal movement move with the tree search and return the best one.
    """
    def choose_best_movement_move(self, board: board.Board, search_depth: int):
        move_start = time.perf_counter()
        candidate_moves = board.get_valid_moves()
        best_move = None
        best_score = float("-inf")
        current_points = board.player_worker.get_points()

        forecasts = self.get_ordered_forecasts(candidate_moves, strong_moves_first=True)

        for _, move in forecasts:
            next_board = board.forecast_move(move)
            if next_board is None:
                continue

            next_board.reverse_perspective()  # Switch perspective to evaluate from opponent's point of view

            move_value = self.chance_value(  # Evaluate move with rat uncertainty
                next_board,
                current_points=current_points,
                depth=search_depth,
                maximizing=False,
                belief=self.belief,
                alpha=float("-inf"),
                beta=float("inf"),
            )

            if move_value > best_score:  # Find the max score among the worst opponent replies for each candidate move, and choose the move that maximizes this score
                best_score = move_value
                best_move = move

        if best_move is not None:
            move_elapsed = time.perf_counter() - move_start
            print(
                f"[timing] choose_best_movement_move elapsed={move_elapsed:.3f}s "
                f"candidate_moves={len(candidate_moves)} best_score={best_score:.3f} "
                f"decision_nodes={self.decision_nodes_visited} chance_nodes={self.chance_nodes_visited}"
            )
            return best_move

        return random.choice(candidate_moves)

    # ----- Expectiminimax -----

    """
    Normal moves still forecast one step ahead
    Search moves go to a special search_action_value function that considers the possible outcomes of searching a cell with the current belief
    """
    def decision_value(self, board: board.Board, current_points: int, depth: int, maximizing: bool, belief: np.ndarray, alpha: float, beta: float) -> float:
        self.decision_nodes_visited += 1
        if depth == 0:
            if not maximizing:
                board.reverse_perspective()  # If board is still in opponent's perspective
            return self.evaluate_board(belief, board, current_points)  # flip to evaluate

        actions = self.get_valid_moves(board, belief, maximizing, depth)

        if not actions:
            if not maximizing:
                board.reverse_perspective()  # If board is still in opponent's perspective
            return self.evaluate_board(belief, board, current_points)  # flip to evaluate

        forecasts = self.get_ordered_forecasts(actions, strong_moves_first=True)

        if maximizing:
            value = float("-inf")
            for _, action in forecasts:
                if action.move_type == enums.MoveType.SEARCH:
                    child_value = self.search_action_value(
                        board,
                        action,
                        belief,
                        current_points,
                        depth,
                        next_maximizing=False,
                        alpha=alpha,
                        beta=beta,
                    )
                else:
                    child_board, child_belief = self.forecast_expecti_action(board, action, belief)

                    if child_board is None:
                        continue

                    child_board.reverse_perspective()  # Switch perspective to evaluate from opponent's point of view
                    child_value = self.chance_value(child_board, current_points, depth - 1, False, child_belief, alpha, beta)  # rat uncertainty after my move

                value = max(value, child_value)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # Beta cut-off
            return value

        value = float("inf")
        for _, action in forecasts:
            if action.move_type == enums.MoveType.SEARCH:
                child_value = self.search_action_value(
                    board,
                    action,
                    belief,
                    current_points,
                    depth,
                    next_maximizing=True,
                    alpha=alpha,
                    beta=beta,
                )
            else:
                child_board, child_belief = self.forecast_expecti_action(board, action, belief)

                if child_board is None:
                    continue

                child_board.reverse_perspective()  # Switch perspective to evaluate from opponent's point of view
                child_value = self.chance_value(child_board, current_points, depth - 1, True, child_belief, alpha, beta)

            value = min(value, child_value)
            beta = min(beta, value)
            if alpha >= beta:
                break  # Alpha cut-off
        return value

    """
    Handles randomness
    The rat might go to several places next.
    Let’s evaluate each possible rat outcome.
    Then average them using their probabilities.
    """
    def chance_value(self, board: board.Board, current_points: int, depth: int, maximizing: bool, belief: np.ndarray, alpha: float, beta: float) -> float:
        self.chance_nodes_visited += 1
        expected_value = 0
        outcomes = self.get_chance_outcomes(belief)
        for probability, next_belief in outcomes:
            expected_value += probability * self.decision_value(
                board.get_copy(),
                current_points,
                depth,
                maximizing,
                next_belief,
                alpha,
                beta,
            )
        return expected_value

    """
    Evaluate SEARCH inside one concrete rat-location branch.

    search_index = the cell we chose to search
    rat_index = the cell where the rat actually is in this branch

    If search_index == rat_index, the search hits for +4 and the rat respawns.
    Otherwise, the search misses for -2 and we rule out that searched cell.
    """
    def search_action_value(
        self,
        board: board.Board,
        action: Move,
        belief: np.ndarray,
        current_points: int,
        depth: int,
        next_maximizing: bool,
        alpha: float,
        beta: float,
    ) -> float:
        child_board = board.forecast_move(action)
        if child_board is None:
            return float("-inf") if not next_maximizing else float("inf")

        search_index = self.loc_to_index(action.search_loc)
        rat_index = int(np.argmax(belief))  # most likely rat cell in this branch, belief should be one-hot encoded
        next_belief = belief.copy()

        if search_index == rat_index:
            child_board.player_worker.increment_points(enums.RAT_BONUS)
            next_belief = self.initial_belief.copy()  # rat respawns, belief resets to initial
        else:
            child_board.player_worker.decrement_points(enums.RAT_PENALTY)
            next_belief[search_index] = 0  # we are certain the rat is not in the searched cell, set belief to 0
            total = np.sum(next_belief)
            if total > 0:
                next_belief /= total  # renormalize belief if there are still possible rat locations
            else:
                next_belief = self.initial_belief.copy()  # if no possible locations remain, reset to initial belief

        child_board.reverse_perspective()  # Switch perspective to evaluate from opponent's point of view
        return self.chance_value(child_board, current_points, depth - 1, next_maximizing, next_belief, alpha, beta)  # rat uncertainty after search action

    """
    Forecast a normal move and carry the current belief into the child state.
    """
    def forecast_expecti_action(
        self,
        board: board.Board,
        action: Move,
        belief: np.ndarray,
    ) -> Tuple[board.Board, np.ndarray]:
        """
        Simulate one chosen action
        For normal movement moves, just return the next board state and same belief (since we only update belief with new sensor data, not from movement outcomes)
        """
        next_board = board.forecast_move(action)
        if next_board is None:
            return None, belief

        next_belief = belief.copy()

        return next_board, next_belief

    # ----- Action Generation And Move Ordering -----

    """
    return the actions the tree is allowed to consider
    includes all normal movement moves
    includes one search action: search the cell with the highest belief of containing rat
    """
    def get_valid_moves(self, board: board.Board, belief: np.ndarray, maximizing: bool, depth: int) -> List[Move]:
        actions = board.get_valid_moves()  # normal movement moves

        best_index = int(np.argmax(belief))  # most likely rat cell
        best_loc = self.index_to_loc(best_index)
        
        # only consider searching when maximizing, near a leaf, and belief is concentrated - avoids branching too much
        #if maximizing and depth <= 1 and np.max(belief) > SEARCH_PROB_THRESHOLD:
        if np.max(belief) > SEARCH_PROB_THRESHOLD:
            actions.append(Move.search(best_loc)) # optional one search move for the most likely rat cell

        return actions

    """
    Give each move a cheap score so we can search stronger-looking moves first.
    """
    def get_ordered_forecasts(self, candidate_moves: List[Move], strong_moves_first: bool = True) -> List[Tuple[float, Move]]:
        forecasts = []
        for move in candidate_moves:
            score = self.move_order_score(move)
            forecasts.append((score, move))
        forecasts.sort(key=lambda x: x[0], reverse=strong_moves_first)
        return forecasts

    """
    Cheap move-ordering heuristic: prefer carpets, then primes, then plain moves.
    """
    def move_order_score(self, move):
        if move.move_type == enums.MoveType.CARPET:
            return 100 + move.roll_length
        if move.move_type == enums.MoveType.PRIME:
            return 50
        if move.move_type == enums.MoveType.PLAIN:
            return 0
        return -100

    """
    predict where the rat might go next
    keep only the top k most likely rat positions
    return each possibility as a tuple of
    probability: how likely the rat is to move to this position
    one-hot belief: the new belief if we assume the rat moved to this position with certainty
    """
    def get_chance_outcomes(self, belief: np.ndarray, k: int = 2) -> List[Tuple[float, np.ndarray]]:
        next_probs = belief @ self.transition_matrix
        top_indices = np.argsort(next_probs)[::-1][:k]
        outcomes = []
        total = float(sum(next_probs[i] for i in top_indices))

        # Safety check
        if total == 0:
            return [(1.0, belief.copy())]

        for i in top_indices:
            prob = float(next_probs[i] / total)
            outcomes.append((prob, self.one_hot(len(belief), int(i))))

        return outcomes

    # ----- Heuristic Evaluation -----

    """
    Score the board using points, carpet quality, mobility, and opponent threat.
    """
    def evaluate_board(self, belief: np.ndarray, board: board.Board, current_points: int) -> float:
        new_points = board.player_worker.get_points()

        # Calculate the score as the difference in points between the new board state and the current board state
        opponent_points = board.opponent_worker.get_points()

        # score = new_points - current_points

        score = new_points - opponent_points  # Evaluate the score based on the change in points relative to the opponent's points

        # Additional heuristics can be added here to evaluate the board state more effectively
        next_moves = board.get_valid_moves()
        carpet_count = 0
        carpet_value = 0
        largest_carpet_value = 0
        for move in next_moves:
            if move.move_type == enums.MoveType.CARPET:
                carpet_points = enums.CARPET_POINTS_TABLE[move.roll_length]
                if move.roll_length != 1:
                    carpet_count += 1

                carpet_value += carpet_points

                if carpet_points > largest_carpet_value:
                    largest_carpet_value = carpet_points

        score += carpet_count * 0.2  # Reward having more carpet moves available
        score += carpet_value * 0.1  # Reward having longer carpet moves available
        score += largest_carpet_value * 0.3  # Reward having the option for a high-value carpet move

        # Evaluate mobility for next state based on prime and plain moves
        mobility_score = 0
        for m in next_moves:
            if m.move_type == enums.MoveType.PRIME:
                mobility_score += 1
            elif m.move_type == enums.MoveType.PLAIN:
                mobility_score += 0.2

        score += mobility_score * 0.3  # Weight mobility score

        # Penalize score if opponent has strong move available
        opponent_board = board.get_copy()
        opponent_board.reverse_perspective()
        opponent_best_potential = 0
        for move in opponent_board.get_valid_moves():
            if move.move_type == enums.MoveType.CARPET:
                value = enums.CARPET_POINTS_TABLE[move.roll_length]
            elif move.move_type == enums.MoveType.PRIME:
                value = 1
            elif move.move_type == enums.MoveType.PLAIN:
                value = 0.2
            else:
                value = 0

            opponent_best_potential = max(opponent_best_potential, value)

        score -= opponent_best_potential * 0.1

        # a leaf where the rat if highly concentrated should score a bit better
        best_prob = np.max(belief)
        expected_value = best_prob * enums.RAT_BONUS + (1 - best_prob) * (-enums.RAT_PENALTY)
        score += expected_value * 0.5

        return score

    # ----- Utilities -----

    """
    Convert an (x, y) board location into a single 0..63 index.
    """
    def loc_to_index(self, loc: Tuple[int, int]) -> int:
        x, y = loc
        return y * enums.BOARD_SIZE + x

    """
    Convert a 0..63 index back into an (x, y) board location.
    """
    def index_to_loc(self, index: int) -> Tuple[int, int]:
        x = index % enums.BOARD_SIZE
        y = index // enums.BOARD_SIZE
        return (x, y)

    """
    Create a one-hot belief vector that puts all probability on one cell.
    """
    def one_hot(self, size: int, index: int) -> np.ndarray:
        vec = np.zeros(size)
        vec[index] = 1
        return vec

    """
    Reset search node counters at the start of each turn.
    """
    def reset_search_counters(self):
        self.decision_nodes_visited = 0
        self.chance_nodes_visited = 0
