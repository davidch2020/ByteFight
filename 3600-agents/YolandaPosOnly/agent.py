from collections.abc import Callable
from typing import List, Set, Tuple
import random
import numpy as np

from game import board, move, enums, rat
from game.move import Move

SEARCH_PROB_THRESHOLD = 0.55
MINIMAX_DEPTH = 5


class PlayerAgent:
    """
    /you may add and modify functions, however, __init__, commentate and play are the entry points for
    your program and should not be changed.
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
        self.detected_time_budget = None

        pass

    def commentate(self):
        """
        Optional: You can use this function to print out any commentary you want at the end of the game.
        """
        return ""

    def loc_to_index(self, loc: Tuple[int, int]) -> int:
        x, y = loc
        return y * enums.BOARD_SIZE + x

    def index_to_loc(self, index: int) -> Tuple[int, int]:
        x = index % enums.BOARD_SIZE
        y = index // enums.BOARD_SIZE
        return (x, y)

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
        # First refresh our belief over rat locations from search history and sensor data.
        self.update_rat_belief(board, sensor_data)
        search_move = self.choose_search_move()
        if search_move is not None:
            return search_move

        # If search is not confident enough, fall back to board-play heuristics.
        search_depth = self.choose_minimax_depth(time_left)
        return self.choose_best_movement_move(board, search_depth)

    def normalize_belief(self):
        total = np.sum(self.belief)
        if total > 0:
            self.belief /= total

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

    def choose_search_move(self):
        best_index = np.argmax(self.belief)
        best_prob = self.belief[best_index]
        best_loc = self.index_to_loc(best_index)

        # Search only when one location is confident enough to be worth the risk.
        if best_prob > SEARCH_PROB_THRESHOLD:
            return Move.search(best_loc)

        return None

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

    def move_order_score(self, move):
        if move.move_type == enums.MoveType.CARPET:
            return 100 + move.roll_length
        if move.move_type == enums.MoveType.PRIME:
            return 50
        if move.move_type == enums.MoveType.PLAIN:
            return 0
        return -100

    def get_ordered_forecasts(self, candidate_moves: List[Move], strong_moves_first: bool = True) -> List[Tuple[float, Move]]:
        forecasts = []
        for move in candidate_moves:
            score = self.move_order_score(move)
            forecasts.append((score, move))
        forecasts.sort(key=lambda x: x[0], reverse=strong_moves_first)
        return forecasts

    def choose_best_movement_move(self, board: board.Board, search_depth: int):
        candidate_moves = board.get_valid_moves()
        best_move = None
        best_score = float('-inf')
        current_points = board.player_worker.get_points()

        forecasts = self.get_ordered_forecasts(candidate_moves, strong_moves_first=True)

        for score, move in forecasts:
            next_board = board.forecast_move(move)
            if next_board is None:
                continue

            next_board.reverse_perspective()  # Switch perspective to evaluate from opponent's point of view

            move_value = self.minimax_value(
                next_board,
                alpha = float('-inf'),
                beta = float('inf'),
                current_points=current_points,
                depth=search_depth,
                maximizing_player=False
            )  # Look ahead 2 moves (1 for opponent, 1 for player) using minimax

            if move_value > best_score: # Find the max score among the worst opponent replies for each candidate move, and choose the move that maximizes this score
                best_score = move_value
                best_move = move

        if best_move is not None:
            return best_move

        return random.choice(candidate_moves)

    def minimax_value(self, board: board.Board, alpha:float, beta:float, current_points: int, depth: int, maximizing_player: bool) -> float:
        if depth == 0:
            if maximizing_player == False:
                board.reverse_perspective()  # Switch perspective back to player's point of view before evaluating
            return self.evaluate_board(board, current_points)

        candidate_moves = board.get_valid_moves()
        if candidate_moves == []:
            if maximizing_player == False:
                board.reverse_perspective()
            return self.evaluate_board(board, current_points)

        # Order strong-looking current-player moves first to improve alpha-beta pruning.
        forecasts = self.get_ordered_forecasts(candidate_moves, strong_moves_first=True)

        if maximizing_player:
            best_eval = float('-inf')
            for score, move in forecasts:
                next_board = board.forecast_move(move)
                if next_board is None:
                    continue
                next_board.reverse_perspective()  # Switch perspective to evaluate from opponent's point of view

                eval = self.minimax_value(next_board, alpha, beta, current_points, depth - 1, False)
                best_eval = max(best_eval, eval)
                if best_eval >= beta:
                    return best_eval  # Beta cut-off

                alpha = max(alpha, best_eval)

            return best_eval
        else:
            best_eval = float('inf')
            for score, move in forecasts:
                next_board = board.forecast_move(move)
                if next_board is None:
                    continue
                next_board.reverse_perspective()  # Switch perspective to evaluate from opponent's point of view

                eval = self.minimax_value(next_board, alpha, beta, current_points, depth - 1, True)
                best_eval = min(best_eval, eval)
                if best_eval <= alpha:
                    return best_eval  # Alpha cut-off

                beta = min(beta, best_eval)

            return best_eval

    def evaluate_board(self, board: board.Board, current_points: int) -> float:        
        return self.heuristic_breakdown(board)["score"]

    def heuristic_breakdown(self, board: board.Board) -> dict[str, float]:
        new_points = board.player_worker.get_points()
        opponent_points = board.opponent_worker.get_points()
        points_score = new_points - opponent_points

        # Evaluate position using the value board
        my_pos = board.player_worker.get_location()    
        opp_pos = board.opponent_worker.get_location()
        value_board = self.build_value_board(board)
        position_score = self.position_value(value_board, my_pos) * 0.25 - self.position_value(value_board, opp_pos) * 0.1

        score = (
            points_score
            + position_score
        )

        return {
            "score": score,
            "points": points_score,
            "position": position_score,
        }
    
    def build_value_board(self, board: board.Board) -> List[List[float]]:
        values = []
        for y in range(enums.BOARD_SIZE):
            row = []
            for x in range(enums.BOARD_SIZE):
                cell_value = self.cell_potential(board, (x, y))
                row.append(cell_value)
            values.append(row)
        return values

    def cell_potential(self, board: board.Board, loc: Tuple[int, int]) -> float:
        if board.is_cell_blocked(loc):
            return 0
        
        worker_x, worker_y = board.player_worker.get_location()
        opp_x, opp_y = board.opponent_worker.get_location()

        x, y = loc

        # Prefer cells that are closer to us
        my_dist = abs(worker_x - x) + abs(worker_y - y) # manhattan distance from worker to cell
        opp_dist = abs(opp_x - x) + abs(opp_y - y) # manhattan distance from opponent to cell

        # count how much open lane extends from this cell in each direction
        longest_ray = 0
        total_open_ray = 0
        prime_on_rays = 0
        
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for dx, dy in directions:
            ray_len = 0
            nx, ny = x, y

            while True:
                nx += dx
                ny += dy
                next_loc = (nx, ny)

                if not board.is_valid_cell(next_loc):
                    break # Stop if we go off the board
                if board.get_cell(next_loc) == enums.Cell.BLOCKED:
                    break # Stop if we hit a blocked cell
                if next_loc == board.player_worker.get_location():
                    break # Stop if we hit our own worker
                if next_loc == board.opponent_worker.get_location():
                    break # Stop if we hit the opponent worker

                ray_len += 1

                if board.get_cell(next_loc) == enums.Cell.PRIMED:
                    prime_on_rays += 1 # Count how many primed cells are on the rays extending from this cell, since they can boost carpet points

            longest_ray = max(longest_ray, ray_len)
            total_open_ray += ray_len # We want cells with long rays of open space, since they can support longer carpets

        score = 0
        score += longest_ray * 0.8
        score += total_open_ray * 0.2
        score += prime_on_rays * 0.5
        score -= my_dist * 0.2
        score += max(0, opp_dist - my_dist) * 0.2 # Prefer cells that are closer to us than the opponent

        return score

    def position_value(self, value_board: List[List[float]], loc: Tuple[int, int]) -> float:
        # blocked cells are randomized so we have to check
        x, y = loc
        return value_board[y][x]
    
    def best_carpet_points_future(self, board: board.Board) -> int:
        best_future_points = 0
        for move in board.get_valid_moves():
            if move.move_type == enums.MoveType.CARPET:
                continue  # Skip current carpet moves since they are already evaluated in best_carpet_points_now

            forecast_board = board.forecast_move(move)

            if forecast_board is None:
                continue

            future_points = self.best_carpet_points_now(forecast_board)
            best_future_points = max(best_future_points, future_points)

        return best_future_points

    
    def best_carpet_points_now(self, board: board.Board) -> int:
        best_points = 0
        for move in board.get_valid_moves():
            if move.move_type == enums.MoveType.CARPET:
                carpet_points = enums.CARPET_POINTS_TABLE[move.roll_length]
                best_points = max(best_points, carpet_points)

        return best_points

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
