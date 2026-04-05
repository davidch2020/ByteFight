from collections.abc import Callable
from typing import List, Set, Tuple
import random
import numpy as np

from game import board, move, enums, rat
from game.move import Move

SEARCH_PROB_THRESHOLD = 0.55


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
        return self.choose_best_movement_move(board)

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

    def choose_best_movement_move(self, board: board.Board):
        candidate_moves = board.get_valid_moves()
        best_move = None
        best_score = float('-inf')
        current_points = board.player_worker.get_points()

        for move in candidate_moves:
            next_board = board.forecast_move(move)
            if next_board is None:
                continue

            next_board.reverse_perspective()  # Switch perspective to evaluate from opponent's point of view

            move_value = self.minimax_value(
                next_board, 
                current_points, 
                depth=2, 
                maximizing_player=False
            )  # Look ahead 2 moves (1 for opponent, 1 for player) using minimax

            if move_value > best_score: # Find the max score among the worst opponent replies for each candidate move, and choose the move that maximizes this score
                best_score = move_value
                best_move = move
        
        if best_move is not None:
            return best_move

        return random.choice(candidate_moves)
    
    def minimax_value(self, board: board.Board, current_points: int, depth: int, maximizing_player: bool) -> float:
        if depth == 0:
            if maximizing_player == False:
                board.reverse_perspective()  # Switch perspective back to player's point of view before evaluating
            return self.evaluate_board(board, current_points)
        
        candidate_moves = board.get_valid_moves()
        if candidate_moves == []:
            if maximizing_player == False:
                board.reverse_perspective() 
            return self.evaluate_board(board, current_points)
        
        if maximizing_player:
            best_eval = float('-inf')
            for move in candidate_moves:
                next_board = board.forecast_move(move)
                if next_board is None:
                    continue
                next_board.reverse_perspective()  # Switch perspective to evaluate from opponent's point of view
                eval = self.minimax_value(next_board, current_points, depth - 1, False)
                best_eval = max(best_eval, eval)
            
            return best_eval
        else:
            best_eval = float('inf')
            for move in candidate_moves:
                next_board = board.forecast_move(move)
                if next_board is None:
                    continue
                next_board.reverse_perspective()  # Switch perspective to evaluate from opponent's point of view
                eval = self.minimax_value(next_board, current_points, depth - 1, True)
                best_eval = min(best_eval, eval)
            
            return best_eval


    def evaluate_board(self, board: board.Board, current_points: int) -> float:
        new_points = board.player_worker.get_points()

        # Calculate the score as the difference in points between the new board state and the current board state
        opponent_points = board.opponent_worker.get_points()

        # score = new_points - current_points

        score = new_points - opponent_points  # Evaluate the score based on the change in points relative to the opponent's points

        # Additional heuristics can be added here to evaluate the board state more effectively
        next_moves = board.get_valid_moves()
        carpet_count = 0
        carpet_value = 0
        largest_carpet_rolls = 0
        for move in next_moves:
            if move.move_type == enums.MoveType.CARPET:
                if move.roll_length == 1:
                    score -= 1  # Penalize using a carpet of length 1

                carpet_count += 1
                carpet_value += move.roll_length

                if move.roll_length > largest_carpet_rolls:
                   largest_carpet_rolls = move.roll_length
            

        score += carpet_count * 0.2  # Reward having more carpet moves available
        score += carpet_value * 0.1  # Reward having longer carpet moves available
        score += largest_carpet_rolls * 0.3  # Reward having the option for a long carpet move

        # Evaluate mobility for next state based on prime and plain moves
        mobility_score = 0
        for m in next_moves:
            if m.move_type == enums.MoveType.PRIME:
                mobility_score += 1
            elif m.move_type == enums.MoveType.PLAIN:
                mobility_score += 0.2

        score += mobility_score * 0.3  # Weight mobility score

        return score

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
