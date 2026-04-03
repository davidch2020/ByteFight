from collections.abc import Callable
from typing import List, Set, Tuple
import random

from game import board, move, enums


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
        pass
        
    def commentate(self):
        """
        Optional: You can use this function to print out any commentary you want at the end of the game.
        """
        return ""

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
        candidate_moves = board.get_valid_moves()

        best_move = None
        best_score = float('-inf')
        for move in candidate_moves:
            next_board = board.forecast_move(move)
            if next_board is None:
                continue
            current_points = board.player_worker.get_points()
            score = self.evaluate_board(next_board, current_points)

            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move is not None:
            return best_move
        
        return random.choice(candidate_moves)

    def evaluate_board(self, board: board.Board, current_points: int) -> float:
        new_points = board.player_worker.get_points()

        # Calculate the score as the difference in points between the new board state and the current board state
        score = new_points - current_points

        # Additional heuristics can be added here to evaluate the board state more effectively
        next_moves = board.get_valid_moves()
        carpet_count = 0
        carpet_value = 0
        for move in next_moves:
            if move.move_type == enums.MoveType.CARPET:
                if move.roll_length == 1:
                    score -= 1  # Penalize using a carpet of length 1
                carpet_count += 1
                carpet_value += move.roll_length

        score += carpet_count * 0.2  # Reward having more carpet moves available
        score += carpet_value * 0.5  # Reward having longer carpet moves available
        
        # Evaluate mobility for next state based on prime and plain moves
        mobility_score = 0
        for m in next_moves:
            if m.move_type == enums.MoveType.PRIME:
                mobility_score += 1
            elif m.move_type == enums.MoveType.PLAIN:
                mobility_score += 0.2

        score += mobility_score * 0.3  # Weight mobility score

        return score


