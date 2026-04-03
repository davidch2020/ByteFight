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
        moves = board.get_valid_moves()

        best_carpet = None
        current_prime = None
        for move in moves:
            if move.move_type == enums.MoveType.PRIME:
                current_prime = move
            elif move.move_type == enums.MoveType.CARPET and move.roll_length != 1:
                if best_carpet is None or move.roll_length > best_carpet.roll_length:
                    best_carpet = move

        if best_carpet is not None:
            return best_carpet
        if current_prime is not None:
            return current_prime

        return random.choice(moves)
