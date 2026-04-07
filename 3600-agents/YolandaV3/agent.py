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
        opponent_loc, opponent_found = board.opponent_search
        player_loc, player_found = board.player_search

        if player_found:
            self.belief = self.initial_belief.copy()
        elif player_loc is not None:
            player_index = self.loc_to_index(player_loc)
            self.belief[player_index] = 0 
            total = np.sum(self.belief)
            if total > 0:
                self.belief /= total

        if opponent_found:
            self.belief = self.initial_belief.copy()
        elif opponent_loc is not None:
            opponent_index = self.loc_to_index(opponent_loc)
            self.belief[opponent_index] = 0 
            total = np.sum(self.belief)
            if total > 0:
                self.belief /= total

        if self.transition_matrix is not None:
            self.belief = self.belief @ self.transition_matrix  # Update belief based on transition probabilities

        # Incorporate sensor data to update belief
        self.update_belief_with_sensor_data(sensor_data, board)
        best_index = np.argmax(self.belief)
        best_prob = self.belief[best_index]
        best_loc = self.index_to_loc(best_index)

        if best_prob > SEARCH_PROB_THRESHOLD:
            return Move.search(best_loc)

        candidate_moves = board.get_valid_moves()

        best_move = None
        best_score = float('-inf')
        current_points = board.player_worker.get_points()
        for move in candidate_moves:
            next_board = board.forecast_move(move)
            if next_board is None:
                continue
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


        # Opponent aware heuristic - penalize opponent having good mobility in the next state
        board.reverse_perspective()
        opponent_mobility_score = 0
        opponent_moves = board.get_valid_moves()
        for m in opponent_moves:
            if m.move_type == enums.MoveType.CARPET:
                if m.roll_length == 1:
                    opponent_mobility_score -= 1  # Penalize opponent using a carpet of length 1
                else:
                    opponent_mobility_score += 0.2
            if m.move_type == enums.MoveType.PRIME:
                opponent_mobility_score += 0.5  
            elif m.move_type == enums.MoveType.PLAIN:
                opponent_mobility_score += 0.1 

        score -= opponent_mobility_score * 0.3  # Penalize opponent mobility

        return score

    def update_belief_with_sensor_data(self, sensor_data: Tuple, board: board.Board):
            # Placeholder for updating belief based on sensor data
            # This function should incorporate the sensor readings to adjust the belief distribution over the opponent's location
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

                self.belief[i] = self.belief[i] * noise_prob * distance_prob  # Update the belief for position i based on the sensor data

            
            # Normalize the belief to ensure it sums to 1
            total = np.sum(self.belief)
            if total > 0:
                self.belief /= total
            pass
