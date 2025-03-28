from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.game.light_action import LightAction
from seahorse.game.game_layout.board import Piece

class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that makes random moves.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerDivercite instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type, name)
        # Add any information you want to store about the player here
        # self.json_additional_info = {}

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """
        return self.maxValue(state=current_state, alpha=float('-inf'), beta=float('inf'), depth=self.dynamic_depth(remaining_time, current_state.get_step()))[1]
    
    def dynamic_depth(self, remaining_time: int, step: int) -> int:
        """
        Calculate the depth of the minimax algorithm based on the remaining time and the step of the game.

        Args:
            remaining_time (int): The remaining time in seconds.
            step (int): The step of the game.

        Returns:
            int: The depth of the minimax algorithm.
        """
        dynamic_depth = ((remaining_time / max(1, 40 - step)) // 10) + 1
        return min(dynamic_depth, 4)
    
    def maxValue(self, state: GameStateDivercite, alpha: float, beta: float, depth: int) -> tuple[float, LightAction]:
        """
        Find the best action for the maximizing player.

        Args:
            state (GameStateDivercite): The current game state.
            alpha (float): The alpha value.
            beta (float): The beta value.
            depth (int): The depth of the minimax algorithm.

        Returns:
            tuple[float, LightAction]: The best value and action.
        """
        if state.is_done() or depth == 0:
            return self.evaluate_state(state), None

        value = float('-inf')
        best_action = None

        for action in state.generate_possible_light_actions():
            new_state = state.apply_action(action)
            new_value = self.minValue(new_state, alpha, beta, depth - 1)[0]

            if new_value > value:
                value = new_value
                best_action = action

            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return value, best_action
    
    def minValue(self, state: GameStateDivercite, alpha: float, beta: float, depth: int) -> tuple[float, LightAction]:
        """
        Find the best action for the minimizing player.

        Args:
            state (GameStateDivercite): The current game state.
            alpha (float): The alpha value.
            beta (float): The beta value.
            depth (int): The depth of the minimax algorithm.

        Returns:
            tuple[float, LightAction]: The best value and action.
        """
        if state.is_done() or depth == 0:
            return self.evaluate_state(state), None

        value = float('inf')
        best_action = None

        for action in state.generate_possible_light_actions():
            new_state = state.apply_action(action)
            new_value = self.maxValue(new_state, alpha, beta, depth - 1)[0]

            if new_value < value:
                value = new_value
                best_action = action

            beta = min(beta, value)
            if alpha >= beta:
                break

        return value, best_action
    
    def evaluate_state(self, state: GameStateDivercite) -> float:
        """
        Evaluate the game state based on the implemented heuristic function.

        Args:
            state (GameStateDivercite): The current game state.

        Returns:
            float: The value of the game state.
        """
        my_id = self.get_id()
        opponent_id = next(p.get_id() for p in state.players if p.get_id() != my_id)

        score_diff = state.scores[my_id] - state.scores[opponent_id]

        my_potential, my_urgent = self.analyze_divercite_potential(state, my_id)
        opponent_potential, opponent_urgent = self.analyze_divercite_potential(state, opponent_id)

        heuristic = (
            score_diff 
            + 1.8 * my_potential 
            - 2.5 * opponent_urgent  
        )

        remaining_turns_ratio = (state.max_step - state.get_step()) / state.max_step
        position_value = self.evaluate_board_control(state, my_id) * remaining_turns_ratio
        heuristic += position_value

        return heuristic
    
    def analyze_divercite_potential(self, state: GameState, player_id: int) -> tuple[int, int]:
        """
        Analyze the potential of a player in the Divercite game.

        Args:
            state (GameState): The current game state.
            player_id (int): The id of the player.

        Returns:
            tuple[int, int]: The potential and urgency of the player.
        """
        potential_count = 0
        urgent_count = 0
        board = state.get_rep().get_env()

        for (row, col), tile in board.items():
            if tile.get_owner_id() == player_id and tile.get_type()[1] == 'C':  
                adjacent_colors = set()
                empty_slots = []

                for _, neighbor in state.get_neighbours(row, col).items():
                    piece = neighbor[0]
                    if isinstance(piece, Piece):
                        adjacent_colors.add(piece.get_type()[0])
                    else:
                        empty_slots.append(neighbor[1])

                total_possible_colors = len(adjacent_colors) + len(empty_slots)

                if total_possible_colors >= 4:
                    potential_count += 1
                    if len(adjacent_colors) == 3 and empty_slots:
                        urgent_count += 1

        return potential_count, urgent_count  
    
    def evaluate_board_control(self, state: GameState, player_id: int) -> float:
        """
        Evaluate the control of the board by a player.

        Args:
            state (GameState): The current game state.
            player_id (int): The id of the player.

        Returns:
            float: The value of the board control.
        """
        influence_score = 0
        board_representation = state.get_rep()
        center_x, center_y = board_representation.get_dimensions()[0] // 2, board_representation.get_dimensions()[1] // 2
        board_state = board_representation.get_env()

        for (row, col), tile in board_state.items():
            if tile.get_owner_id() == player_id:
                proximity_bonus = max(0, 3 - (abs(row - center_x) + abs(col - center_y)))
                influence_score += proximity_bonus

                for neighbor in state.get_neighbours(row, col).values():
                    if isinstance(neighbor[0], Piece):
                        influence_score += 0.5

        return influence_score