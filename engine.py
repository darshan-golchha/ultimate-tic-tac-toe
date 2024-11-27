import math
import random
import numpy as np
from typing import List, Tuple, Dict, Any

class SuperTicTacToeAI:
    def __init__(self):
        # Initialize the 3x3 board of 3x3 sub-boards
        self.board = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(9)]
        # Track which sub-boards are won
        self.sub_board_states = [0 for _ in range(9)]
        # Track the next allowed board to play in
        self.next_board = None
        # Player 1 is maximizer, Player -1 is minimizer
        self.current_player = 1
        # Track move history for learning
        self.move_history = []

    def is_valid_move(self, global_board: int, local_board: int) -> bool:
        """
        Check if a move is valid based on game rules
        """
        # If no specific next board is set, any board is valid
        if self.next_board is None:
            return self.board[global_board][local_board // 3][local_board % 3] == 0
        
        # Must play in the specified next board
        return (global_board == self.next_board and 
                self.board[global_board][local_board // 3][local_board % 3] == 0)

    def make_move(self, global_board: int, local_board: int, player: int) -> bool:
        """
        Make a move on the board
        """
        if not self.is_valid_move(global_board, local_board):
            return False
        
        # Make the move
        self.board[global_board][local_board // 3][local_board % 3] = player
        
        # Track move history for learning
        self.move_history.append((global_board, local_board, player))
        
        # Check if sub-board is won
        if self._check_sub_board_win(global_board):
            self.sub_board_states[global_board] = player
        
        # Determine next allowed board
        next_local_board = local_board % 9
        
        # If that board is already won or full, next board is None (free choice)
        if self.sub_board_states[next_local_board] != 0 or self._is_sub_board_full(next_local_board):
            self.next_board = None
        else:
            self.next_board = next_local_board
        
        # Switch player
        self.current_player *= -1
        
        return True

    def _check_sub_board_win(self, board_index: int) -> bool:
        """
        Check if a specific sub-board is won
        """
        sub_board = self.board[board_index]
        
        # Check rows
        for row in sub_board:
            if abs(sum(row)) == 3:
                return True
        
        # Check columns
        for col in range(3):
            if abs(sum(sub_board[row][col] for row in range(3))) == 3:
                return True
        
        # Check diagonals
        if abs(sum(sub_board[i][i] for i in range(3))) == 3:
            return True
        if abs(sum(sub_board[i][2-i] for i in range(3))) == 3:
            return True
        
        return False

    def _is_sub_board_full(self, board_index: int) -> bool:
        """
        Check if a sub-board is completely filled
        """
        return all(self.board[board_index][row][col] != 0 
                   for row in range(3) for col in range(3))

    def check_global_win(self) -> int:
        """
        Check if the global board is won
        """
        # Check rows
        for i in range(0, 9, 3):
            if (self.sub_board_states[i] == self.sub_board_states[i+1] == 
                self.sub_board_states[i+2] != 0):
                return self.sub_board_states[i]
        
        # Check columns
        for i in range(3):
            if (self.sub_board_states[i] == self.sub_board_states[i+3] == 
                self.sub_board_states[i+6] != 0):
                return self.sub_board_states[i]
        
        # Check diagonals
        if (self.sub_board_states[0] == self.sub_board_states[4] == 
            self.sub_board_states[8] != 0):
            return self.sub_board_states[0]
        
        if (self.sub_board_states[2] == self.sub_board_states[4] == 
            self.sub_board_states[6] != 0):
            return self.sub_board_states[2]
        
        return 0  # No winner yet

    def get_possible_moves(self) -> List[Tuple[int, int]]:
        """
        Get all possible moves considering game rules
        """
        moves = []
        
        # If next board is specified
        if self.next_board is not None:
            for local in range(9):
                if self.board[self.next_board][local // 3][local % 3] == 0:
                    moves.append((self.next_board, local))
            return moves
        
        # Otherwise, check all boards
        for global_board in range(9):
            # Skip if sub-board is already won
            if self.sub_board_states[global_board] != 0:
                continue
            
            for local in range(9):
                if self.board[global_board][local // 3][local % 3] == 0:
                    moves.append((global_board, local))
        
        return moves

    def evaluate_heuristic(self) -> float:
        """
        Strategically balanced heuristic evaluation"""
        # Terminal state check
        global_win = self.check_global_win()
        if global_win == 1:
            return 10000  # Absolute AI win
        elif global_win == -1:
            return -10000  # Absolute opponent win

        # Balanced scoring components
        global_control_score = self._evaluate_global_board_control() * 40
        threat_score = self._evaluate_threats() * 35
        sub_board_score = self._evaluate_sub_board_control() * 20
        mobility_score = self._evaluate_move_flexibility() * 5

        # Combine scores
        score = global_control_score + threat_score + sub_board_score + mobility_score

        # Player perspective bias
        return score * (1 if self.current_player == 1 else -1)

    def _evaluate_global_board_control(self) -> float:
        """Comprehensive global board control assessment"""
        global_score = 0

        # Prioritized board control
        control_hierarchy = [
            (4, 100),        # Center board critical
            ((0, 2, 6, 8), 50),  # Corner boards significant
            ((1, 3, 5, 7), 25)   # Side boards moderate
        ]

        for boards, base_score in control_hierarchy:
            boards = [boards] if isinstance(boards, int) else boards
            for board in boards:
                if self.sub_board_states[board] == 1:
                    global_score += base_score
                elif self.sub_board_states[board] == -1:
                    global_score -= base_score

        # Global win path analysis
        win_paths = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],    # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],    # Columns
            [0, 4, 8], [2, 4, 6]                # Diagonals
        ]

        for path in win_paths:
            ai_boards = sum(1 for board in [self.sub_board_states[p] for p in path] if board == 1)
            opp_boards = sum(1 for board in [self.sub_board_states[p] for p in path] if board == -1)
            
            # Near-global win path bonuses
            if ai_boards == 2 and opp_boards == 0:
                global_score += 200
            if opp_boards == 2 and ai_boards == 0:
                global_score -= 200

        return global_score

    def _evaluate_threats(self) -> float:
        """Enhanced multi-level threat detection"""
        threat_score = 0
        
        for global_board in range(9):
            if self.sub_board_states[global_board] == 0:
                sub_board = self.board[global_board]
                
                # Comprehensive line threat analysis
                lines = [
                    [sub_board[0][0], sub_board[0][1], sub_board[0][2]],
                    [sub_board[1][0], sub_board[1][1], sub_board[1][2]],
                    [sub_board[2][0], sub_board[2][1], sub_board[2][2]],
                    [sub_board[0][0], sub_board[1][0], sub_board[2][0]],
                    [sub_board[0][1], sub_board[1][1], sub_board[2][1]],
                    [sub_board[0][2], sub_board[1][2], sub_board[2][2]],
                    [sub_board[0][0], sub_board[1][1], sub_board[2][2]],
                    [sub_board[0][2], sub_board[1][1], sub_board[2][0]]
                ]
                
                for line in lines:
                    player_count = line.count(1)
                    opponent_count = line.count(-1)
                    empty_count = line.count(0)
                    
                    # Aggressive threat scoring
                    if player_count == 2 and empty_count == 1:
                        threat_score += 50  # Strong winning potential
                    if opponent_count == 2 and empty_count == 1:
                        threat_score -= 50  # Strong blocking need

        return threat_score

    def _evaluate_sub_board_control(self) -> float:
        sub_board_score = 0
        for board in range(9):
            # More nuanced sub-board evaluation
            if self.sub_board_states[board] == 1:
                sub_board_score += 40  # Higher reward for controlling sub-board
            elif self.sub_board_states[board] == -1:
                sub_board_score -= 40
            else:
                # Refined potential assessment
                board_potential = self._evaluate_sub_board_potential(board)
                sub_board_score += board_potential * (1 if self.current_player == 1 else -1)
        return sub_board_score

    def _evaluate_sub_board_potential(self, board_index: int) -> float:
        sub_board = self.board[board_index]
        score = 0

        # Strategic position weightings
        position_weights = {
            (1,1): 25,   # Center most critical
            (0,0): 15, (0,2): 15, (2,0): 15, (2,2): 15,  # Corners significant
            (0,1): 10, (1,0): 10, (1,2): 10, (2,1): 10   # Sides moderate
        }

        # Apply position weights
        for (x,y), weight in position_weights.items():
            if sub_board[x][y] == 1:
                score += weight
            elif sub_board[x][y] == -1:
                score -= weight

        # Line potential analysis
        lines = [
            [sub_board[0][0], sub_board[0][1], sub_board[0][2]],
            [sub_board[1][0], sub_board[1][1], sub_board[1][2]],
            [sub_board[2][0], sub_board[2][1], sub_board[2][2]],
            [sub_board[0][0], sub_board[1][0], sub_board[2][0]],
            [sub_board[0][1], sub_board[1][1], sub_board[2][1]],
            [sub_board[0][2], sub_board[1][2], sub_board[2][2]],
            [sub_board[0][0], sub_board[1][1], sub_board[2][2]],
            [sub_board[0][2], sub_board[1][1], sub_board[2][0]]
        ]

        for line in lines:
            player_count = line.count(1)
            opponent_count = line.count(-1)
            empty_count = line.count(0)

            # Scoring for potential lines
            if player_count == 2 and empty_count == 1:
                score += 30  # Strong winning potential
            if opponent_count == 2 and empty_count == 1:
                score -= 30  # Strong blocking need

        return score

    def _evaluate_move_flexibility(self) -> float:
        """
        Refined move flexibility assessment
        """
        flexibility_score = 0
        
        # Weighted move count
        possible_moves = len(self.get_possible_moves())
        flexibility_score += possible_moves * 3

        # Strategic board restriction penalty
        if self.next_board is not None:
            flexibility_score -= 15

        # Bonus for open board states
        if self.next_board is None:
            flexibility_score += 10

        return flexibility_score

    def _check_board_symmetry(self) -> float:
        """
        Enhanced symmetry and pattern detection
        """
        symmetry_score = 0

        # More comprehensive symmetry patterns
        symmetry_patterns = [
            ((0, 2), 15),   # Horizontal symmetry
            ((6, 8), 15),   # Horizontal symmetry
            ((0, 6), 12),   # Vertical symmetry
            ((2, 8), 12),   # Vertical symmetry
            ((0, 8), 20),   # Diagonal symmetry
            ((2, 6), 20)    # Diagonal symmetry
        ]

        for (a, b), bonus in symmetry_patterns:
            if self.sub_board_states[a] == self.sub_board_states[b] and self.sub_board_states[a] != 0:
                symmetry_score += bonus if self.sub_board_states[a] == 1 else -bonus

        # Center symmetry with higher significance
        if self.sub_board_states[4] != 0:
            symmetry_score += 25 if self.sub_board_states[4] == 1 else -25

        return symmetry_score

    def get_best_move(self, depth: int = 5) -> Tuple[int, int]:
        """
        Find the best move using minimax
        """
        best_value = -math.inf
        best_move = None
        
        moves = self.get_possible_moves()
        
        for move in moves:
            # Create a deep copy of the game state
            game_copy = self._deep_copy()
            game_copy.make_move(move[0], move[1], 1)
            
            # Compute minimax value
            move_value = game_copy.minimax(depth - 1, False)
            print(f"Move: {move}, Value: {move_value}")
            
            if move_value > best_value:
                best_value = move_value
                best_move = move
        
        return best_move if best_move else moves[0]
    
    def minimax(self, depth: int, is_maximizing: bool, alpha: float = -math.inf, beta: float = math.inf) -> float:
        """
        Minimax algorithm with alpha-beta pruning and advanced heuristic evaluation
        """
        # Check terminal states first
        global_win = self.check_global_win()
        if global_win != 0:
            return global_win * 1000

        # Reached max depth
        if depth == 0:
            return self.evaluate_heuristic()

        moves = self.get_possible_moves()
        
        if is_maximizing:
            max_eval = -math.inf
            for move in moves:
                # Create a deep copy of the game state
                game_copy = self._deep_copy()
                game_copy.make_move(move[0], move[1], 1)
                
                eval = game_copy.minimax(depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in moves:
                # Create a deep copy of the game state
                game_copy = self._deep_copy()
                game_copy.make_move(move[0], move[1], -1)
                
                eval = game_copy.minimax(depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                
                if beta <= alpha:
                    break
            return min_eval

    def _deep_copy(self):
        """
        Create a deep copy of the game state using numpy for more efficient copying
        """
        new_game = SuperTicTacToeAI()
        new_game.board = np.array(self.board).copy().tolist()
        new_game.sub_board_states = np.array(self.sub_board_states).copy().tolist()
        new_game.next_board = self.next_board
        new_game.current_player = self.current_player
        new_game.move_history = self.move_history.copy()
        return new_game
    
    def reset(self):
        """
        Reset the game state
        """
        self.board = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(9)]
        self.sub_board_states = [0 for _ in range(9)]
        self.next_board = None
        self.current_player = 1
        self.move_history = []
    
    def play_game(self, player1: Any, player2: Any, verbose: bool = False) -> int:
        """
        Play a game between two players
        """
        self.reset()
        
        while True:
            if self.current_player == 1:
                move = player1.get_move(self)
            else:
                move = player2.get_move(self)
            
            if not self.make_move(move[0], move[1], self.current_player):
                # Invalid move, opponent wins
                return -self.current_player
            
            if verbose:
                print(f"Player {self.current_player} moves: {move}")
                self.print_board()
            
            # Check for global win
            global_win = self.check_global_win()
            if global_win != 0:
                return global_win

    