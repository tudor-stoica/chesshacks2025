from .utils import chess_manager, GameContext
from chess import Move
import random
import time

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

def get_model_info(board, model):
    value, policy = model(board)
    return value, policy

import chess
import time
import math
import random

class Node:
    """
    A node in the MCTS tree. Stores statistics for a particular board state.
    """
    def __init__(self, parent, move, prior_p):
        self.parent = parent  # Parent node
        self.move = move      # The move that led to this node
        self.children = {}    # A map from move (chess.Move) to Node
        
        self.N = 0            # Visit count
        self.Q = 0.0          # Total value. The sum of all values backpropagated through this node.
        self.P = prior_p      # Prior probability of selecting this node (from the model's policy)

    def get_value(self):
        """
        Returns the average value (W) of this node.
        Q is the total value, N is the visit count.
        """
        if self.N == 0:
            return 0.0
        return self.Q / self.N

    def is_leaf(self):
        """
        Checks if this node is a leaf (i.e., has no expanded children).
        """
        return len(self.children) == 0

class MCTS:
    """
    The main MCTS class.
    """
    def __init__(self, model, c_puct=1.41):
        """
        :param model: A function that takes a chess.Board and returns (policy, value).
                      - policy: A dictionary mapping move.uci() string to probability.
                      - value: A float from -1 (loss) to +1 (win) for the current player.
        :param c_puct: The exploration constant (controls trade-off between exploitation and exploration).
        """
        self.model = model
        self.c_puct = c_puct

    def select_best_move(self, root):
        """
        After the search, selects the best move from the root node.
        The most robust choice is the one with the highest visit count.
        """
        best_n = -1
        best_move = None
        for move, child in root.children.items():
            if child.N > best_n:
                best_n = child.N
                best_move = move
        return best_move

    def search(self, root_board, timelimit):
        """
        Runs the MCTS search for a given time limit.
        """
        # Create the root node
        # The root's prior is 1.0, and it has no parent or move
        root = Node(parent=None, move=None, prior_p=1.0)
        
        start_time = time.time()
        
        # Main MCTS loop
        while time.time() - start_time < timelimit:
            # We create a copy of the board for each simulation
            # to avoid modifying the original
            board_sim = root_board.copy()
            
            # Run one simulation (select, expand, backpropagate)
            self.run_simulation(root, board_sim)
            
        # After the time is up, select the best move
        return self.select_best_move(root)

    def run_simulation(self, node, board):
        """
        Performs one simulation from the given node.
        This involves the 4 MCTS steps:
        1. Selection
        2. Expansion
        3. Simulation (Model Evaluation)
        4. Backpropagation
        """
        
        # 1. SELECTION: Traverse the tree using PUCT
        current_node = node
        while not current_node.is_leaf():
            best_move = self.select_child_puct(current_node)
            current_node = current_node.children[best_move]
            board.push(best_move)

        # Now `current_node` is a leaf, and `board` is the state at that leaf.
        
        # Check if the game is over at this leaf
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                value = 1.0  # White won
            elif result == "0-1":
                value = -1.0 # Black won
            else:
                value = 0.0  # Draw
            
            # We need to flip the value if it's not the current player's "turn"
            # The value should be from the perspective of the player *whose turn it just was*.
            # `board.turn` is the player *to move*, who *lost* by checkmate or drew by stalemate.
            # So, if turn is WHITE (True), Black just moved and won/drew.
            # If turn is BLACK (False), White just moved and won/drew.
            if board.turn == chess.WHITE: # Black just moved
                value = -value
            
        else:
            # 2. EXPANSION: If not a terminal node, expand it
            # 3. SIMULATION: Get policy and value from the model
            
            # `model` returns value from the perspective of the *current* player
            policy_dict_uci, value = self.model(board)
            
            legal_moves = list(board.legal_moves)
            
            # Create child nodes for all legal moves
            for move in legal_moves:
                move_uci = move.uci()
                # Get the prior probability for this move from the model's policy
                move_prob = policy_dict_uci.get(move_uci, 0.0)
                
                current_node.children[move] = Node(parent=current_node, move=move, prior_p=move_prob)

        # 4. BACKPROPAGATION: Update statistics up the tree
        temp_node = current_node
        while temp_node is not None:
            temp_node.N += 1
            temp_node.Q += value
            # The value flips for the parent (opponent's perspective)
            value = -value
            temp_node = temp_node.parent

    def select_child_puct(self, node):
        """
        Selects the best child node to explore using the PUCT formula.
        """
        best_score = -float('inf')
        best_move = None
        
        # Total visit count of the parent (sqrt for exploration scaling)
        sqrt_parent_N = math.sqrt(node.N)

        for move, child in node.children.items():
            
            # --- The PUCT Formula ---
            # Q = Exploitation term (average value)
            # U = Exploration term
            
            # Q is the value *from the perspective of the current (parent) node*
            # child.get_value() is from the child's perspective (opponent)
            # So, we must negate it: Q = -child.get_value()
            Q = -child.get_value()
            
            # U = c_puct * P(s,a) * (sqrt(N_parent) / (1 + N_child))
            U = self.c_puct * child.P * (sqrt_parent_N / (1 + child.N))
            
            score = Q + U
            
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move

# --- Dummy Model for Demonstration ---

def dummy_model(board):
    """
    A simple dummy model that returns a uniform policy and a random value.
    Your real model (e.g., a neural network) would replace this.
    
    :param board: chess.Board
    :return: (policy_dict, value)
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return {}, 0.0
        
    # Policy: Uniform distribution over all legal moves
    move_count = len(legal_moves)
    policy_dict = {move.uci(): 1.0 / move_count for move in legal_moves}
    
    # Value: A random score between -1 and 1
    # A real model would evaluate the board position
    value = random.uniform(-1.0, 1.0)
    
    return policy_dict, value

# --- Example Usage ---

if __name__ == "__main__":
    
    board = chess.Board()
    print("Starting board:")
    print(board)
    print("-----------------")

    # Initialize MCTS with the dummy model
    mcts_searcher = MCTS(model=dummy_model, c_puct=1.41)

    # Search for 5 seconds
    search_time_seconds = 5
    print(f"Searching for {search_time_seconds} seconds...")
    best_move = mcts_searcher.search(board, timelimit=search_time_seconds)

    if best_move:
        print(f"\nBest move found: {best_move.uci()}")
        board.push(best_move)
        print("\nBoard after move:")
        print(board)
    else:
        print("\nNo legal moves found or search failed.")


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.1)

    legal_moves = list(ctx.board.generate_legal_moves())

    return legal_moves[0]


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
