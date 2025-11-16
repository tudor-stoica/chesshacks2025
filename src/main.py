import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import time
import math
import pickle
import numpy as np
import os

# Assuming 'utils' is a module in the same package or directory
# If not, these might need to be defined or imported differently
try:
    from .utils import chess_manager, GameContext
except ImportError:
    # Fallback for running as a standalone script
    print("Could not import from .utils, using placeholder classes.")
    
    class ChessManager:
        def entrypoint(self, func):
            self._entrypoint = func
            return func
        
        def reset(self, func):
            self._reset = func
            return func
            
    chess_manager = ChessManager()
    
    class GameContext:
        def __init__(self, board):
            self.board = board
            # Add other fields as needed, e.g., time_remaining_ms
            self.time_remaining_ms = 300000 


# A helper class for a single Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)

# Define the "Leaner" model
class ChessCNN(nn.Module):
    def __init__(self, num_policy_outputs):
        super().__init__()
        
        # --- Configuration ---
        self.board_size = 8
        self.in_channels = 19 # 19 planes
        
        self.num_channels = 256
        self.num_res_blocks = 10
        head_fc_size = 32
        head_conv_channels = 2
        
        fc1_input_size = head_conv_channels * self.board_size * self.board_size

        # --- 1. Initial Convolutional Layer ---
        self.conv_in = nn.Conv2d(self.in_channels, self.num_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(self.num_channels)

        # --- 2. Residual Stack ---
        self.res_stack = nn.ModuleList([ResidualBlock(self.num_channels, self.num_channels) for _ in range(self.num_res_blocks)])
        
        self.flatten = nn.Flatten()
        
        # --- 3. The "Value Head" ---
        self.value_conv = nn.Conv2d(self.num_channels, head_conv_channels, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(head_conv_channels)
        self.value_fc1 = nn.Linear(fc1_input_size, head_fc_size)
        self.value_fc2 = nn.Linear(head_fc_size, 1) # Final output neuron

        # --- 4. The "Policy Head" ---
        self.policy_conv = nn.Conv2d(self.num_channels, head_conv_channels, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(head_conv_channels)
        self.policy_fc1 = nn.Linear(fc1_input_size, num_policy_outputs)

    def forward(self, x):
        # 1. Initial layer
        x = F.relu(self.bn_in(self.conv_in(x)))
        
        # 2. Pass through all residual blocks
        for block in self.res_stack:
            x = block(x)
            
        # --- 3. Value Head Path ---
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = self.flatten(v) 
        v = F.relu(self.value_fc1(v))
        
        # --- UNCHANGED (as requested) ---
        # Output remains [0, 1] as requested by user.
        # We will convert this in the _model_wrapper.
        value_output = torch.sigmoid(self.value_fc2(v))
        # ---------------------------------------------------
        
        # --- 4. Policy Head Path ---
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = self.flatten(p)
        policy_logits = self.policy_fc1(p)
        
        # Return the [0, 1] value and policy logits
        return value_output, policy_logits

PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

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
    def __init__(self, model, c_puct=2, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        """
        :param model: A function that takes a chess.Board and returns (policy, value).
                      - policy: A dictionary mapping move.uci() string to probability.
                      - value: A float from -1 (loss) to +1 (win) for the current player.
        :param c_puct: The exploration constant (controls trade-off between exploitation and exploration).
        :param dirichlet_alpha: Alpha parameter for Dirichlet noise.
        :param dirichlet_epsilon: Epsilon (weight) for Dirichlet noise.
        """
        self.model = model
        self.c_puct = c_puct
        # --- NEW: Store Dirichlet parameters ---
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

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
        
        simulations= 0
        # Main MCTS loop
        while time.time() - start_time < timelimit:
            # We create a copy of the board for each simulation
            # to avoid modifying the original
            board_sim = root_board.copy()
            
            simulations += 1
            # Run one simulation (select, expand, backpropagate)
            self.run_simulation(root, board_sim)
            
        # After the time is up, select the best move
        best_move = self.select_best_move(root)
        
        # --- MODIFIED: Return the root node as well ---
        return best_move, root

    # --- NEW: Helper for Dirichlet Noise ---
    def add_dirichlet_noise(self, policy_dict_uci, legal_moves):
        """
        Applies Dirichlet noise to the policy probabilities for exploration.
        """
        legal_moves_uci = [m.uci() for m in legal_moves]
        
        # Generate noise
        num_legal = len(legal_moves_uci)
        if num_legal <= 1:
            return # No noise needed for 0 or 1 moves

        noise = np.random.dirichlet([self.dirichlet_alpha] * num_legal)
        
        # Apply noise
        for i, move_uci in enumerate(legal_moves_uci):
            if move_uci in policy_dict_uci:
                policy_dict_uci[move_uci] = (1 - self.dirichlet_epsilon) * policy_dict_uci[move_uci] + \
                                            self.dirichlet_epsilon * noise[i]
        
        # Renormalize to ensure probabilities sum to 1 (optional, but good practice)
        total_prob = sum(policy_dict_uci.get(m_uci, 0.0) for m_uci in legal_moves_uci)
        if total_prob > 0:
            for m_uci in legal_moves_uci:
                if m_uci in policy_dict_uci:
                    policy_dict_uci[m_uci] /= total_prob
        
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
        
        value = 0.0

        # Check if the game is over at this leaf
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                value = 1.0  # White won
            elif result == "0-1":
                value = -1.0 # Black won
            else:
                value = 0.0  # Draw
            
            if board.turn == chess.BLACK:
                value = -value

        else:
            is_root_expansion = (current_node.parent is None)
            
            policy_dict_uci, value = self.model(board)
            
            legal_moves = list(board.legal_moves)
            
            # Apply Dirichlet Noise if this is the root
            if is_root_expansion:
                self.add_dirichlet_noise(policy_dict_uci, legal_moves)
            # --------------------------------------------------------
            
            # Create child nodes for all legal moves
            for move in legal_moves:
                move_uci = move.uci()
                # Get the prior probability for this move from the model's policy
                move_prob = policy_dict_uci.get(move_uci, 0.0)
                
                if move not in current_node.children:
                    current_node.children[move] = Node(parent=current_node, move=move, prior_p=move_prob)

        temp_node = current_node
        while temp_node is not None:
            temp_node.N += 1
            # `value` is from the perspective of the player-to-move at the
            # child node. We must flip it for the parent.
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

g_model = None
g_move_map = None
g_device = None
g_mcts = None

MODEL_FILE = "chess_cnn.pth"
MAP_FILE = "chess_cnn_move_map.pkl"

PIECE_TO_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

def convert_board_to_tensor(board: chess.Board, repetition_plane_value: float) -> np.ndarray:
    """
    Convert board to tensor matching the training data format.
    When it's Black's turn, we mirror the board but keep castling rights in absolute White/Black order.
    This matches how generate_data.py creates canonical positions.
    """
    tensor = np.zeros((19, 8, 8), dtype=np.float32)
    
    if board.turn == chess.WHITE:
        # White's turn: straightforward representation
        # Channels 0-5: White pieces
        for i, piece_type in enumerate(PIECES):
            for sq in board.pieces(piece_type, chess.WHITE):
                tensor[i, chess.square_rank(sq), chess.square_file(sq)] = 1.0
        
        # Channels 6-11: Black pieces
        for i, piece_type in enumerate(PIECES):
            for sq in board.pieces(piece_type, chess.BLACK):
                tensor[i + 6, chess.square_rank(sq), chess.square_file(sq)] = 1.0
        
        # Castling rights in absolute White/Black order (as in training data)
        if board.has_kingside_castling_rights(chess.WHITE): tensor[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): tensor[13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK): tensor[14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): tensor[15, :, :] = 1.0
        
        # En passant square
        ep_sq = board.ep_square
        if ep_sq:
            tensor[16, chess.square_rank(ep_sq), chess.square_file(ep_sq)] = 1.0
    else:
        # Black's turn: mirror the board to create canonical position
        # After mirroring: Channels 0-5 are Black pieces (now at bottom), 6-11 are White pieces (now at top)
        
        # Channels 0-5: Black pieces (mirrored)
        for i, piece_type in enumerate(PIECES):
            for sq in board.pieces(piece_type, chess.BLACK):
                tensor[i, 7 - chess.square_rank(sq), 7 - chess.square_file(sq)] = 1.0
        
        # Channels 6-11: White pieces (mirrored)
        for i, piece_type in enumerate(PIECES):
            for sq in board.pieces(piece_type, chess.WHITE):
                tensor[i + 6, 7 - chess.square_rank(sq), 7 - chess.square_file(sq)] = 1.0
        
        # CRITICAL: Castling rights stay in absolute White/Black order (NOT flipped)
        # This matches training data where board.mirror() flips pieces but not castling rights
        if board.has_kingside_castling_rights(chess.WHITE): tensor[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): tensor[13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK): tensor[14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): tensor[15, :, :] = 1.0
        
        # En passant square (mirrored)
        ep_sq = board.ep_square
        if ep_sq:
            tensor[16, 7 - chess.square_rank(ep_sq), 7 - chess.square_file(ep_sq)] = 1.0
    
    tensor[17, :, :] = repetition_plane_value
    tensor[18, :, :] = board.halfmove_clock / 100.0
    return tensor

def _model_wrapper(board: chess.Board) -> tuple[dict[str, float], float]:
    """
    A wrapper function that adheres to the MCTS's expected `model(board)` signature.
    It handles:
    1. Board-to-tensor conversion.
    2. Model inference.
    3. Post-processing of policy and value outputs.
    """
    global g_model, g_move_map, g_device
    
    # Remember if we need to flip the policy output
    is_black_to_move = (board.turn == chess.BLACK)
    
    # 1. Board-to-tensor conversion
    # Check for repetition to set the 17th plane
    repetition_val = 1.0 if board.is_repetition(count=2) else 0.0
    tensor_np = convert_board_to_tensor(board, repetition_val)
    
    # Convert to PyTorch tensor, add batch dim, and send to device
    tensor_torch = torch.from_numpy(tensor_np).unsqueeze(0).to(g_device)

    # 2. Model inference
    with torch.no_grad():
        # Model returns (value_output, policy_logits)
        # value_output is in the [0, 1] range (due to sigmoid)
        value_output_sigmoid, policy_logits = g_model(tensor_torch)

    # 3. Post-processing
    
    # --- Process Value ---
    # --- CRITICAL FIX: Convert [0, 1] sigmoid to [-1, 1] negamax range ---
    # This scales the [0, 1] output to the [-1, 1] range
    # 0.0 -> -1.0 (loss)
    # 0.5 ->  0.0 (draw)
    # 1.0 -> +1.0 (win)
    # This allows the MCTS's `value = -value` logic to work correctly.
    value_sigmoid = value_output_sigmoid.item()
    value_negamax = (value_sigmoid * 2.0) - 1.0
    # --------------------------------------------------------------------

    # --- Process Policy ---
    # The model outputs raw logits. We apply softmax to get probabilities.
    # g_move_map is assumed to be a list where index 'i' corresponds to the
    # i-th logit, and the value is the UCI move string.
    policy_probs = F.softmax(policy_logits.squeeze(0), dim=0).cpu().numpy()
    
    policy_dict_uci = {}
    
    if is_black_to_move:
        # The model was trained on mirrored positions, so it outputs moves as if
        # playing from the bottom of the board. We need to flip these moves back
        # to the original board orientation.
        for i, canonical_move_uci in enumerate(g_move_map):
            if canonical_move_uci == "GAME_END":
                continue
            # Flip the move from the canonical (mirrored) space back to the actual board
            try:
                # Parse the canonical move
                canonical_move = chess.Move.from_uci(canonical_move_uci)
                # Flip it back to the original board orientation
                flipped_from = chess.square_mirror(canonical_move.from_square)
                flipped_to = chess.square_mirror(canonical_move.to_square)
                actual_move = chess.Move(flipped_from, flipped_to, canonical_move.promotion)
                policy_dict_uci[actual_move.uci()] = policy_probs[i]
            except:
                pass  # Skip invalid moves
    else:
        # White to move: no flipping needed
        policy_dict_uci = {
            move_uci: policy_probs[i] 
            for i, move_uci in enumerate(g_move_map)
            if move_uci != "GAME_END"
        }

    # Return the NEGMAX value
    return policy_dict_uci, value_negamax

# --- Updated Entrypoint ---

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    This is the main "thinking" function. It's called when the engine
    needs to decide on a move.
    """
    global g_mcts
    
    # Ensure the model and MCTS are loaded.
    # This is a safety check in case reset wasn't called.
    if g_mcts is None:
        print("MCTS not initialized, calling reset_func...")
        reset_func(ctx)

    # Get the current board state from the context
    board = ctx.board
    
    # --- NEW: Log base policy before MCTS search ---
    print("\n--- Base Policy (Raw Model Output) ---")
    policy_dict_uci, base_value = _model_wrapper(board.copy())
    
    # Get legal moves for filtering
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [m.uci() for m in legal_moves]
    
    # Filter policy to only legal moves and sort by probability
    legal_policy = [(move_uci, policy_dict_uci.get(move_uci, 0.0)) 
                    for move_uci in legal_moves_uci]
    legal_policy.sort(key=lambda x: x[1], reverse=True)
    
    # Display base value
    print(f"Base Value Estimate: {base_value:.4f} (from current player's perspective)")
    
    # Display top moves from base policy
    print(f"\nTop moves from base policy (before MCTS):")
    print(f"{'Move':<10} | {'Probability':<12}")
    print("-" * 24)
    for move_uci, prob in legal_policy[:10]:  # Show top 10
        print(f"{move_uci:<10} | {prob*100:11.6f}%")
    
    print(f"\nTotal legal moves: {len(legal_moves)}")
    # ------------------------------------------------
    
    # Set a time limit for the search (e.g., 3 seconds)
    # In a real system, you might get this from the context (e.g., ctx.time_remaining_ms)
    time_limit_sec = 3.0
    print(f"\nStarting MCTS search for {time_limit_sec} seconds...")
    
    # Run the MCTS search
    # We pass a copy of the board to be safe
    # --- MODIFIED: Receive the 'root_node' back from search ---
    best_move, root_node = g_mcts.search(board.copy(), timelimit=time_limit_sec)
    
    print(f"MCTS search complete. Best move: {best_move.uci()}")
    
    # --- NEW: Print statistics for all child moves ---
    print("\n--- Move Statistics (After MCTS) ---")
    
    # Get a list of (move_uci, visit_count, avg_value) tuples
    move_stats = []
    if root_node and root_node.N > 0: # Avoid division by zero if no sims ran
        for move, child_node in root_node.children.items():
            if child_node.N > 0:
                # child.get_value() is value from child's perspective (opponent)
                # We negate it to get the value for the current player
                avg_value = -child_node.get_value() 
                move_stats.append((move.uci(), child_node.N, avg_value))
                
        # Sort by visit count (most simulations first)
        move_stats.sort(key=lambda item: item[1], reverse=True)
        
        # Print the sorted list
        total_sims = root_node.N
        print(f"Total simulations from root: {total_sims}")
        print(f"{'Move':<10} | {'Visits':<10} | {'Percentage':<12} | {'Value (for me)':<15}")
        print("-" * 52) # Adjusted width
        for move_uci, n, val in move_stats:
            percentage = (n / total_sims) * 100
            print(f"{move_uci:<10} | {n:<10} | {percentage:11.2f}% | {val:14.4f}")
    else:
        print("No simulations were run or root node is invalid.")
    # -------------------------------------------------
    
    print("\n") # Add a newline for cleaner logs
    
    # Return the best move found by the search
    return best_move

@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    This function is called once to initialize the engine, load models,
    and set up the MCTS.
    """
    global g_model, g_move_map, g_device, g_mcts
    
    print("--- Resetting and loading model ---")
    
    # --- MODIFICATION: Build absolute paths relative to this script file ---
    # __file__ is a special variable in Python that holds the path to the current script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    map_path = os.path.join(script_dir, MAP_FILE)
    model_path = os.path.join(script_dir, MODEL_FILE)
    # ---------------------------------------------------------------------
    
    # 1. Set device
    g_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {g_device}")
    
    # 2. Load the move map
    try:
        # Use the absolute path
        print(f"Loading move map from: {map_path}")
        with open(map_path, 'rb') as f:
            # train.py saves a dict: {'MOVE_TO_INDEX': ..., 'INDEX_TO_MOVE': ...}
            move_map_dict = pickle.load(f)
            # We need the INDEX_TO_MOVE list for our MCTS wrapper
            g_move_map = move_map_dict['INDEX_TO_MOVE']
    except Exception as e:
        # Print the full path on error for easier debugging
        print(f"Error loading move map '{map_path}': {e}")
        return

    num_policy_outputs = len(g_move_map)
    print(f"Loaded move map with {num_policy_outputs} possible moves.")

    # 3. Initialize the model
    g_model = ChessCNN(num_policy_outputs=num_policy_outputs).to(g_device)
    
    # 4. Load model weights
    try:
        # Use the absolute path
        print(f"Loading model weights from: {model_path}")
        g_model.load_state_dict(torch.load(model_path, map_location=g_device))
    except Exception as e:
        # Print the full path on error for easier debugging
        print(f"Error loading model weights '{model_path}': {e}")
        return
        
    # 5. Set model to evaluation mode (disables dropout, batchnorm updates, etc.)
    g_model.eval()
    
    # 6. Initialize the MCTS
    # --- MODIFIED: Pass Dirichlet noise parameters ---
    # `c_puct=1.41` (sqrt(2)) is a common value
    g_mcts = MCTS(model=_model_wrapper, 
                  c_puct=1.41, 
                  dirichlet_alpha=0.3, 
                  dirichlet_epsilon=0.25)
    
    print("--- Model and MCTS initialized successfully ---")

# --- Example of how to run (if this script is executed directly) ---
if __name__ == "__main__":
    print("Running a test of the chess AI script...")
    
    # Create a dummy context and board
    test_board = chess.Board()
    test_ctx = GameContext(board=test_board)
    
    # 1. Call reset to load models (this happens automatically in the real env)
    reset_func(test_ctx)
    
    # 2. Check if loading was successful
    if g_mcts:
        print("\n--- Starting a test game ---")
        print(test_ctx.board)
        
        # 3. Ask for the first move
        # Use the entrypoint function directly
        move1 = chess_manager._entrypoint(test_ctx)
        test_ctx.board.push(move1)
        print(f"\nAI moved: {move1.uci()}")
        print(test_ctx.board)
        
        # 4. Make a simple opponent move
        opponent_move = chess.Move.from_uci("e7e6") # A simple pawn move
        if opponent_move in test_ctx.board.legal_moves:
            test_ctx.board.push(opponent_move)
            print(f"\nOpponent moved: {opponent_move.uci()}")
            print(test_ctx.board)
            
            # 5. Ask for the second AI move
            # Use the entrypoint function directly
            move2 = chess_manager._entrypoint(test_ctx)
            test_ctx.board.push(move2)
            print(f"\nAI moved: {move2.uci()}")
            print(test_ctx.board)
        else:
            print(f"Test move {opponent_move.uci()} is not legal.")
    
    else:
        print("\nTest run failed: MCTS was not initialized.")
        print("This is likely because the model files ('chess_cnn.pth' and 'chess_cnn_move_map.pkl')")
        print(f"were not found in the same directory as this script: {os.path.dirname(os.path.abspath(__file__))}")