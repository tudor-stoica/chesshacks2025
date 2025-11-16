import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import time
import math
import pickle
import numpy as np
import os

from .utils import chess_manager, GameContext


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

# Define the "Leaner" model (Policy-Only)
class ChessCNN(nn.Module):
    def __init__(self, num_policy_outputs):
        super().__init__()
        
        # --- Configuration ---
        self.board_size = 8
        self.in_channels = 19 # 19 planes
        
        self.num_channels = 256
        self.num_res_blocks = 10
        # head_fc_size = 32 # No longer needed for value head
        head_conv_channels = 2
        
        fc1_input_size = head_conv_channels * self.board_size * self.board_size

        # --- 1. Initial Convolutional Layer ---
        self.conv_in = nn.Conv2d(self.in_channels, self.num_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(self.num_channels)

        # --- 2. Residual Stack ---
        self.res_stack = nn.ModuleList([ResidualBlock(self.num_channels, self.num_channels) for _ in range(self.num_res_blocks)])
        
        self.flatten = nn.Flatten()
        
        # --- 3. The "Value Head" (REMOVED) ---
        # self.value_conv = ...
        # self.value_bn = ...
        # self.value_fc1 = ...
        # self.value_fc2 = ...

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
            
        # --- 3. Value Head Path (REMOVED) ---
        
        # --- 4. Policy Head Path ---
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = self.flatten(p)
        policy_logits = self.policy_fc1(p)
        
        # Return only the policy logits
        return policy_logits

PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

# --- Node Class (REMOVED) ---
# --- MCTS Class (REMOVED) ---

g_model = None
g_move_map = None
g_device = None
# g_mcts = None # No longer needed

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

def _model_wrapper(board: chess.Board) -> dict[str, float]:
    """
    A wrapper function that handles:
    1. Board-to-tensor conversion.
    2. Model inference.
    3. Post-processing of policy output.
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
        # Model now only returns policy_logits
        policy_logits = g_model(tensor_torch)

    # 3. Post-processing
    
    # --- Process Value (REMOVED) ---

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

    # Return only the policy dictionary
    return policy_dict_uci

# --- Updated Entrypoint ---

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    This is the main "thinking" function. It's called when the engine
    needs to decide on a move.
    
    It now works by:
    1. Calling the policy network.
    2. Selecting the legal move with the highest probability.
    """
    global g_model
    
    # Ensure the model is loaded.
    if g_model is None:
        reset_func(ctx)

    # Get the current board state from the context
    board = ctx.board
    
    # 1. Get the policy dictionary from the model
    policy_dict_uci = _model_wrapper(board.copy())
    
    # 2. Get all legal moves
    legal_moves = list(board.legal_moves)
    
    # Filter policy to only legal moves and sort by probability
    legal_policy_with_moves = []
    for move in legal_moves:
        move_uci = move.uci()
        prob = policy_dict_uci.get(move_uci, 0.0)
        legal_policy_with_moves.append((move, prob))

    # 3. Find the best legal move
    best_move = None
    if legal_policy_with_moves:
        # Sort by probability (item[1]) in descending order
        legal_policy_with_moves.sort(key=lambda x: x[1], reverse=True)
        
        # The best move is the one with the highest probability
        best_move = legal_policy_with_moves[0][0]
        
    else:
        # This should only happen if the game is over, but as a fallback:
        # The context/game manager will handle game-over states,
        # but we must return something if called.
        if legal_moves:
            best_move = legal_moves[0] # Just pick the first one
        else:
            return None # Or raise an exception

    
    # Return the best move found by the policy
    return best_move

@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    This function is called once to initialize the engine and load the model.
    """
    global g_model, g_move_map, g_device
    
    
    # --- MODIFICATION: Build absolute paths relative to this script file ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    map_path = os.path.join(script_dir, MAP_FILE)
    model_path = os.path.join(script_dir, MODEL_FILE)
    # ---------------------------------------------------------------------
    
    # 1. Set device
    g_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load the move map
    try:
        # Use the absolute path
        with open(map_path, 'rb') as f:
            move_map_dict = pickle.load(f)
            g_move_map = move_map_dict['INDEX_TO_MOVE']
    except Exception as e:
        return

    num_policy_outputs = len(g_move_map)

    # 3. Initialize the model
    g_model = ChessCNN(num_policy_outputs=num_policy_outputs).to(g_device)
    
    # 4. Load model weights
    try:
        # Use the absolute path
        g_model.load_state_dict(torch.load(model_path, map_location=g_device))
    except Exception as e:
        return
        
    # 5. Set model to evaluation mode
    g_model.eval()
    
    # 6. Initialize the MCTS (REMOVED)