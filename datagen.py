import chess
import chess.engine
import numpy as np
# import matplotlib.pyplot as plt # Removed plotting
import random
from tqdm import tqdm
import os
import sys
import pickle
import multiprocessing
import functools
import collections

# --- CONFIGURATION ---
STOCKFISH_PATH = r"stockfish"
NUM_GAMES = 1000
RANDOM_MOVE_PROB = 0.15
STOCKFISH_TIME_LIMIT_MS = 10
EVAL_SCALE_FACTOR = 410.0
CPU_CORES_TO_USE = os.cpu_count() or 1
VALIDATION_SPLIT = 0.1
RAW_DB_FILE = 'chess_positions_db.pkl'
ORIGINAL_TRAIN_FILE = 'chess_training_data_original.pkl'
VALIDATION_FILE = 'validation.pkl'

FORCE_VALUE_UNIFORM = True
FORCE_POLICY_UNIFORM = True
VALUE_UNIFORM_TRAIN_FILE = 'chess_training_data_value_uniform.pkl'
POLICY_UNIFORM_TRAIN_FILE = 'chess_training_data_policy_uniform.pkl'

MY_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
OPP_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

# Pre-allocate arrays for tensor conversion to avoid repeated allocation
_TENSOR_CACHE = {}

# --- HELPER FUNCTIONS ---

def handle_stockfish_eval(info: dict) -> int:
    score = info.get("score")
    if score is None: return 0
    if score.is_mate():
        mate_in = score.relative.moves
        return (30000 - mate_in) if mate_in > 0 else (-30000 - mate_in)
    return score.relative.cp

# Pre-compute flip lookup table
_FLIP_CACHE = {}
for sq in range(64):
    file, rank = chess.square_file(sq), chess.square_rank(sq)
    _FLIP_CACHE[sq] = chess.square(file, 7 - rank)

def flip_move_uci(uci_move: str) -> str:
    if uci_move == "GAME_END": return "GAME_END"
    try:
        move = chess.Move.from_uci(uci_move)
        flipped_from = _FLIP_CACHE[move.from_square]
        flipped_to = _FLIP_CACHE[move.to_square]
        return chess.Move(flipped_from, flipped_to, move.promotion).uci()
    except:
        return "GAME_END"

def convert_board_to_tensor(board: chess.Board, repetition_plane_value: float) -> np.ndarray:
    tensor = np.zeros((19, 8, 8), dtype=np.float16)
    
    # My pieces (White)
    for i, piece_type in enumerate(MY_PIECES):
        for sq in board.pieces(piece_type, chess.WHITE):
            rank, file = chess.square_rank(sq), chess.square_file(sq)
            tensor[i, rank, file] = 1.0
    
    # Opponent pieces (Black)
    for i, piece_type in enumerate(OPP_PIECES):
        for sq in board.pieces(piece_type, chess.BLACK):
            rank, file = chess.square_rank(sq), chess.square_file(sq)
            tensor[i + 6, rank, file] = 1.0
    
    # Castling rights (broadcast to full plane)
    if board.has_kingside_castling_rights(chess.WHITE): tensor[12] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): tensor[13] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK): tensor[14] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): tensor[15] = 1.0
    
    # En passant
    ep_sq = board.ep_square
    if ep_sq is not None:
        rank, file = chess.square_rank(ep_sq), chess.square_file(ep_sq)
        tensor[16, rank, file] = 1.0
    
    tensor[17] = repetition_plane_value
    tensor[18] = board.halfmove_clock / 100.0
    return tensor

# Vectorized sigmoid (faster than per-value)
def calculate_value(eval_cp: int) -> float:
    scaled = eval_cp / EVAL_SCALE_FACTOR
    return 1.0 / (1.0 + np.exp(-scaled))

# --- OPTIMIZED GAME WORKER ---
def play_game_worker(game_index, existing_fens_set, random_prob, time_limit_ms, stockfish_path):
    if game_index == 0: 
        print("--- Using optimized v4 worker. ---")
    
    engine = None
    new_positions_found = {}
    
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": 1, "Hash": 16})
        
        board = chess.Board()
        game_fen_history = {}
        time_limit_sec = time_limit_ms / 1000.0

        while True:
            is_terminal = board.is_game_over(claim_draw=True)
            
            # Fast FEN base (avoid rsplit on large strings)
            fen_full = board.fen()
            last_space = fen_full.rfind(' ')
            second_last_space = fen_full.rfind(' ', 0, last_space)
            base_fen = fen_full[:second_last_space]
            
            repetition_count = game_fen_history.get(base_fen, 0) + 1
            game_fen_history[base_fen] = repetition_count
            
            is_repetition_draw = (repetition_count >= 2)
            is_game_over = is_terminal or is_repetition_draw
            repetition_plane_value = 1.0 if repetition_count > 1 else 0.0

            best_move_uci = "GAME_END"
            eval_cp = 0
            
            try:
                # Single Stockfish call
                result = engine.analyse(board, chess.engine.Limit(time=time_limit_sec))
                eval_cp = handle_stockfish_eval(result)

                if not is_game_over:
                    pv = result.get("pv")
                    if pv:
                        best_move_uci = pv[0].uci()
                elif is_repetition_draw and not is_terminal:
                    eval_cp = 0

            except (chess.engine.EngineError, chess.engine.EngineTerminatedError): 
                break

            # Canonical position (flip if Black to move)
            if board.turn == chess.WHITE:
                canonical_fen = fen_full
                canonical_move_uci = best_move_uci
                canonical_eval_cp = eval_cp
            else:
                canonical_fen = board.mirror().fen()
                canonical_move_uci = flip_move_uci(best_move_uci)
                canonical_eval_cp = eval_cp
            
            position_key = (canonical_fen, repetition_plane_value)
            
            # Only store new positions
            if position_key not in existing_fens_set and position_key not in new_positions_found:
                new_positions_found[position_key] = (canonical_move_uci, canonical_eval_cp)

            if is_game_over: 
                break
            
            # Decide move
            if random.random() <= random_prob:
                # Random move
                legal_moves = list(board.legal_moves)
                if not legal_moves: break
                move_to_play = random.choice(legal_moves)
            else:
                # Best move
                if best_move_uci == "GAME_END":
                    legal_moves = list(board.legal_moves)
                    if not legal_moves: break
                    move_to_play = legal_moves[0]
                else:
                    move_to_play = chess.Move.from_uci(best_move_uci) if board.turn == chess.WHITE else chess.Move.from_uci(flip_move_uci(best_move_uci))
                    # Validate move is legal
                    if move_to_play not in board.legal_moves:
                        legal_moves = list(board.legal_moves)
                        if not legal_moves: break
                        move_to_play = legal_moves[0]

            board.push(move_to_play)

    except Exception:
        pass
    finally:
        if engine: 
            engine.quit()
            
    return new_positions_found

def run_playouts(num_games: int, random_prob: float, existing_fens_set: set, num_workers: int) -> tuple:
    print(f"Running {num_games} new playouts in parallel on {num_workers} cores...")
    worker_func = functools.partial(play_game_worker,
                                    existing_fens_set=existing_fens_set,
                                    random_prob=random_prob,
                                    time_limit_ms=STOCKFISH_TIME_LIMIT_MS,
                                    stockfish_path=STOCKFISH_PATH)
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results_list = list(tqdm(pool.imap_unordered(worker_func, range(num_games)), 
                                 total=num_games, desc="Playing Games"))
    
    print("\nMerging results from workers...")
    new_raw_data_dict = {}
    for new_positions_found in results_list:
        new_raw_data_dict.update(new_positions_found)
    
    return new_raw_data_dict, len(new_raw_data_dict)

def process_position_worker(item):
    try:
        position_key, (canonical_move_uci, eval_cp) = item
        fen, repetition_plane_value = position_key
        
        board = chess.Board(fen)
        board_tensor = convert_board_to_tensor(board, repetition_plane_value)
        value_target = calculate_value(eval_cp)
        
        return (board_tensor, value_target, canonical_move_uci)
    except:
        return None

def process_data(raw_positions_dict: dict, num_workers: int) -> list:
    print(f"Processing {len(raw_positions_dict)} raw positions in parallel on {num_workers} cores...")
    data_items_list = list(raw_positions_dict.items())
    
    if not data_items_list:
        print("No data to process.")
        return []
    
    # Larger chunksize for efficiency
    chunksize = max(100, len(data_items_list) // (num_workers * 2))
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results_list = list(tqdm(pool.imap_unordered(process_position_worker, data_items_list, 
                                                     chunksize=chunksize), 
                                 total=len(data_items_list), desc="Processing Data"))
    
    training_data = [result for result in results_list if result is not None]
    print(f"\nSuccessfully processed {len(training_data)} positions.")
    return training_data

# --- Removed plot_value_distribution function ---

# --- Removed plot_policy_distribution function ---

def load_raw_db(filename: str) -> dict:
    if os.path.exists(filename):
        print(f"Loading existing position database from {filename}...")
        with open(filename, 'rb') as f:
            try: 
                return pickle.load(f)
            except: 
                return {}
    print("No existing position database found. Starting fresh.")
    return {}

def load_processed_data(filename: str) -> list:
    if os.path.exists(filename):
        print(f"Loading existing processed data from {filename}...")
        with open(filename, 'rb') as f:
            try: 
                return pickle.load(f)
            except: 
                return []
    print(f"No existing processed data found at {filename}. Starting fresh.")
    return []

def save_raw_db(filename: str, data: dict):
    print(f"Saving updated position database ({len(data)} positions) to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_processed_data(filename: str, data: list):
    print(f"Saving processed training data ({len(data)} samples) to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def create_value_uniform_dataset_by_pruning(training_data: list) -> list:
    print("Creating uniform *value* distribution dataset (Pruning to Min Height)...")
    if not training_data:
        print("Cannot create uniform dataset: No training data.")
        return []
    
    binned_data = {i: [] for i in range(20)}
    for sample in training_data:
        value = sample[1]
        bin_index = 19 if value == 1.0 else int(value * 20)
        binned_data[bin_index].append(sample)
    
    bin_heights = [len(samples) for samples in binned_data.values() if samples]
    if not bin_heights:
        print("No data in any bins. Skipping uniform creation.")
        return []
    
    target_bin_height = min(bin_heights)
    print(f"Target (min) bin height: {target_bin_height}")
    
    uniform_data = []
    for samples in binned_data.values():
        if samples:
            random.shuffle(samples)
            uniform_data.extend(samples[:target_bin_height])
    
    print(f"Created value-uniform dataset with {len(uniform_data)} samples (by pruning).")
    return uniform_data

def create_policy_uniform_dataset_by_pruning(training_data: list) -> list:
    print("Creating uniform *policy* distribution dataset (Pruning to Min Height)...")
    if not training_data:
        print("Cannot create uniform dataset: No training data.")
        return []
    
    binned_data = {}
    for sample in training_data:
        move_uci = sample[2]
        if move_uci not in binned_data:
            binned_data[move_uci] = []
        binned_data[move_uci].append(sample)
    
    bin_heights = [len(samples) for samples in binned_data.values() if samples]
    if not bin_heights:
        print("No data in any move bins. Skipping uniform creation.")
        return []
    
    target_bin_height = min(bin_heights)
    print(f"Target (min) move count: {target_bin_height}")
    
    uniform_data = []
    for samples in binned_data.values():
        if samples:
            random.shuffle(samples)
            uniform_data.extend(samples[:target_bin_height])
    
    print(f"Created policy-uniform dataset with {len(uniform_data)} samples (by pruning).")
    return uniform_data

def main():
    if not STOCKFISH_PATH or not os.path.exists(STOCKFISH_PATH):
        print("="*50)
        print(f"ERROR: Stockfish not found at: {STOCKFISH_PATH}")
        print("="*50)
        sys.exit(1)

    try:
        all_positions_data = load_raw_db(RAW_DB_FILE)
        initial_position_count = len(all_positions_data)
        print(f"Loaded {initial_position_count} unique canonical positions.")
        
        existing_fens_set = set(all_positions_data.keys())

        new_raw_data, new_positions_added = run_playouts(NUM_GAMES, 
                                                         RANDOM_MOVE_PROB, 
                                                         existing_fens_set, 
                                                         CPU_CORES_TO_USE)
        
        if new_positions_added == 0:
            print("\nNo new unique positions were added. Exiting.")
            return

        print(f"\nAdded {new_positions_added} new unique canonical positions.")
        
        new_processed_data = process_data(new_raw_data, CPU_CORES_TO_USE)
        
        if not new_processed_data:
            print("Processing failed to produce any new training samples. Exiting.")
            return

        all_positions_data.update(new_raw_data)
        save_raw_db(RAW_DB_FILE, all_positions_data)
        print(f"Total unique positions in database now: {len(all_positions_data)}")

        np.random.shuffle(new_processed_data)
        val_split_index = int(len(new_processed_data) * VALIDATION_SPLIT)
        
        new_val_data = new_processed_data[:val_split_index]
        new_train_data = new_processed_data[val_split_index:]
        
        print(f"Split new data: {len(new_train_data)} training samples, {len(new_val_data)} validation samples.")

        all_train_data = load_processed_data(ORIGINAL_TRAIN_FILE)
        all_val_data = load_processed_data(VALIDATION_FILE)
        
        all_train_data.extend(new_train_data)
        all_val_data.extend(new_val_data)
        
        print(f"\nTotal training samples (original): {len(all_train_data)}")
        print(f"Total validation samples: {len(all_val_data)}")

        if FORCE_VALUE_UNIFORM:
            value_uniform_data = create_value_uniform_dataset_by_pruning(all_train_data)
            if value_uniform_data:
                save_processed_data(VALUE_UNIFORM_TRAIN_FILE, value_uniform_data)
                # plot_value_distribution(value_uniform_data, title_suffix=" (Value Uniform Training Set)") # Removed
        
        if FORCE_POLICY_UNIFORM:
            policy_uniform_data = create_policy_uniform_dataset_by_pruning(all_train_data)
            if policy_uniform_data:
                save_processed_data(POLICY_UNIFORM_TRAIN_FILE, policy_uniform_data)
                # plot_policy_distribution(policy_uniform_data, title_suffix=" (Policy Uniform Training Set)") # Removed
        
        # plot_value_distribution(all_train_data, title_suffix=" (Original Training Set)") # Removed
        # plot_policy_distribution(all_train_data, title_suffix=" (Original Training Set)") # Removed
        
        save_processed_data(ORIGINAL_TRAIN_FILE, all_train_data)
        save_processed_data(VALIDATION_FILE, all_val_data)
        print(f"New total validation samples: {len(all_val_data)}")

    except Exception as e:
        print(f"\nAn unexpected error occurred in main: {e}")
    
    print("Script finished.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()