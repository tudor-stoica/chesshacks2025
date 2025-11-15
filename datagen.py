import chess
import chess.engine
import numpy as np
# import matplotlib.pyplot as plt  <-- REMOVED
import random
from tqdm import tqdm
import os
import sys
import pickle
import multiprocessing
import functools

# --- CONFIGURATION ---
STOCKFISH_PATH = r"stockfish"
# --- REDUCED FOR A FAST TEST ---
NUM_GAMES = 1000
RANDOM_MOVE_PROB = 0.15
STOCKFISH_TIME_LIMIT_MS = 10 
EVAL_SCALE_FACTOR = 410.0
CPU_CORES_TO_USE = 16
VALIDATION_SPLIT = 0
RAW_DB_FILE = 'chess_positions_db.pkl'
ORIGINAL_TRAIN_FILE = 'chess_training_data_original.pkl'
VALIDATION_FILE = 'validation.pkl'

# --- UPDATED CONFIG AS PER YOUR REQUEST ---
FORCE_VALUE_UNIFORM = True  # Renamed from FORCE_UNIFORM
FORCE_POLICY_UNIFORM = True # New flag for move distribution
VALUE_UNIFORM_TRAIN_FILE = 'chess_training_data_value_uniform.pkl'
POLICY_UNIFORM_TRAIN_FILE = 'chess_training_data_policy_uniform.pkl'
# --- END UPDATED CONFIG ---

MY_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
OPP_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

# --- ALL HELPER FUNCTIONS ---

def handle_stockfish_eval(info: dict) -> int:
    score = info.get("score")
    if score is None: return 0
    if score.is_mate():
        mate_in = score.relative.moves
        if mate_in > 0: return 30000 - mate_in
        else: return -30000 - mate_in
    else: return score.relative.cp

def flip_move_uci(uci_move: str) -> str:
    if uci_move == "GAME_END": return "GAME_END"
    try:
        move = chess.Move.from_uci(uci_move)
        from_sq, to_sq = move.from_square, move.to_square
        flipped_from = chess.square(chess.square_file(from_sq), 7 - chess.square_rank(from_sq))
        flipped_to = chess.square(chess.square_file(to_sq), 7 - chess.square_rank(to_sq))
        return chess.Move(flipped_from, flipped_to, move.promotion).uci()
    except Exception: return "GAME_END"

def convert_board_to_tensor(board: chess.Board, repetition_plane_value: float) -> np.ndarray:
    tensor = np.zeros((19, 8, 8), dtype=np.float32)
    for i, piece_type in enumerate(MY_PIECES):
        for sq in board.pieces(piece_type, chess.WHITE):
            tensor[i, chess.square_rank(sq), chess.square_file(sq)] = 1.0
    for i, piece_type in enumerate(OPP_PIECES):
        for sq in board.pieces(piece_type, chess.BLACK):
            tensor[i + 6, chess.square_rank(sq), chess.square_file(sq)] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE): tensor[12, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): tensor[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK): tensor[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): tensor[15, :, :] = 1.0
    ep_sq = board.ep_square
    if ep_sq: tensor[16, chess.square_rank(ep_sq), chess.square_file(ep_sq)] = 1.0
    tensor[17, :, :] = repetition_plane_value
    tensor[18, :, :] = board.halfmove_clock / 100.0
    return tensor

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def calculate_value(eval_cp: int) -> float:
    scaled_eval = eval_cp / EVAL_SCALE_FACTOR
    return sigmoid(scaled_eval)

def in_sample(position_data: tuple, filter_func=None) -> bool:
    if filter_func is None: return True
    else: return filter_func(position_data)

def play_game_worker(game_index, existing_fens_set, random_prob, time_limit_ms, stockfish_path):
    
    # --- ADDED THIS LINE FOR VERIFICATION ---
    if game_index == 0: print("--- Using v2 (eval-flipping) worker. ---")
    
    engine = None
    new_positions_found = {}
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": 1})
        
        board = chess.Board()
        game_fen_history = {} 

        while True:
            is_terminal = board.is_game_over(claim_draw=True)
            base_fen = board.fen().rsplit(' ', 2)[0]
            repetition_count = game_fen_history.get(base_fen, 0) + 1
            game_fen_history[base_fen] = repetition_count
            
            is_repetition_draw = (repetition_count >= 2)
            is_game_over = is_terminal or is_repetition_draw
            repetition_plane_value = 1.0 if repetition_count > 1 else 0.0

            best_move = None
            eval_cp = 0
            best_move_uci = "GAME_END"
            
            try:
                info = engine.analyse(board, chess.engine.Limit(time=time_limit_ms / 1000.0))
                eval_cp = handle_stockfish_eval(info)

                if is_game_over:
                    if is_repetition_draw and not is_terminal: eval_cp = 0
                else:
                    best_move = info.get("pv", [None])[0]
                    if best_move is None:
                        best_move = engine.play(board, chess.engine.Limit(time=time_limit_ms / 1000.0)).move
                    best_move_uci = best_move.uci()

            except (chess.engine.EngineError, chess.engine.EngineTerminatedError): break 

            # --- START CANONICAL FEN LOGIC (WITH FIX) ---
            canonical_fen = ""
            canonical_move_uci = "GAME_END"
            canonical_eval_cp = 0
            
            if board.turn == chess.WHITE:
                canonical_fen = board.fen()
                canonical_move_uci = best_move_uci
                canonical_eval_cp = eval_cp
            else:
                canonical_fen = board.mirror().fen()
                canonical_move_uci = flip_move_uci(best_move_uci)
                canonical_eval_cp = eval_cp 
            
            position_key = (canonical_fen, repetition_plane_value)
            
            if position_key not in existing_fens_set and position_key not in new_positions_found:
                new_positions_found[position_key] = (canonical_move_uci, canonical_eval_cp)
            # --- END CANONICAL FEN LOGIC ---

            if is_game_over: break
            
            legal_moves = list(board.legal_moves)
            if not legal_moves: break 

            move_to_play = best_move if random.random() > random_prob else random.choice(legal_moves)
            if move_to_play not in legal_moves:
                move_to_play = legal_moves[0]

            board.push(move_to_play)

    except Exception: pass 
    finally:
        if engine: engine.quit()
            
    return new_positions_found

def run_playouts(num_games: int, random_prob: float, existing_fens_set: set, num_workers: int) -> (dict, int):
    print(f"Running {num_games} new playouts in parallel on {num_workers} cores...")
    worker_func = functools.partial(play_game_worker,
                                    existing_fens_set=existing_fens_set,
                                    random_prob=random_prob,
                                    time_limit_ms=STOCKFISH_TIME_LIMIT_MS,
                                    stockfish_path=STOCKFISH_PATH)
    new_raw_data_dict = {}
    with multiprocessing.Pool(processes=num_workers) as pool:
        # We pass range(num_games) so the game_index is passed
        results_list = list(tqdm(pool.imap_unordered(worker_func, range(num_games)), total=num_games, desc="Playing Games"))
    print("\nMerging results from workers...")
    for new_positions_found in tqdm(results_list, desc="Merging Results"):
        new_raw_data_dict.update(new_positions_found)
    new_positions_count = len(new_raw_data_dict)
    return new_raw_data_dict, new_positions_count

def process_position_worker(item):
    fen = None
    try:
        position_key, (canonical_move_uci, eval_cp) = item
        fen, repetition_plane_value = position_key
        position_data = (fen, canonical_move_uci, eval_cp, repetition_plane_value)
        if not in_sample(position_data, filter_func=None):
            return None
        board = chess.Board(fen)
        board_tensor = convert_board_to_tensor(board, repetition_plane_value)
        value_target = calculate_value(eval_cp)
        policy_target_uci = canonical_move_uci
        return (board_tensor, value_target, policy_target_uci)
    except Exception as e:
        # Suppress errors for speed
        # print(f"Skipping invalid FEN '{fen}': {e}")
        pass
    return None

def process_data(raw_positions_dict: dict, num_workers: int) -> list:
    print(f"Processing {len(raw_positions_dict)} raw positions in parallel on {num_workers} cores...")
    data_items_list = list(raw_positions_dict.items())
    if not data_items_list:
        print("No data to process.")
        return []
    with multiprocessing.Pool(processes=num_workers) as pool:
        results_list = list(tqdm(pool.imap_unordered(process_position_worker, data_items_list), total=len(data_items_list), desc="Processing Data"))
    training_data = [result for result in results_list if result is not None]
    print(f"\nSuccessfully processed {len(training_data)} positions.")
    return training_data

def load_raw_db(filename: str) -> dict:
    if os.path.exists(filename):
        print(f"Loading existing position database from {filename}...")
        with open(filename, 'rb') as f:
            try: return pickle.load(f)
            except Exception: return {}
    else:
        print("No existing position database found. Starting fresh.")
        return {}

def load_processed_data(filename: str) -> list:
    if os.path.exists(filename):
        print(f"Loading existing processed data from {filename}...")
        with open(filename, 'rb') as f:
            try: return pickle.load(f)
            except Exception: return []
    else:
        print(f"No existing processed data found at {filename}. Starting fresh.")
        return []

def save_raw_db(filename: str, data: dict):
    print(f"Saving updated position database ({len(data)} positions) to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def save_processed_data(filename: str, data: list):
    print(f"Saving processed training data ({len(data)} samples) to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

# --- UPDATED FUNCTION: RENAMED AND USES PRUNING-ONLY STRATEGY ---
def create_value_uniform_dataset_by_pruning(training_data: list) -> list:
    """Creates a uniform dataset based on position *value* by pruning all bins to the minimum bin size."""
    print("Creating uniform *value* distribution dataset (Pruning to Min Height)...")
    if not training_data:
        print("Cannot create uniform dataset: No training data.")
        return []
        
    binned_data = {i: [] for i in range(20)}
    for sample in training_data:
        value = sample[1] # Bin by value
        bin_index = 19 if value == 1.0 else int(value * 20)
        binned_data[bin_index].append(sample)
        
    bin_heights = [len(samples) for samples in binned_data.values() if samples]
    if not bin_heights:
        print("No data in any bins. Skipping uniform creation.")
        return []
        
    # New Strategy: Prune to minimum height, don't resample
    target_bin_height = min(bin_heights)
    total_samples = sum(bin_heights)
    non_empty_bins = len(bin_heights)

    print(f"Total samples: {total_samples}, Non-empty bins: {non_empty_bins}")
    print(f"Target (min) bin height: {target_bin_height}")
    
    uniform_data = []
    for bin_index, samples in binned_data.items():
        if not samples:
            continue
            
        # Prune all bins down to the target minimum height
        random.shuffle(samples)
        samples_to_keep = samples[:target_bin_height]
        uniform_data.extend(samples_to_keep)

    print(f"Created value-uniform dataset with {len(uniform_data)} samples (by pruning).")
    # No longer returns pruned_data
    return uniform_data
# --- END UPDATED FUNCTION ---

# --- NEW FUNCTION FOR POLICY UNIFORMITY ---
def create_policy_uniform_dataset_by_pruning(training_data: list) -> list:
    """Creates a uniform dataset based on *move* (policy) by pruning all move bins to the minimum bin size."""
    print("Creating uniform *policy* distribution dataset (Pruning to Min Height)...")
    if not training_data:
        print("Cannot create uniform dataset: No training data.")
        return []
        
    binned_data = {}
    for sample in training_data:
        move_uci = sample[2] # Bin by policy (move)
        if move_uci not in binned_data:
            binned_data[move_uci] = []
        binned_data[move_uci].append(sample)
        
    bin_heights = [len(samples) for samples in binned_data.values() if samples]
    if not bin_heights:
        print("No data in any move bins. Skipping uniform creation.")
        return []
        
    # New Strategy: Prune to minimum height
    target_bin_height = min(bin_heights)
    total_samples = sum(bin_heights)
    non_empty_bins = len(bin_heights)

    print(f"Total samples: {total_samples}, Non-empty move bins: {non_empty_bins}")
    print(f"Target (min) move count: {target_bin_height}")
    
    uniform_data = []
    for move_uci, samples in binned_data.items():
        if not samples:
            continue
            
        # Prune all bins down to the target minimum height
        random.shuffle(samples)
        samples_to_keep = samples[:target_bin_height]
        uniform_data.extend(samples_to_keep)

    print(f"Created policy-uniform dataset with {len(uniform_data)} samples (by pruning).")
    return uniform_data
# --- END NEW FUNCTION ---


def make_sample_hashable(sample: tuple) -> tuple:
    # This is no longer used to clean pruned data, but might be useful later.
    board_tensor, value, policy_str = sample
    hashable_tensor = board_tensor.tobytes()
    return (hashable_tensor, value, policy_str)

def main():
    if not STOCKFISH_PATH or not os.path.exists(STOCKFISH_PATH):
        print("="*50); print(f"ERROR: Stockfish not found at: {STOCKFISH_PATH}"); print("="*50)
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
            # We might still want to re-process existing data if flags changed
            # but for now, this is fine.
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
        print(f"Total validation samples: {len(all_val_data)}") # This is before pruning logic

        # --- UPDATED UNIFORMITY AND SAVING LOGIC ---
        
        if FORCE_VALUE_UNIFORM:
            # This function now only returns the uniform data (pruning-only)
            value_uniform_data = create_value_uniform_dataset_by_pruning(all_train_data)
            
            if value_uniform_data:
                save_processed_data(VALUE_UNIFORM_TRAIN_FILE, value_uniform_data)
                # plot_value_distribution(value_uniform_data, title_suffix=" (Value Uniform Training Set)") <-- REMOVED
        
        if FORCE_POLICY_UNIFORM:
            # Create a separate policy-uniform dataset from the *original* training data
            policy_uniform_data = create_policy_uniform_dataset_by_pruning(all_train_data)
            
            if policy_uniform_data:
                save_processed_data(POLICY_UNIFORM_TRAIN_FILE, policy_uniform_data)
                # Plot the new policy distribution
                # plot_policy_distribution(policy_uniform_data, title_suffix=" (Policy Uniform Training Set)") <-- REMOVED
        
        # Plot the original, non-uniform distributions for comparison
        # plot_value_distribution(all_train_data, title_suffix=" (Original Training Set)") <-- REMOVED
        # plot_policy_distribution(all_train_data, title_suffix=" (Original Training Set)") <-- REMOVED
        
        # The `if pruned_data:` block has been REMOVED as requested.
        # Pruned data is no longer added to the validation set.
        
        # Save the full, original training data
        save_processed_data(ORIGINAL_TRAIN_FILE, all_train_data)
        
        # Save the validation data (which was NOT modified with pruned data)
        save_processed_data(VALIDATION_FILE, all_val_data)
        print(f"New total validation samples: {len(all_val_data)}")
        # --- END UPDATED LOGIC ---

    except Exception as e:
        print(f"\nAn unexpected error occurred in main: {e}")
    
    print("Script finished.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    main()
    
    # # --- PROFILING SETUP ---
    # import cProfile, pstats
    # import io # To print stats
    
    # profiler = cProfile.Profile()
    # profiler.enable()
    # # --- END PROFILING SETUP ---

    # print("--- Starting main() under profiler ---")
    # main() 
    # print("--- Finished main() ---")

    # # --- PROFILING RESULTS ---
    # profiler.disable()
    
    # print("\n\n" + "="*50)
    # print("--- PROFILING RESULTS (Main Thread) ---")
    # print("="*50 + "\n")
    
    # # --- Option 1: Print to console ---
    # s = io.StringIO()
    # # Sort by cumulative time spent in the function and its sub-functions
    # stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    # stats.print_stats(40) # Print top 40 most expensive functions
    # print(s.getvalue())

    # # --- Option 2: Save for Snakeviz (Recommended) ---
    # profiler_output_file = 'main_thread.prof'
    # profiler.dump_stats(profiler_output_file)
    # print(f"\n--- Profiling data saved to '{profiler_output_file}' ---")
    # print(f"--- To view, run:  pip install snakeviz  ---")
    # print(f"--- Then run:      snakeviz {profiler_output_file}     ---")
    # print("="*50)
    # --- END PROFILING RESULTS ---