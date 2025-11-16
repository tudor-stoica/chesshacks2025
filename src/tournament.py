import os
import sys
import importlib
import chess
import chess.pgn
import chess.engine
import time
import math
from typing import Optional, Tuple

from utils import GameContext

# --- Configuration ---
# ADD YOUR BOT FILENAMES HERE (without .py)
BOT_NAMES = [
    'main',
    # 'main_fengtaowang',
    'monteCarloImprove',
    # 'alpha'
]

# !! IMPORTANT !!
# Set this to the full path of your Stockfish executable (e.g., "C:/stockfish/stockfish.exe" or "/usr/games/stockfish")
# If you don't have Stockfish or don't want evaluations, set this to None.
# You can download Stockfish from: https://stockfishchess.org/download/
STOCKFISH_PATH: Optional[str] = "../stockfish" 

K_FACTOR = 32             # Elo K-factor for rating calculations
STARTING_ELO = 1500       # Elo all bots start with
PLAYER_TIME_LIMIT_SECONDS = 60 # Max time PER PLAYER for a whole game
# ---------------------

# Add the current directory to sys.path to ensure imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

class BotWrapper:
    """A wrapper class to hold a loaded bot module and its stats."""
    def __init__(self, name: str, module):
        self.name = name
        self.module = module
        self.elo = STARTING_ELO
        
        # Get the registered functions from the bot's manager
        # We assume the bot's code has 'import bot_framework'
        # and has 'chess_manager = bot_framework.BotManager()'
        bot_manager = getattr(module, 'chess_manager', None)
        if bot_manager is None:
             raise AttributeError(f"Bot {name} does not have a 'chess_manager' instance. Did you add 'import bot_framework' and 'chess_manager = bot_framework.BotManager()' to it?")
        
        self.entrypoint = bot_manager.entrypoint_func
        self.reset = bot_manager.reset_func

        if not callable(self.entrypoint) or not callable(self.reset):
            raise TypeError(f"Bot {name}'s entrypoint or reset function is not callable.")

    def __repr__(self):
        return f"<BotWrapper: {self.name} (Elo: {self.elo:.0f})>"


def calculate_elo_update(
    elo_a: float, elo_b: float, score_a: float
) -> Tuple[float, float]:
    """
    Calculates the new Elo ratings for two players after a game.
    :param elo_a: Player A's current Elo
    :param elo_b: Player B's current Elo
    :param score_a: Player A's score (1.0 for win, 0.5 for draw, 0.0 for loss)
    :return: (new_elo_a, new_elo_b)
    """
    expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    expected_b = 1 - expected_a

    score_b = 1.0 - score_a

    new_elo_a = elo_a + K_FACTOR * (score_a - expected_a)
    new_elo_b = elo_b + K_FACTOR * (score_b - expected_b)

    return new_elo_a, new_elo_b


def play_game(
    white_bot: BotWrapper,
    black_bot: BotWrapper,
    engine: Optional[chess.engine.SimpleEngine],
) -> Tuple[float, float, str, str]:
    """
    Plays a single game between two bots.
    Logs time taken for each move and remaining time.
    :return: (white_score, black_score, game_result_str, reason_str)
    """
    board = chess.Board()
    
    # --- PGN Setup ---
    game = chess.pgn.Game()
    game.headers["Event"] = "Bot Tournament"
    game.headers["Site"] = "Local"
    game.headers["Date"] = time.strftime("%Y.%m.%d", time.gmtime())
    game.headers["Round"] = "?"
    game.headers["White"] = white_bot.name
    game.headers["Black"] = black_bot.name
    pgn_node = game  # Start at the root node
    # -----------------

    # Per-player time limits
    white_time_remaining = PLAYER_TIME_LIMIT_SECONDS
    black_time_remaining = PLAYER_TIME_LIMIT_SECONDS

    # Create a basic context for each bot
    try:
        white_bot.reset(GameContext(board.copy(), white_time_remaining, logProbabilities=lambda p: None))
    except Exception as e:
        print(f"Error resetting white bot {white_bot.name}: {e}")
        return 0.0, 1.0, "0-1", f"{white_bot.name} failed to reset"

    try:
        black_bot.reset(GameContext(board.copy(), black_time_remaining, logProbabilities=lambda p: None))
    except Exception as e:
        print(f"Error resetting black bot {black_bot.name}: {e}")
        return 1.0, 0.0, "1-0", f"{black_bot.name} failed to reset"

    game_start_time = time.time()
    move_count = 1

    while not board.is_game_over(claim_draw=True):
        
        is_white_turn = (board.turn == chess.WHITE)
        
        if is_white_turn:
            bot = white_bot
            ctx = GameContext(board.copy(), white_time_remaining, lambda p: None)
            player_str = f"White ({bot.name})"
        else:
            bot = black_bot
            ctx = GameContext(board.copy(), black_time_remaining, lambda p: None)
            player_str = f"Black ({bot.name})"
            
        # Display move number for White's turn
        if is_white_turn:
            print(f"--- Move {move_count} ---")
        
        move_obj = None # To store the chess.Move object
        move_uci = ""   # To store the string representation
        move_input = None # To store the raw bot output

        try:
            move_start_time = time.time()
            try:
                move_input = bot.entrypoint(ctx) # This could be a Move object or a UCI string
            except Exception as e:
                print(f"!! Bot {bot.name} raised an exception during its turn: {e}")
                if is_white_turn:
                    return 0.0, 1.0, "0-1", f"{bot.name} (White) raised an exception"
                else:
                    return 1.0, 0.0, "1-0", f"{bot.name} (Black) raised an exception"
            move_end_time = time.time()
            
            move_duration = move_end_time - move_start_time
            
            # --- Time Check ---
            # Subtract time and check for timeout
            if is_white_turn:
                white_time_remaining -= move_duration
                if white_time_remaining <= 0:
                    print(f"!! {player_str} ran out of time (took {move_duration:.2f}s)")
                    return 0.0, 1.0, "0-1", f"{bot.name} (White) ran out of time"
            else:
                black_time_remaining -= move_duration
                if black_time_remaining <= 0:
                    print(f"!! {player_str} ran out of time (took {move_duration:.2f}s)")
                    return 1.0, 0.0, "1-0", f"{bot.name} (Black) ran out of time"

            # --- Move Validation ---
            if isinstance(move_input, chess.Move):
                move_obj = move_input
                move_uci = move_obj.uci()
            elif isinstance(move_input, str):
                move_uci = move_input
                try:
                    move_obj = board.parse_uci(move_uci)
                except (ValueError, TypeError):
                    move_obj = None # Will fail the next check
            
            if move_obj is None or move_obj not in board.legal_moves:
                reason = f"{player_str} made an illegal move: {move_input}"
                print(f"!! {reason}")
                if is_white_turn:
                    return 0.0, 1.0, "0-1", reason
                else:
                    return 1.0, 0.0, "1-0", reason
            
            # --- Log the move and time ---
            remaining_time = white_time_remaining if is_white_turn else black_time_remaining
            print(f"  {player_str:<25} | Move: {move_uci:<6} | Took: {move_duration:5.2f}s | Remaining: {remaining_time:5.2f}s")


        except Exception as e:
            print(f"!! Error during {player_str}'s move: {e}")
            if is_white_turn:
                return 0.0, 1.0, "0-1", f"{bot.name} (White) raised an exception"
            else:
                return 1.0, 0.0, "1-0", f"{bot.name} (Black) raised an exception"

        # Push the validated move
        board.push(move_obj)
        
        # --- Add move to PGN ---
        pgn_node = pgn_node.add_variation(move_obj)
        
        # Increment move count after Black moves
        if not is_white_turn:
            move_count += 1

    # Game is over
    game_duration = time.time() - game_start_time
    print(f"Game over after {game_duration:.2f}s.")
    result = board.result(claim_draw=True)
    
    # --- Set PGN Result ---
    game.headers["Result"] = result

    # --- Print PGN ---
    print("\n--- PGN ---")
    print(game)
    print("-----------")

    # --- Stockfish Evaluation ---
    if engine:
        try:
            print("Getting Stockfish evaluation...")
            # Use a short time limit for a quick analysis
            info = engine.analyse(board, chess.engine.Limit(time=0.1))
            score = info.get("score")
            
            if score:
                # Get the score from White's perspective
                print(f"Final Position Evaluation: {score.white()}")
            else:
                print("Could not get a score from Stockfish.")
        except Exception as e:
            print(f"!! Error during Stockfish analysis: {e}")
    
    if result == "1-0":
        return 1.0, 0.0, "1-0", "Checkmate or resignation"
    elif result == "0-1":
        return 0.0, 1.0, "0-1", "Checkmate or resignation"
    else:
        # e.g., "1/2-1/2"
        reason = "Draw"
        if board.is_stalemate():
            reason = "Draw (Stalemate)"
        elif board.is_insufficient_material():
            reason = "Draw (Insufficient Material)"
        elif board.is_seventyfive_moves():
            reason = "Draw (75-move rule)"
        elif board.is_fivefold_repetition():
            reason = "Draw (5-fold repetition)"
        elif board.can_claim_draw():
            reason = "Draw (3-fold repetition or 50-move rule)"
            
        return 0.5, 0.5, "1/2-1/2", reason


def load_bots_from_names(bot_names: list[str]) -> list[BotWrapper]:
    """Dynamically loads all valid bot modules from the list of names."""
    bots = []
    
    print(f"Loading bots...")
    for bot_name in bot_names:
        try:
            # Import the module
            module = importlib.import_module(bot_name)
            # Force a reload in case it's been changed
            importlib.reload(module)
            
            # Wrap the loaded module
            bot_wrapper = BotWrapper(bot_name, module)
            bots.append(bot_wrapper)
            print(f"  > Successfully loaded bot: {bot_name}")
        except Exception as e:
            print(f"  > FAILED to load bot {bot_name}: {e}")
            print(f"    Check that the file exists and has no errors. (Error: {e})") # Added more detail
    
    return bots


def main():
    """Main tournament loop."""
    print("--- Chess Bot Tournament ---")

    # --- Load Stockfish Engine ---
    engine: Optional[chess.engine.SimpleEngine] = None
    if STOCKFISH_PATH:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            print(f"Stockfish engine loaded from: {STOCKFISH_PATH}")
        except FileNotFoundError:
            print(f"!! WARNING: Stockfish executable not found at '{STOCKFISH_PATH}'.")
            print("!! You must download Stockfish and set the STOCKFISH_PATH variable.")
            print("!! No engine evaluations will be available.")
        except Exception as e:
            print(f"!! WARNING: Failed to load Stockfish engine: {e}")
            print("!! No engine evaluations will be available.")
    else:
        print("Stockfish path not set. No engine evaluations will be available.")
    # -----------------------------

    try:
        bots = load_bots_from_names(BOT_NAMES)

        if len(bots) < 2:
            print("Tournament cancelled: Need at least 2 valid bots to play.")
            return

        print(f"\nStarting round-robin tournament with {len(bots)} bots.")
        
        # Store scores for ranking
        scores = {bot.name: 0.0 for bot in bots}
        
        # Simple round-robin: each bot plays every other bot twice
        for i in range(len(bots)):
            for j in range(len(bots)):
                if i == j:
                    continue

                bot_a = bots[i]
                bot_b = bots[j]

                # --- Game 1: A as White, B as Black ---
                print(f"\nMatch: {bot_a.name} (White) vs {bot_b.name} (Black)")
                # Pass the engine to the play_game function
                score_a, score_b, result_str, reason = play_game(bot_a, bot_b, engine)
                print(f"Result: {result_str} ({reason})")

                # Store scores
                scores[bot_a.name] += score_a
                scores[bot_b.name] += score_b

                # Calculate and apply Elo update
                old_elo_a, old_elo_b = bot_a.elo, bot_b.elo
                new_elo_a, new_elo_b = calculate_elo_update(bot_a.elo, bot_b.elo, score_a)
                
                print(f"Elo: {bot_a.name} ({old_elo_a:+.0f} -> {new_elo_a:+.0f}) | "
                      f"{bot_b.name} ({old_elo_b:+.0f} -> {new_elo_b:+.0f})")
                bot_a.elo, bot_b.elo = new_elo_a, new_elo_b


        # All games finished, print final standings
        print("\n--- Tournament Finished ---")
        print("Final Standings (by Elo):")

        sorted_bots_elo = sorted(bots, key=lambda b: b.elo, reverse=True)
        for rank, bot in enumerate(sorted_bots_elo, 1):
            print(f"  {rank}. {bot.name:<20} {bot.elo:.2f} Elo")

        print("\nFinal Standings (by Score):")
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        for rank, (name, score) in enumerate(sorted_scores, 1):
            print(f"  {rank}. {name:<20} {score} points")

    finally:
        # --- Quit Stockfish Engine ---
        if engine:
            print("\nQuitting Stockfish engine.")
            engine.quit()
        # -----------------------------


if __name__ == "__main__":
    main()