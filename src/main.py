from .utils import chess_manager, GameContext
from chess import Move
import random
import time

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

def get_model_info(board, model):
    value, policy = model(board)
    return value, policy


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.1)

    legal_moves = list(ctx.board.generate_legal_moves())

    return random.choice(legal_moves)


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
