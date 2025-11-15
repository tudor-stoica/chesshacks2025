import os
import modal

# ---------- Modal configuration ----------

VOLUME_NAME = "chess-data-volume"
VOLUME_MOUNT_PATH = "/data"

image = (
    modal.Image.debian_slim()
    .pip_install(
        "python-chess",
        "numpy",
        "matplotlib",
        "tqdm",
    )
    # ship your datagen.py module
    .add_local_python_source("datagen")
    # ship your local Stockfish binary into the image
    # (relative to where you run `modal run`)
    .add_local_file("stockfish", "/usr/local/bin/stockfish")
)

app = modal.App("chess-datagen", image=image)

chess_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    timeout=60 * 60 * 12,
    cpu=64,  # ask Modal for 8 vCPUs – bump if you want more
    volumes={VOLUME_MOUNT_PATH: chess_volume},
)
def generate_data(
    num_games: int = 10_000,
    random_move_prob: float = 0.15,
    cpu_cores: int = 64,
):
    """
    Run datagen.main() inside Modal, using the shared volume and bundled Stockfish.
    """

    import multiprocessing

    # Work inside the mounted volume
    os.chdir(VOLUME_MOUNT_PATH)
    print(f"Working directory (inside container): {os.getcwd()}")

    # Import your script (bundled via add_local_python_source)
    import datagen as dg

    # Point to the Stockfish binary we baked into the image
    dg.STOCKFISH_PATH = "/usr/local/bin/stockfish"

    # Point all data files into the volume (cwd is the volume root)
    dg.RAW_DB_FILE = "chess_positions_db.pkl"
    dg.ORIGINAL_TRAIN_FILE = "chess_training_data_original.pkl"
    dg.VALUE_UNIFORM_TRAIN_FILE = "chess_training_data_value_uniform.pkl"
    dg.POLICY_UNIFORM_TRAIN_FILE = "chess_training_data_policy_uniform.pkl"
    dg.VALIDATION_FILE = "validation.pkl"

    # Override runtime knobs
    dg.NUM_GAMES = num_games
    dg.RANDOM_MOVE_PROB = random_move_prob
    dg.CPU_CORES_TO_USE = cpu_cores

    # Ensure spawn for multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    print("Starting datagen.main() inside Modal...")
    dg.main()
    print("Finished datagen.main().")


@app.local_entrypoint()
def run(
    num_games: int = 10_000,
    random_move_prob: float = 0.15,
    cpu_cores: int = 64,
    runs: int = 1,
):
    """
    Launch one or more data-generation jobs on Modal.

    Examples:

      modal run modal_datagen.py::run

      modal run modal_datagen.py::run --num-games=20000 --cpu-cores=16

      modal run modal_datagen.py::run --runs=3
    """

    for i in range(runs):
        print(f"\n=== Launching job {i+1}/{runs} ===")
        generate_data.remote(
            num_games=num_games,
            random_move_prob=random_move_prob,
            cpu_cores=cpu_cores,
        )
        print(f"=== Job {i+1}/{runs} completed ===")