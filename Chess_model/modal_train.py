import os
import modal

VOLUME_NAME = "chess-data-volume"
DATA_MOUNT_PATH = "/data"

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "tqdm",
        "numpy",
        # add these if train.py or chess_cnn.py need them:
        # "python-chess",
        # "matplotlib",
    )
    # Ship train.py and chess_cnn.py as importable modules
    .add_local_python_source("train")
    .add_local_python_source("chess_cnn")
)

app = modal.App("chess-train", image=image)

data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    # remove gpu=... if you want CPU-only
    gpu="A100-40GB",
    timeout=60 * 60 * 12,
    volumes={DATA_MOUNT_PATH: data_volume},
)
def train_remote(
    num_epochs: int = 10,
    batch_size: int = 256,
    num_workers: int = 4,
):
    """
    Run train.main() inside Modal using dataset from the chess-data-volume.
    """

    import sys

    print("=== Inside train_remote ===")
    print("cwd:", os.getcwd())
    print("sys.path:", sys.path)

    # Import your local training script as a module (bundled via add_local_python_source)
    import train as tr

    # Point train.py's paths to the Volume (where your .pkl files live)
    tr.TRAIN_DATA_FILE = f"{DATA_MOUNT_PATH}/data/chess_training_data_value_uniform.pkl"
    tr.VAL_DATA_FILE   = f"{DATA_MOUNT_PATH}/data/validation.pkl"

    # Save model into the Volume so you can download it later
    tr.MODEL_SAVE_PATH = f"{DATA_MOUNT_PATH}/chess_cnn1.pth"

    # Override hyperparameters without editing train.py
    tr.NUM_EPOCHS  = num_epochs
    tr.BATCH_SIZE  = batch_size
    tr.NUM_WORKERS = num_workers

    print("Starting remote training with:")
    print(f"  TRAIN_DATA_FILE = {tr.TRAIN_DATA_FILE}")
    print(f"  VAL_DATA_FILE   = {tr.VAL_DATA_FILE}")
    print(f"  MODEL_SAVE_PATH = {tr.MODEL_SAVE_PATH}")
    print(f"  NUM_EPOCHS      = {tr.NUM_EPOCHS}")
    print(f"  BATCH_SIZE      = {tr.BATCH_SIZE}")
    print(f"  NUM_WORKERS     = {tr.NUM_WORKERS}")

    tr.main()

    print("Remote training finished.")


@app.local_entrypoint()
def main(
    num_epochs: int = 10,
    batch_size: int = 256,
    num_workers: int = 4,
):
    """
    Local entrypoint: runs one training job on Modal.

    Usage:

      modal run modal_train.py::main
      modal run modal_train.py::main --num-epochs=20 --batch-size=512 --num-workers=8
    """
    train_remote.remote(
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_workers=num_workers,
    )