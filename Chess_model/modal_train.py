import modal
from pathlib import Path

PROJECT_REMOTE_PATH = "/root/chess_project"

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "tqdm",
        "python-chess",
        "apache-beam",
        "numpy",
    )
    .add_local_dir(".", PROJECT_REMOTE_PATH)
)

app = modal.App("chess-cnn-modal-train", image=image)

volume = modal.Volume.from_name("chess-data")
DATA_ROOT = Path("/data")          # inside container
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR  = DATA_ROOT / "test"
CKPT_DIR  = DATA_ROOT / "checkpoints"


@app.function(
    gpu="T4",                      # or comment out for CPU only
    image=image,
    volumes={str(DATA_ROOT): volume},
    timeout=60 * 60,
)
def train_remote(
    train_bag_file: str = "action_value-00087-of-02148_data.bag",
    num_epochs: int = 10,
    batch_size: int = 128,
):
    """
    Runs your training loop on Modal.
    `train_bag_file` is relative to /data/train.
    """
    import os
    import sys
    import time

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    # Make sure Python can see the bundled project code
    if PROJECT_REMOTE_PATH not in sys.path:
        sys.path.append(PROJECT_REMOTE_PATH)

    # Now we can import your local modules
    import chess_cnn
    import chess_utils
    import bag_dataset

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    NUM_POLICY_OUTPUTS = chess_utils.NUM_POLICY_OUTPUTS

    TRAIN_BAG_PATH = TRAIN_DIR / train_bag_file
    TEST_BAG_PATH  = TEST_DIR / "action_value_data.bag"  # adjust if needed

    print(f"Training on: {TRAIN_BAG_PATH}")
    print(f"Testing on:  {TEST_BAG_PATH}")

    # --- Model ---
    model = chess_cnn.ChessCNN(num_policy_outputs=NUM_POLICY_OUTPUTS).to(DEVICE)

    # --- Losses ---
    value_loss_fn = nn.MSELoss()
    policy_loss_fn = nn.CrossEntropyLoss()

    # --- Optimizer (using your 3 param groups) ---
    print("Setting up optimizer parameter groups...")
    body_params = list(model.conv_in.parameters()) + \
                  list(model.bn_in.parameters()) + \
                  list(model.res_stack.parameters())

    value_head_params = list(model.value_conv.parameters()) + \
                        list(model.value_bn.parameters()) + \
                        list(model.value_fc1.parameters()) + \
                        list(model.value_fc2.parameters())

    policy_head_params = list(model.policy_conv.parameters()) + \
                         list(model.policy_bn.parameters()) + \
                         list(model.policy_fc1.parameters())

    optimizer = optim.Adam([
        {'params': body_params, 'lr': 1e-3},
        {'params': value_head_params, 'lr': 1e-3},
        {'params': policy_head_params, 'lr': 5e-3},
    ])

    # --- Data ---
    train_data = bag_dataset.ActionValueDataset(filepath=str(TRAIN_BAG_PATH))
    test_data  = bag_dataset.ActionValueDataset(filepath=str(TEST_BAG_PATH))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    test_loader  = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=4,
    )

    os.makedirs(CKPT_DIR, exist_ok=True)

    # --- Helper: train & test (same logic as your local script) ---
    def train_epoch(epoch_num: int):
        model.train()
        total_loss = 0.0
        total_loss_v = 0.0
        total_loss_p = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch_num}/{num_epochs} Training", leave=False)
        for batch, (X, y_value, y_policy) in enumerate(loop):
            X = X.to(DEVICE)
            y_value = y_value.to(DEVICE)
            y_policy = y_policy.to(DEVICE)

            pred_value, pred_policy = model(X)
            loss_v = value_loss_fn(pred_value, y_value)
            loss_p = policy_loss_fn(pred_policy, y_policy)
            loss = 2.0 * loss_v + loss_p

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_loss_v += loss_v.item()
            total_loss_p += loss_p.item()

            loop.set_postfix(
                loss=f"{total_loss / (batch + 1):.4f}",
                loss_v=f"{total_loss_v / (batch + 1):.4f}",
                loss_p=f"{total_loss_p / (batch + 1):.4f}",
            )

    def test_epoch():
        size = len(test_loader.dataset)
        num_batches = len(test_loader)
        model.eval()

        total_loss = total_loss_v = total_loss_p = 0.0
        policy_correct = 0

        with torch.no_grad():
            loop = tqdm(test_loader, desc="Testing", leave=False)
            for X, y_value, y_policy in loop:
                X = X.to(DEVICE)
                y_value = y_value.to(DEVICE)
                y_policy = y_policy.to(DEVICE)

                pred_value, pred_policy = model(X)
                loss_v = value_loss_fn(pred_value, y_value)
                loss_p = policy_loss_fn(pred_policy, y_policy)

                total_loss_v += loss_v.item()
                total_loss_p += loss_p.item()
                total_loss += (2.0 * loss_v + loss_p).item()

                policy_correct += (pred_policy.argmax(1) == y_policy).float().sum().item()

                loop.set_postfix(
                    avg_loss=f"{total_loss / (loop.n + 1):.4f}",
                    acc=f"{(100 * policy_correct / size):.1f}%",
                )

        avg_combined_loss = total_loss / num_batches
        avg_value_loss = total_loss_v / num_batches
        avg_policy_loss = total_loss_p / num_batches
        policy_accuracy = policy_correct / size

        print(f"\n--- Test Results ---")
        print(f"  Policy Accuracy: {(100 * policy_accuracy):>0.1f}%")
        print(f"  Value Loss:      {avg_value_loss:>8f}")
        print(f"  Policy Loss:     {avg_policy_loss:>8f}")
        print(f"  Avg Combined Loss: {avg_combined_loss:>8f}")
        print(f"--------------------\n")

    # --- Main training loop ---
    for epoch in range(1, num_epochs + 1):
        start_t = time.time()
        train_epoch(epoch)
        print(f"Epoch {epoch} training time: {time.time() - start_t:.2f}s")

        test_epoch()

        ckpt_path = CKPT_DIR / f"chess_cnn_epoch_{epoch}.pth"
        print(f"Saving checkpoint to {ckpt_path}...")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_path,
        )
        print("Checkpoint saved.\n")

    print("Remote training done.")


@app.local_entrypoint()
def main(
    train_bag_file: str = "action_value-00087-of-02148_data.bag",
    num_epochs: int = 10,
    batch_size: int = 128,
):
    train_remote.remote(train_bag_file, num_epochs, batch_size)