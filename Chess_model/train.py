 # Save this file as train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from tqdm import tqdm
from chess_cnn import ChessCNN

TRAIN_DATA_FILE = '../chess_training_data_value_uniform.pkl'
VAL_DATA_FILE = '../validation.pkl'
MODEL_SAVE_PATH = 'chess_cnn.pth'
NUM_WORKERS = 1

BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
L1_LAMBDA = 1e-5

def load_data_and_create_maps(train_file, val_file):
    """
    Loads the processed data from both train and validation files
    and creates a unified mapping from move strings to an integer index.
    """
    print(f"Loading training data from {train_file}...")
    if not os.path.exists(train_file):
        print(f"Error: Training file not found at {train_file}")
        exit(1)
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    if not train_data:
        print("Error: Training data file is empty.")
        exit(1)
    print(f"Loaded {len(train_data)} training samples.")

    print(f"Loading validation data from {val_file}...")
    if not os.path.exists(val_file):
        print(f"Error: Validation file not found at {val_file}")
        exit(1)
    with open(val_file, 'rb') as f:
        val_data = pickle.load(f)
    if not val_data:
        print("Error: Validation data file is empty.")
        exit(1)
    print(f"Loaded {len(val_data)} validation samples.")

    train_moves = set(item[2] for item in train_data)
    val_moves = set(item[2] for item in val_data)
    policy_moves = train_moves | val_moves

    INDEX_TO_MOVE = sorted(list(policy_moves))
    MOVE_TO_INDEX = {move: i for i, move in enumerate(INDEX_TO_MOVE)}

    num_policy_outputs = len(INDEX_TO_MOVE)
    print(f"Created unified policy map with {num_policy_outputs} unique moves.")

    return train_data, val_data, MOVE_TO_INDEX, INDEX_TO_MOVE, num_policy_outputs

class ChessDataset(Dataset):
    """Custom PyTorch Dataset for loading the chess data."""

    def __init__(self, data, move_to_index_map):
        self.data = data
        self.move_to_index_map = move_to_index_map

        # Debug: check the structure of one sample
        if self.data:
            print("Sample training entry:", self.data[0])
        else:
            print("Warning: Dataset is empty.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board_tensor, value_target, policy_target_str = self.data[idx]

        # 1. Board Tensor
        board_tensor = torch.tensor(board_tensor, dtype=torch.float32)

        # 2. Value Target
        value_target = torch.tensor([value_target], dtype=torch.float32)

        # 3. Policy Target
        policy_target_idx = self.move_to_index_map[policy_target_str]
        policy_target = torch.tensor(policy_target_idx, dtype=torch.long)

        return board_tensor, value_target, policy_target

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data and create mappings
    train_data, val_data, MOVE_TO_INDEX, INDEX_TO_MOVE, NUM_POLICY_OUTPUTS = \
        load_data_and_create_maps(TRAIN_DATA_FILE, VAL_DATA_FILE)

    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    # 2. Create Datasets and DataLoaders
    train_dataset = ChessDataset(train_data, MOVE_TO_INDEX)
    val_dataset = ChessDataset(val_data, MOVE_TO_INDEX)

    num_workers_to_use = NUM_WORKERS
    print(f"Using {num_workers_to_use} workers for data loading.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers_to_use,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers_to_use > 0)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers_to_use,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers_to_use > 0)
    )

    # 3. Initialize Model, Losses, and Optimizer
   
    # --- Create model structure first ---
    model = ChessCNN(num_policy_outputs=NUM_POLICY_OUTPUTS).to(device)

    # --- Load pre-existing model if it exists ---
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Found existing model at {MODEL_SAVE_PATH}. Loading weights...")
        try:
            # Load the state_dict, mapping to the correct device
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Continuing with a new, un-trained model.")
    else:
        print(f"No model found at {MODEL_SAVE_PATH}. Creating a new model.")
    # --- End of new logic ---

    value_loss_fn = nn.MSELoss()
    policy_loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training for {NUM_EPOCHS} epochs...")

    # 4. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss_v = 0.0
        total_train_loss_p = 0.0
        total_correct_p = 0
        total_samples_p = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for boards, values, policies in progress_bar:
            boards = boards.to(device)
            values = values.to(device)
            policies = policies.to(device)

            optimizer.zero_grad()
            pred_values, pred_policy_logits = model(boards)

            # Ensure value prediction shape matches target
            pred_values = pred_values.view_as(values)

            loss_v = value_loss_fn(pred_values, values)
            loss_p = policy_loss_fn(pred_policy_logits, policies)

            # Compute L1 penalty efficiently
            l1_norm = 0.0
            for param in model.parameters():
                l1_norm += torch.abs(param).sum()

            total_loss = loss_v + loss_p + (L1_LAMBDA * l1_norm)

            total_loss.backward()
            optimizer.step()

            total_train_loss_v += loss_v.item()
            total_train_loss_p += loss_p.item()

            _, pred_indices = torch.max(pred_policy_logits, 1)
            total_correct_p += (pred_indices == policies).sum().item()
            total_samples_p += policies.size(0)

            progress_bar.set_postfix(v_loss=f"{loss_v.item():.4f}", p_loss=f"{loss_p.item():.4f}")

        # --- Validation ---
        model.eval()
        total_val_loss_v = 0.0
        total_val_loss_p = 0.0
        total_val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for boards, values, policies in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                boards = boards.to(device)
                values = values.to(device)
                policies = policies.to(device)

                pred_values, pred_policy_logits = model(boards)
                pred_values = pred_values.view_as(values)

                loss_v = value_loss_fn(pred_values, values)
                loss_p = policy_loss_fn(pred_policy_logits, policies)

                total_val_loss_v += loss_v.item()
                total_val_loss_p += loss_p.item()

                _, pred_indices = torch.max(pred_policy_logits, 1)
                total_val_correct += (pred_indices == policies).sum().item()
                total_val_samples += policies.size(0)

        # --- Epoch Summary ---
        avg_train_loss_v = total_train_loss_v / len(train_loader)
        avg_train_loss_p = total_train_loss_p / len(train_loader)
        avg_train_acc_p = total_correct_p / total_samples_p

        avg_val_loss_v = total_val_loss_v / len(val_loader)
        avg_val_loss_p = total_val_loss_p / len(val_loader)
        avg_val_acc_p = total_val_correct / total_val_samples

        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"  [Train] Value Loss: {avg_train_loss_v:.4f} | Policy Loss: {avg_train_loss_p:.4f} | Policy Acc: {avg_train_acc_p*100:.2f}%")
        print(f"  [Val]   Value Loss: {avg_val_loss_v:.4f} | Policy Loss: {avg_val_loss_p:.4f} | Policy Acc: {avg_val_acc_p*100:.2f}%")
        print("-" * (30 + len(str(epoch+1))))

        # 5. Save the trained model
        print(f"Training complete. Saving model to {MODEL_SAVE_PATH}")
        # Save the model's state_dict, which is the recommended way
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # Save the move map as well
    # Updated this line to correctly replace .pth
    map_save_path = MODEL_SAVE_PATH.replace(".pth", "_move_map.pkl")
    with open(map_save_path, 'wb') as f:
        pickle.dump({'MOVE_TO_INDEX': MOVE_TO_INDEX, 'INDEX_TO_MOVE': INDEX_TO_MOVE}, f)
    print(f"Saved move mapping to {map_save_path}")

if __name__ == "__main__":
    main() 