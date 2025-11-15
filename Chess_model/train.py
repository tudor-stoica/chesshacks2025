import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import chess_cnn

# --- 1. Configuration ---
BATCH_SIZE = 256
NUM_EPOCHS = 5
VALUE_LR = 1e-4
POLICY_LR = 5e-4
BODY_LR = 1e-3

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

class DummyChessDataset(Dataset):
    def __init__(self, num_samples, in_channels, num_policy_outputs):
        self.num_samples = num_samples
        self.in_channels = in_channels
        self.num_policy_outputs = num_policy_outputs
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Dummy input board state
        board_state = torch.randn((self.in_channels, 8, 8))
        
        # Dummy value target (random float between -1 and 1)
        value_target = torch.rand(1) * 2 - 1
        
        # Dummy policy target (a single move index)
        # CrossEntropyLoss expects a single long int, not a one-hot vector
        policy_target = torch.randint(0, self.num_policy_outputs, ()).long()
        
        return board_state, value_target, policy_target

# --- 3. Loss Functions ---
# Use standard nn.MSELoss for the value head (it's the L2 loss)
value_loss_fn = nn.MSELoss()
# Use CrossEntropyLoss for the policy head (classification)
policy_loss_fn = nn.CrossEntropyLoss()


# --- Set up Optimizer with Different Learning Rates ---
print("Setting up optimizer parameter groups...")

# Group 1: Shared "Body" parameters
body_params = list(model.conv_in.parameters()) + \
              list(model.bn_in.parameters()) + \
              list(model.res_stack.parameters())

# Group 2: Value Head parameters
value_head_params = list(model.value_head.parameters())

# Group 3: Policy Head parameters
policy_head_params = list(model.policy_head.parameters())

optimizer = optim.Adam([
    {'params': body_params, 'lr': BODY_LR},
    {'params': value_head_params, 'lr': VALUE_LR},
    {'params': policy_head_params, 'lr': POLICY_LR}
])

print(f"Total param groups: {len(optimizer.param_groups)}")
print(f"  Body LR:   {optimizer.param_groups[0]['lr']}")
print(f"  Value LR:  {optimizer.param_groups[1]['lr']}")
print(f"  Policy LR: {optimizer.param_groups[2]['lr']}")

# --- 5. Create DataLoaders ---
# (Using the Dummy Dataset for this example)
train_data = DummyChessDataset(1000, model.in_channels, NUM_POLICY_OUTPUTS)
test_data = DummyChessDataset(200, model.in_channels, NUM_POLICY_OUTPUTS)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

# --- 6. Training Function ---
def train(dataloader, model, val_loss_fn, pol_loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y_value, y_policy) in enumerate(dataloader):
        # Move data to the device
        X = X.to(DEVICE)
        y_value = y_value.to(DEVICE)
        y_policy = y_policy.to(DEVICE)

        # --- Compute prediction error ---
        # Model returns two heads
        pred_value, pred_policy = model(X)

        # --- Calculate losses for each head ---
        # Note: pred_value is (N, 1) and y_value is (N, 1)
        #       pred_policy is (N, C) and y_policy is (N)
        loss_v = val_loss_fn(pred_value, y_value)
        loss_p = pol_loss_fn(pred_policy, y_policy)
        
        # Combine losses (with 2x weight on value loss, as in your example)
        loss = (loss_v * 2.0) + loss_p

        # --- Backpropagation ---
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss_val = loss.item()
            current = (batch + 1) * len(X)
            print(f"  loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

# --- 7. Test Function ---
def test(dataloader, model, val_loss_fn, pol_loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    
    total_loss, policy_correct = 0, 0

    with torch.no_grad():
        for X, y_value, y_policy in dataloader:
            # Move data to the device
            X = X.to(DEVICE)
            y_value = y_value.to(DEVICE)
            y_policy = y_policy.to(DEVICE)
            
            # --- Forward pass ---
            pred_value, pred_policy = model(X)
            
            # --- Calculate and sum loss ---
            loss_v = val_loss_fn(pred_value, y_value)
            loss_p = pol_loss_fn(pred_policy, y_policy)
            total_loss += ((loss_v * 2.0) + loss_p).item()

            # --- Calculate policy accuracy ---
            policy_correct += (pred_policy.argmax(1) == y_policy).type(torch.float).sum().item()

    avg_loss = total_loss / num_batches
    policy_accuracy = policy_correct / size
    
    print(f"Test Error: ")
    print(f"  Policy Accuracy: {(100*policy_accuracy):>0.1f}%")
    print(f"  Avg Combined loss: {avg_loss:>8f} \n")

# --- 8. Main Execution Loop ---
def main():
    
    model = chess_cnn.ChessCNN(num_policy_outputs=NUM_POLICY_OUTPUTS).to(DEVICE)

    print("Starting Training...\n")
    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train(train_dataloader, model, value_loss_fn, policy_loss_fn, optimizer)
        test(test_dataloader, model, value_loss_fn, policy_loss_fn)
    print("Done!")

if __name__ == "__main__":
    main()