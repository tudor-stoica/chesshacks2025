import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ResidualBlock (unchanged)
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

# --- Updated ChessCNN Model ---
class ChessCNN(nn.Module):
    def __init__(self, num_policy_outputs):
        super().__init__()
        
        # --- Configuration ---
        self.board_size = 8
        self.half_move_lookback = 0
        self.in_channels = (12 + 4 + 1 + 1) * (1 + self.half_move_lookback)
        
        self.num_channels = 256
        self.num_res_blocks = 10
        head_fc_size = 32
        head_conv_channels = 2
        
        fc1_input_size = head_conv_channels * self.board_size * self.board_size

        # --- 1. Initial Convolutional Layer (Shared Body) ---
        self.conv_in = nn.Conv2d(self.in_channels, self.num_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(self.num_channels)

        # --- 2. Residual Stack (Shared Body) ---
        self.res_stack = nn.ModuleList([ResidualBlock(self.num_channels, self.num_channels) for _ in range(self.num_res_blocks)])
        
        # --- 3. The "Value Head" (Grouped into nn.Sequential) ---
        self.value_head = nn.Sequential(
            nn.Conv2d(self.num_channels, head_conv_channels, kernel_size=1),
            nn.BatchNorm2d(head_conv_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc1_input_size, head_fc_size),
            nn.ReLU(),
            nn.Linear(head_fc_size, 1) # Final output neuron
        )

        # --- 4. The "Policy Head" (Grouped into nn.Sequential) ---
        self.policy_head = nn.Sequential(
            nn.Conv2d(self.num_channels, head_conv_channels, kernel_size=1),
            nn.BatchNorm2d(head_conv_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc1_input_size, num_policy_outputs)
        )

    def forward(self, x):
        # 1. Initial layer (Body)
        x = F.relu(self.bn_in(self.conv_in(x)))
        
        # 2. Pass through all residual blocks (Body)
        for block in self.res_stack:
            x = block(x)
            
        # --- 3. Value Head Path ---
        # Pass the shared "body" output 'x' to the value head
        value_output = self.value_head(x)
        
        # Apply final activation (as in your original code)
        value_logits = F.sigmoid(value_output)
        
        # --- 4. Policy Head Path ---
        # Pass the *same* shared "body" output 'x' to the policy head
        policy_logits = self.policy_head(x)
        
        return value_logits, policy_logits