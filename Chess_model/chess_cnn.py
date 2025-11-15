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

# Define the "Leaner" model
class ChessCNN(nn.Module):
    def __init__(self, num_policy_outputs):
        super().__init__()
        
        # --- Configuration ---
        self.board_size = 8
        self.half_move_lookback = 0
        # pieces + castling + en passent sqaure + 50 move clock normilized to (0-1)
        self.in_channels = (12 + 4 + 1 + 1) * (1 + self.half_move_lookback)
        
        self.num_channels = 100
        self.num_res_blocks = 8
        head_fc_size = 32
        head_conv_channels = 2
        
        fc1_input_size = head_conv_channels * self.board_size * self.board_size

        # --- 1. Initial Convolutional Layer ---
        self.conv_in = nn.Conv2d(self.in_channels, self.num_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(self.num_channels)

        # --- 2. Residual Stack ---
        self.res_stack = nn.ModuleList([ResidualBlock(self.num_channels, self.num_channels) for _ in range(self.num_res_blocks)])
        
        self.flatten = nn.Flatten()
        
        # --- 3. The "Value Head" ---
        def value_head():
            self.value_conv = nn.Conv2d(self.num_channels, head_conv_channels, kernel_size=1)
            self.value_bn = nn.BatchNorm2d(head_conv_channels)
            self.value_fc1 = nn.Linear(fc1_input_size, head_fc_size)
            self.value_fc2 = nn.Linear(head_fc_size, 1) # Final output neuron

        # --- 4. The "Policy Head" ---
        def policy_head():
            self.policy_conv = nn.Conv2d(self.num_channels, head_conv_channels, kernel_size=1)
            self.policy_bn = nn.BatchNorm2d(head_conv_channels)
            self.policy_fc1 = nn.Linear(fc1_input_size, num_policy_outputs)

    def forward(self, x):
        # 1. Initial layer
        x = F.relu(self.bn_in(self.conv_in(x)))
        
        # 2. Pass through all residual blocks
        for block in self.res_stack:
            x = block(x)
            
        # --- 3. Value Head Path ---
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = self.flatten(v) 
        v = F.relu(self.value_fc1(v))
        
        # --- MODIFICATION: Output raw logits ---
        value_logits = F.sigmoid(self.value_fc2(v))
        # -------------------------------------
        
        # --- 4. Policy Head Path ---
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = self.flatten(p)
        policy_logits = self.policy_fc1(p)
        
        # Return the LOGITS for the value head
        return value_logits, policy_logits