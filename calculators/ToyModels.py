import torch.nn as nn
import torch
import numpy as np
class SimpleMLP(nn.Module):
    def __init__(self, input_size=10, hidden_size=10, output_size=2, num_layers=3):
        super(SimpleMLP, self).__init__()
        
        # Create a ModuleList to store variable number of layers
        self.layers = nn.ModuleList()
        
        # First layer (input to hidden)
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        
        # Output layer
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Pass through all layers sequentially
        for layer in self.layers:
            x = layer(x)
        return self.output(x)
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=10, output_size=2, num_layers=3):
        super(SimpleCNN, self).__init__()
        
        self.layers = nn.ModuleList()
        current_channels = input_channels
        
        # Create convolutional layers
        for i in range(num_layers):
            # More controlled channel growth
            out_channels = hidden_channels * (2 if i > 0 else 1)
            
            # Create conv block with standard pooling
            conv_block = nn.Sequential(
                nn.Conv1d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
            
            self.layers.append(conv_block)
            current_channels = out_channels
        
        # Global average pooling
        self.global_pool = nn.AvgPool1d(kernel_size=2)
        
        # Output layer
        self.output = nn.Linear(current_channels, output_size)
    
    def forward(self, x):
        # Pass through all convolutional blocks
        for layer in self.layers:
            x = layer(x)
            
        
        # Ensure we have at least one feature
        if x.size(-1) > 1:
            x = self.global_pool(x)
        
        # Global average pooling
        x = torch.mean(x, dim=-1)
        
        return self.output(x)