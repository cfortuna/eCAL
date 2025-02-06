import torch.nn as nn
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
    def __init__(self, input_channels=1, hidden_channels=10, output_size=2, sequence_length=10, num_layers=3):
        super(SimpleCNN, self).__init__()
        
        self.layers = nn.ModuleList()
        current_channels = input_channels
        
        # Calculate how many pooling layers we can safely use
        max_possible_layers = int(np.log2(sequence_length))
        num_layers = min(num_layers, max_possible_layers)
        
        # Create convolutional layers
        for i in range(num_layers):
            # Double channels after first layer
            out_channels = hidden_channels * (2 if i > 0 else 1)
            
            # Create conv block
            conv_block = nn.ModuleList([
                nn.Conv1d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                # Only add pooling if the sequence length is sufficient
                nn.MaxPool1d(kernel_size=2) if sequence_length // (2**(i+1)) >= 1 else nn.Identity()
            ])
            
            self.layers.append(conv_block)
            current_channels = out_channels
            
        # Calculate the size after all pooling layers
        final_sequence_length = sequence_length // (2**num_layers)
        self.flatten_size = current_channels * final_sequence_length
        
        # Output layer
        self.output = nn.Linear(self.flatten_size, output_size)
    
    def forward(self, x):
        # Pass through all convolutional blocks
        for conv_block in self.layers:
            for layer in conv_block:
                x = layer(x)
        
        x = x.view(-1, self.flatten_size)
        return self.output(x)