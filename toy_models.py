import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size=10, hidden_size=8, output_size=2):
        super(SimpleMLP, self).__init__()
        
        # First hidden layer
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        
        # Second hidden layer
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        # Output layer
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.relu3(self.layer3(x))
        return self.output(x)

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=16, output_size=2, sequence_length=128):
        super(SimpleCNN, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels*2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Third convolutional layer
        self.conv3 = nn.Conv1d(hidden_channels*2, hidden_channels*2, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate the size after 3 pooling layers (each dividing length by 2)
        self.flatten_size = hidden_channels * 2 * (sequence_length // (2**3))
        
        # Output layer
        self.output = nn.Linear(self.flatten_size, output_size)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, self.flatten_size)
        return self.output(x)
    

    