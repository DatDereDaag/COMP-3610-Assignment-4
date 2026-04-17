import torch
import torch.nn as nn

class RegressionNeuralNetwork(nn.Module):
    def __init__(self,  input_size, hidden_sizes=[128, 64],  dropout_rate=0.3 ):
        super().__init__()

        layers = []
        prev_size = input_size 

        #Adding layers to network dynamically
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU()) 
            layers.append(nn.Dropout(dropout_rate)) 
            prev_size = hidden_size 
        
        # Output layer (single neuron for binary classification) 
        layers.append(nn.Linear(prev_size, 1)) 
        
        #Set layers which we dynamically built
        self.network = nn.Sequential(*layers) 

    def forward(self, x):
        return self.network(x).squeeze()