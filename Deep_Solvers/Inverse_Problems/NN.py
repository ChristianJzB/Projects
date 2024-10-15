import torch
from collections import OrderedDict

# Swish Function 
class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers, activation = "tanh"):
        super(DNN, self).__init__()

        # Number of layers
        self.depth = len(layers) - 1
        
        # Activation Function
        if activation == "tanh":
            self.activation = torch.nn.Tanh
        elif activation == "sigmoid":
            self.activation = torch.nn.Sigmoid
        elif activation == "relu":
            self.activation = torch.nn.ReLU
        if activation == "swish":
            self.activation = Swish

        
        # The following loop organized the layers of the NN         
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # Deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out