import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional, Dict,Callable,Tuple

# Swish Function 
class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    

# Helper function for activation functions
def _get_activation(activation):
    if activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "swish":
        return lambda x: x * torch.sigmoid(x)
    else:
        raise ValueError(f"Unknown activation function: {activation}")


class PeriodEmbs(nn.Module):
    def __init__(self, period: Tuple[float], axis: Tuple[int]):
        """
        Args:
            period: Periods for different axes.
            axis: Axes where the period embeddings are to be applied.
        """
        super(PeriodEmbs, self).__init__()
        self.axis = axis

        # Store period parameters as constants (non-trainable)
        for idx, p in enumerate(period):
            self.register_buffer(f"period_{idx}", torch.tensor(p, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the period embeddings to the specified axes.

        Args:
            x: Input tensor of shape (N, F).

        Returns:
            y: Tensor with period embeddings applied, shape (N, F + 2 * len(axis)).
        """
        y = []  # To store processed features
        
        for i in range(x.size(1)):  # Loop over features (columns)
            xi = x[:, i]  # Extract the i-th feature (column)
            if i in self.axis:  # Apply embedding to specified axes
                idx = self.axis.index(i)
                period = getattr(self, f"period_{idx}")  # Retrieve period
                y.append(torch.cos(period * xi).unsqueeze(-1))  # Add cos embedding
                y.append(torch.sin(period * xi).unsqueeze(-1))  # Add sin embedding
            else:
                y.append(xi.unsqueeze(-1))  # Keep original feature
        
        return torch.cat(y, dim=-1)  # Concatenate along the feature axis
    
class FourierEmbs(nn.Module):
    def __init__(self, embed_scale: float, embed_dim: int, input_dim: int, exclude_last_n: int = 3):
        super(FourierEmbs, self).__init__()
        self.embed_scale = embed_scale
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.exclude_last_n = exclude_last_n

        # Calculate the number of axes to apply Fourier embeddings to
        # If exclude_last_n is 0, we apply Fourier embeddings to all axes.
        self.embedded_axes = self.input_dim - self.exclude_last_n
        
        # Initialize the trainable kernel for the Fourier embedding
        self.kernel = nn.Parameter(torch.empty(self.embedded_axes, embed_dim // 2))
        self._initialize_kernel()

    def _initialize_kernel(self):
        """Custom initialization for the kernel."""
        nn.init.normal_(self.kernel, mean=0.0, std=self.embed_scale)

    def forward(self, x):
        """Compute Fourier embeddings, excluding the last `exclude_last_n` elements along the given axis."""
        
        # If exclude_last_n is 0, apply Fourier embeddings to the entire tensor
        if self.exclude_last_n == 0:
            xi = x  # No exclusion, apply Fourier transformation to all columns
        else:
            # Select all columns except the last `exclude_last_n` columns
            xi = x[..., :-self.exclude_last_n]  # This selects all but the last `exclude_last_n` columns

        # Compute the dot product of xi with the kernel
        transformed = torch.matmul(xi, self.kernel)

        # Apply sine and cosine transformations
        fourier_emb = torch.cat((torch.cos(transformed), torch.sin(transformed)), dim=-1)

        # Reconstruct the full output tensor by concatenating Fourier embeddings with the remaining columns
        if self.exclude_last_n > 0:
            remaining = x[..., -self.exclude_last_n:]  # This selects the last `exclude_last_n` columns
            y = torch.cat((fourier_emb, remaining), dim=-1)
        else:
            y = fourier_emb  # No exclusion, just return the Fourier embeddings
        return y

def _weight_fact(kernel_init, mean=1.0, stddev=0.1, shape=None):
    # Function to initialize weights with factorization
    w = kernel_init(torch.empty(shape))  # Standard initialization
    g = mean + torch.normal(mean=0, std=stddev, size=(shape[-1],))  # Sample g
    g = torch.exp(g)  # Exponential transformation for g
    v = w / g  # Weight factorization
    return g, v

class Dense(nn.Module):
    def __init__(self, features: int, input_dim: int, kernel_init: callable = nn.init.xavier_uniform_, 
                 bias_init: callable = nn.init.zeros_, reparam: Optional[Dict] = None):
        super(Dense, self).__init__()
        self.features = features
        self.input_dim = input_dim
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.reparam = reparam
        
        # Initialize parameters
        if self.reparam is None:
            self.kernel = nn.Parameter(self.kernel_init(torch.empty((self.input_dim, self.features))))  # Initialize kernel with Xavier
        elif self.reparam["type"] == "weight_fact":
            g, v = _weight_fact(
                self.kernel_init,
                mean=self.reparam["mean"],
                stddev=self.reparam["stddev"],
                shape=(self.input_dim, self.features)
            )
            self.g = nn.Parameter(g)
            self.v = nn.Parameter(v)
        
        # Initialize bias
        self.bias = nn.Parameter(self.bias_init(torch.empty((self.features,))))

    def forward(self, x):
        # Dynamically calculate the kernel when using weight factorization
        if self.reparam and self.reparam["type"] == "weight_fact":
            kernel = self.g * self.v
        else:
            kernel = self.kernel
            
        return x @ kernel + self.bias
    
# Deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers, activation = "tanh"):
        super(DNN, self).__init__()

        # Number of layers
        self.depth = len(layers) - 1
        
        self.activation = activation
        self.activation_fn = _get_activation(self.activation)  # Ensure Swish is defined
        
        # The following loop organized the layers of the NN         
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation_fn))
        layer_list.append(
            ('output_layer', torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # Deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out
    

 # Deep neural network
class WRFNN(torch.nn.Module):
    def __init__(self, layers, activation = "tanh",reparam: Optional[Dict] = None):
        super(WRFNN, self).__init__()

        self.activation = activation
        self.activation_fn = _get_activation(self.activation)  # Ensure Swish is defined

        # Initialize layers
        layer_list = []
        for i in range(len(layers) - 1):
            input_dim, output_dim = layers[i], layers[i + 1]
            layer_list.append((f'layer_{i}', Dense(features=output_dim, input_dim=input_dim, reparam=reparam)))
            if i < len(layers) - 2:  # Skip activation on the final layer
                layer_list.append((f'activation_{i}', self.activation_fn))

        # Deploy layers in an OrderedDict
        self.layers = nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        return self.layers(x) 
    
    
class MDNN(nn.Module):
    def __init__(self, arch_name="MDNN", num_layers=2, hidden_dim=20, out_dim=1, 
                 input_dim=3, activation="tanh", fourier_emb=None, period_emb=None, 
                 reparam=None, WRF_output=False):
        super(MDNN, self).__init__()
        
        self.arch_name = arch_name
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.activation = activation
        self.fourier_emb = fourier_emb
        self.period_emb = period_emb
        self.reparam = reparam
        self.WRF_output = WRF_output

        # Setup activation function
        self.activation_fn = _get_activation(self.activation)

        # Add Period embeddings layer if specified
        if self.period_emb:
            self.period_layer = PeriodEmbs(**self.period_emb)
            # Adjust input_dim based on the new dimensions from PeriodEmbs
            self.input_dim += len(self.period_emb['axis'])

        # Add Fourier embeddings layer if specified
        if self.fourier_emb:
            self.fourier_emb["input_dim"] = self.input_dim
            self.fourier_layer = FourierEmbs(**self.fourier_emb)
            self.input_dim = self.fourier_emb['embed_dim'] + self.fourier_emb["exclude_last_n"]
            self.hidden_dim += self.fourier_emb["exclude_last_n"]

        # Define the first layer for u and v components
        if self.reparam:
            self.u_layer = Dense(features=self.hidden_dim, input_dim=self.input_dim, reparam=self.reparam)
            self.v_layer = Dense(features=self.hidden_dim, input_dim=self.input_dim, reparam=self.reparam)
        else:
            self.u_layer = nn.Linear(self.input_dim, self.hidden_dim)
            self.v_layer = nn.Linear(self.input_dim, self.hidden_dim)

        # Define hidden layers dynamically
        self.hidden_layers = nn.ModuleList([
            Dense(features=self.hidden_dim, input_dim=(self.hidden_dim if i > 0 else self.input_dim), reparam=self.reparam)
            if self.reparam else
            nn.Linear(self.hidden_dim if i > 0 else self.input_dim, self.hidden_dim)
            for i in range(num_layers)
        ])

        # Output layer
        if self.WRF_output:
            self.output_layer = Dense(features=self.out_dim, input_dim=self.hidden_dim, reparam=self.reparam)
        else:
            self.output_layer = nn.Linear(in_features=self.hidden_dim, out_features=self.out_dim)

    def forward(self, x):
        # Apply Period embeddings if provided
        if self.period_emb:
            x = self.period_layer(x)

        # Apply Fourier embeddings if provided
        if self.fourier_emb:
            x = self.fourier_layer(x)

        # Initial u and v transformations with activation
        u = self.activation_fn(self.u_layer(x))
        v = self.activation_fn(self.v_layer(x))

        # Apply hidden layers with the mixture operation
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation_fn(x)
            
            # Mixture of u and v with x for interaction
            x = x * u + (1 - x) * v  

        # Final output layer
        x = self.output_layer(x)
        return x
    

# class MDNN(nn.Module):
#     def __init__(self, arch_name="ModifiedDNN", num_layers=2, hidden_dim=20, out_dim=1, 
#                  input_dim=3, activation="tanh", fourier_emb=None, period_emb=None):
#         super(MDNN, self).__init__()
        
#         self.arch_name = arch_name
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.out_dim = out_dim
#         self.input_dim = input_dim
#         self.activation = activation
#         self.fourier_emb = fourier_emb
#         self.period_emb = period_emb

#         # Setup activation function
#         self.activation_fn = _get_activation(self.activation)

#         # Add Period embeddings layer if specified
#         if self.period_emb:
#             self.period_layer = PeriodEmbs(**self.period_emb)
#             # Adjust input_dim based on the new dimensions from PeriodEmbs
#             self.input_dim += len(self.period_emb['axis'])

#         # Add Fourier embeddings layer if specified
#         if self.fourier_emb:
#             self.fourier_emb["input_dim"] = self.input_dim
#             self.fourier_layer = FourierEmbs(**self.fourier_emb)
#             self.input_dim = self.fourier_emb['embed_dim'] + self.fourier_emb["exclude_last_n"]
#             self.hidden_dim += self.fourier_emb["exclude_last_n"]

#         # Define the first layer for u and v components
#         self.u_layer = nn.Linear(self.input_dim, self.hidden_dim)
#         self.v_layer = nn.Linear(self.input_dim, self.hidden_dim)

#         # Define hidden layers dynamically using nn.Linear
#         self.hidden_layers = nn.ModuleList([
#             nn.Linear(self.hidden_dim if i > 0 else self.input_dim, self.hidden_dim)
#             for i in range(num_layers)
#         ])

#         # Output layer
#         self.output_layer = nn.Linear(self.hidden_dim, self.out_dim)

#     def forward(self, x):
#         # Apply Period embeddings if provided
#         if self.period_emb:
#             x = self.period_layer(x)

#         # Apply Fourier embeddings if provided
#         if self.fourier_emb:
#             x = self.fourier_layer(x)

#         # Initial u and v transformations with activation
#         u = self.activation_fn(self.u_layer(x))
#         v = self.activation_fn(self.v_layer(x))

#         # Apply hidden layers with the mixture operation
#         for layer in self.hidden_layers:
#             x = layer(x)
#             x = self.activation_fn(x)
            
#             # Mixture of u and v with x for interaction
#             x = x * u + (1 - x) * v  

#         # Final output layer
#         x = self.output_layer(x)
#         return x

# class WRF_MDNN(nn.Module):
#     def __init__(self, arch_name="ModifiedDNN", num_layers=2, hidden_dim=20, out_dim=1, 
#                  input_dim=3, activation="tanh", fourier_emb=None, reparam=None, period_emb=None, WRF_output = True):
#         super(WRF_MDNN, self).__init__()
        
#         self.arch_name = arch_name
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.out_dim = out_dim
#         self.input_dim = input_dim
#         self.activation = activation
#         self.fourier_emb = fourier_emb
#         self.reparam = reparam
#         self.period_emb = period_emb

#         # Setup activation function
#         self.activation_fn = _get_activation(self.activation)

#         # Add Period embeddings layer if specified
#         if self.period_emb:
#             self.period_layer = PeriodEmbs(**self.period_emb)
#             # Adjust input_dim based on the new dimensions from PeriodEmbs
#             self.input_dim += len(self.period_emb['axis'])

#         # Add Fourier embeddings layer if specified
#         if self.fourier_emb:
#             self.fourier_emb["input_dim"] = self.input_dim
#             self.fourier_layer = FourierEmbs(**self.fourier_emb)
#             self.input_dim = self.fourier_emb['embed_dim'] + self.fourier_emb["exclude_last_n"]
#             self.hidden_dim += self.fourier_emb["exclude_last_n"]
#         # Define the first layer for u and v components
#         self.u_layer = Dense(features=self.hidden_dim, input_dim=self.input_dim, reparam=self.reparam)
#         self.v_layer = Dense(features=self.hidden_dim, input_dim=self.input_dim, reparam=self.reparam)

#         # Define hidden layers dynamically
#         self.hidden_layers = nn.ModuleList([
#             Dense(features=self.hidden_dim, input_dim=(self.hidden_dim if i > 0 else self.input_dim), reparam=self.reparam)
#             for i in range(num_layers)
#         ])

#         # Output layer
#         if WRF_output:
#             self.output_layer = Dense(features=self.out_dim, input_dim=self.hidden_dim, reparam=self.reparam)
#         else:
#             self.output_layer = nn.Linear(in_features=self.hidden_dim, out_features=self.out_dim)

#     def forward(self, x):
#         # Apply Period embeddings if provided
#         if self.period_emb:
#             x = self.period_layer(x)

#         # Apply Fourier embeddings if provided
#         if self.fourier_emb:
#             x = self.fourier_layer(x)

#         # Initial u and v transformations with activation
#         u = self.activation_fn(self.u_layer(x))
#         v = self.activation_fn(self.v_layer(x))

#         # Apply hidden layers with the mixture operation
#         for layer in self.hidden_layers:
#             x = layer(x)
#             x = self.activation_fn(x)
            
#             # Mixture of u and v with x for interaction
#             x = x * u + (1 - x) * v  

#         # Final output layer
#         x = self.output_layer(x)
#         return x