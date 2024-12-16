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
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # Deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out
    


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
    def __init__(self, embed_scale: float, embed_dim: int, input_dim: int):
        """
        Args:
            embed_scale: Scaling factor for initializing the random kernel.
            embed_dim: The total output embedding dimension.
            input_dim: Dimensionality of the input features (last axis of x).
        """
        super(FourierEmbs, self).__init__()
        self.embed_scale = embed_scale
        self.embed_dim = embed_dim
        self.input_dim = input_dim

        # Initialize the trainable kernel
        # self.kernel = nn.Parameter(
        #     torch.randn(input_dim, embed_dim // 2) * embed_scale
        # )

        # Initialize the trainable kernel
        self.kernel = nn.Parameter(torch.empty(input_dim, embed_dim // 2))
        self._initialize_kernel()

    def _initialize_kernel(self):
        """Custom initialization for the kernel."""
        nn.init.normal_(self.kernel, mean=0.0, std=self.embed_scale)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (..., input_dim).

        Returns:
            y: Fourier embeddings of shape (..., embed_dim).
        """
        # Compute the dot product of x with the kernel
        transformed = torch.matmul(x, self.kernel)

        # Apply sine and cosine transformations and concatenate
        y = torch.cat((torch.cos(transformed), torch.sin(transformed)), dim=-1)
        return y


# class FourierEmbs(nn.Module):
#     def __init__(self, embed_scale: float, embed_dim: int, input_dim: int, axes: list = None):
#         """
#         Args:
#             embed_scale: Scaling factor for initializing the random kernel.
#             embed_dim: The total output embedding dimension.
#             input_dim: Dimensionality of the input features (last axis of x).
#             axes: List of axes where Fourier embeddings will be applied. If None, apply to all axes.
#         """
#         super(FourierEmbs, self).__init__()
#         self.embed_scale = embed_scale
#         self.embed_dim = embed_dim
#         self.input_dim = input_dim
#         self.axes = axes if axes is not None else list(range(input_dim))  # Default: apply to all axes

#         # Initialize trainable kernels for selected axes
#         self.kernels = nn.ParameterDict({
#             str(axis): nn.Parameter(torch.randn(1, embed_dim // 2) * embed_scale)
#             for axis in self.axes
#         })

#     def forward(self, x):
#         """
#         Args:
#             x: Input tensor of shape (..., input_dim).

#         Returns:
#             y: Tensor with Fourier embeddings applied to the specified axes.
#         """
#         x_shape = x.shape  # Save original shape
#         x = x.view(-1, x.shape[-1])  # Flatten batch dimensions for ease of processing

#         y = []
#         for i in range(x.shape[-1]):
#             if i in self.axes:
#                 # Apply Fourier embedding to this axis
#                 kernel = self.kernels[str(i)]  # Get the kernel for this axis
#                 transformed = torch.matmul(x[:, i:i+1], kernel)  # Select only the i-th feature
#                 y.append(torch.cat([torch.cos(transformed), torch.sin(transformed)], dim=-1))
#             else:
#                 # Keep the axis unchanged
#                 y.append(x[:, i:i+1])

#         # Concatenate all processed axes
#         y = torch.cat(y, dim=-1)
#         return y.view(*x_shape[:-1], -1) 
    


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
    def __init__(self, arch_name="ModifiedDNN", num_layers=2, hidden_dim=20, out_dim=1, 
                 input_dim=3, activation="tanh", fourier_emb=None, period_emb=None):
        super(MDNN, self).__init__()
        
        self.arch_name = arch_name
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.activation = activation
        self.fourier_emb = fourier_emb
        self.period_emb = period_emb

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
            self.input_dim = self.fourier_emb['embed_dim']

        # Define the first layer for u and v components
        self.u_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.v_layer = nn.Linear(self.input_dim, self.hidden_dim)

        # Define hidden layers dynamically using nn.Linear
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.hidden_dim if i > 0 else self.input_dim, self.hidden_dim)
            for i in range(num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(self.hidden_dim, self.out_dim)

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

class WRF_MDNN(nn.Module):
    def __init__(self, arch_name="ModifiedDNN", num_layers=2, hidden_dim=20, out_dim=1, 
                 input_dim=3, activation="tanh", fourier_emb=None, reparam=None, period_emb=None, WRF_output = True):
        super(WRF_MDNN, self).__init__()
        
        self.arch_name = arch_name
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.activation = activation
        self.fourier_emb = fourier_emb
        self.reparam = reparam
        self.period_emb = period_emb

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
            self.input_dim = self.fourier_emb['embed_dim']

        # Define the first layer for u and v components
        self.u_layer = Dense(features=self.hidden_dim, input_dim=self.input_dim, reparam=self.reparam)
        self.v_layer = Dense(features=self.hidden_dim, input_dim=self.input_dim, reparam=self.reparam)

        # Define hidden layers dynamically
        self.hidden_layers = nn.ModuleList([
            Dense(features=self.hidden_dim, input_dim=(self.hidden_dim if i > 0 else self.input_dim), reparam=self.reparam)
            for i in range(num_layers)
        ])

        # Output layer
        if WRF_output:
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