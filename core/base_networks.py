# dsac/core/base_networks.py
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod


def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal initialization for a layer's weights and constant for bias.
    """
    if hasattr(layer, 'weight'):
        nn.init.orthogonal_(layer.weight, gain=std)
    if hasattr(layer, 'bias') and layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron.
    """

    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256], activation=nn.ReLU, output_activation=None):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(init_layer(nn.Linear(current_dim, hidden_dim)))
            layers.append(activation())
            current_dim = hidden_dim

        layers.append(init_layer(nn.Linear(current_dim, output_dim)))
        if output_activation:
            layers.append(output_activation())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class BaseEncoder(nn.Module, ABC):

    def __init__(self):
        super().__init__()
        self._output_dim = 0  # Subclasses must set this appropriately

    @property
    @abstractmethod
    def output_dim(self):
        """Returns the dimension of the features output by this encoder."""
        return self._output_dim

    @abstractmethod
    def forward(self, obs):
        """Processes observations and returns encoded features."""
        pass


class IdentityEncoder(BaseEncoder):

    def __init__(self, obs_dim):
        super().__init__()
        if isinstance(obs_dim, tuple):  # If shape tuple like (dim,)
            self._output_dim = obs_dim[0]
        else:  # If integer
            self._output_dim = obs_dim
        self.net = nn.Identity()

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, obs):
        return self.net(obs)


class SimpleMLPEncoder(BaseEncoder):

    def __init__(self, obs_dim, hidden_dims_encoder=[64, 64], activation=nn.ReLU):
        super().__init__()
        self._input_dim = obs_dim
        if not hidden_dims_encoder:
            self.mlp_encoder = nn.Identity()
            self._output_dim = self._input_dim
        else:
            self.mlp_encoder = MLP(input_dim=self._input_dim,
                                   output_dim=hidden_dims_encoder[-1],
                                   hidden_dims=hidden_dims_encoder[:-1],
                                   activation=activation,
                                   output_activation=activation())  # Activation after last hidden
            self._output_dim = hidden_dims_encoder[-1]

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, obs):
        return self.mlp_encoder(obs)
