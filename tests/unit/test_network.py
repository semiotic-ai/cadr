"""Test the cadr.network module"""

import pytest
import torch.nn as nn

import cadr.network


def test_mlp():
    # Config
    layer_sizes = [4, 256, 256, 2]
    activations = [nn.ReLU, nn.ReLU, nn.Tanh]

    # Create MLP
    net = cadr.network.mlp(activations=activations, layer_sizes=layer_sizes)

    # Test
    assert isinstance(net[0], nn.Linear)
    assert isinstance(net[-1], nn.Tanh)
    assert len(net) == 6

    # Bad Config
    layer_sizes = [4, 256, 2]
    activations = [nn.ReLU, nn.ReLU, nn.Tanh]

    # Raise Error
    with pytest.raises(ValueError):
        cadr.network.mlp(activations=activations, layer_sizes=layer_sizes)
