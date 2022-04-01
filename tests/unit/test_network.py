"""
Copyright 2022 Semiotic AI, Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

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
