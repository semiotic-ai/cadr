"""
Copyright 2021 Semiotic AI, Inc.
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

"""Implements functions related to networks."""

from typing import Callable, Iterable

import torch
from torch import nn


def mlp(*, activations: tuple[nn.Module], layer_sizes: tuple[int]) -> nn.Module:
    """Create an MLP.

    An MLP, or multi-layer perceptron, consists of several fully-connected layers joined
    by activation functions.

    Parameters
    ----------
    activations: tuple[nn.Module]
        A list of the activations functions that join the fully-connected layers. The
        length of the list should be ``len(layer_sizes) - 1``.
    layer_sizes: tuple[int]
        A list of the sizes of each fully-connected layer. The length of the list should
        be ``len(activations) + 1``

    Returns
    -------
    net: nn.Module:
        The constructed MLP network.

    Raises
    ------
    ValueError:
        If ``len(activations) + 1 != len(layer_sizes)``

    References
    ----------
    See `Wikipedia <https://en.wikipedia.org/wiki/Multilayer_perceptron>`_

    Examples
    --------
    >>> from cadr.network import mlp
    >>> import torch.nn as nn
    >>> activations = (nn.ReLU, nn.ReLU, nn.Tanh)
    >>> layer_sizes = (4, 256, 256, 2)
    >>> net = mlp(activations=activations, layer_sizes=layer_sizes)
    >>> isinstance(net[0], nn.Linear)
    True
    >>> len(net)
    6
    """
    if len(layer_sizes) != len(activations) + 1:
        raise ValueError(
            "The length of the layer sizes must be equal to the length of the "
            "activations + 1"
        )

    layers = []
    lin: int
    lout: int
    act: nn.Module
    for lin, lout, act in zip(layer_sizes[:-1], layer_sizes[1:], activations):
        layers.append(nn.Linear(lin, lout))
        layers.append(act())
    net = nn.Sequential(*layers)
    return net


def polyak(*, agent: nn.Module, rho: float, target: nn.Module) -> None:
    """Update the target's weights using the agent's weights as per:

    ..math:: \\theta_{\\text{target}} \\leftarrow
        \\rho \\theta_{\\text{target}} + (1 - \\rho) \\theta

    Parameters
    ----------
    agent: nn.Module
        The network from which to copy the weights.
    rho: float
        The interpolation factor in polyak averaging.
    target: ActorCritic
        The network to which to copy the weights.

    Examples
    --------
    >>> from cadr.network import mlp, polyak
    >>> import torch.nn as nn
    >>> activations = (nn.ReLU, nn.ReLU, nn.Tanh)
    >>> layer_sizes = (4, 256, 256, 2)
    >>> agent = mlp(activations=activations, layer_sizes=layer_sizes)
    >>> target = mlp(activations=activations, layer_sizes=layer_sizes)
    >>> polyak(agent=agent, rho=0.995, target=target)
    """
    with torch.no_grad():
        for p, pt in zip(agent.parameters(), target.parameters()):
            pt.data.mul_(rho)
            pt.data.add_((1 - rho) * p.data)


def backpropagation(
    *,
    batch: dict[str, torch.Tensor],
    loss: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    optim: torch.optim.Optimizer,
) -> float:
    """Update the weights of the network using backpropagation.

    Parameters
    ----------
    batch: dict[str, torch.Tensor]:
        A batch of samples on which to compute the loss.
    loss: Callable[[dict[str, torch.Tensor]], torch.Tensor]
        The loss function whose gradient will be used to backpropagate. Typically, you
        should use ``functools.partial`` to ensure that the loss is set up with all
        parameters other than the batch.
    optim: torch.optim.Optimizer
        A torch optimizer used to calculate the update step.

    Returns
    -------
    loss_val: float
        The computed loss value.

    Examples
    --------
    >>> import functools
    >>> from torch import nn
    >>> import cadr.network as cnet
    >>> net = cnet.mlp(activations=[nn.ReLU, nn.Identity], layer_sizes=[1, 32, 1])
    >>> optim = nn.Adam(net.parameters(), lr=1e-4)
    >>> def loss_func(*, batch, net):
            pred = net(batch["features"])
            loss = ((batch["labels"] - pred) ** 2).mean()
            return loss
    >>> batch = {
            "features": torch.tensor([[3.0], [4.0], [5.0]]),
            "labels": torch.tensor([[-3.0], [-4.0], [-5.0]]),
        }
    >>> my_loss = functools.partial(loss_func, net=net)
    >>> loss_val = cnet.backpropagation(batch=batch, loss=my_loss, optim=optim)
    """
    optim.zero_grad()
    # Ignore because mypy doesn't like kwargs on Callables
    loss_val = loss(batch=batch)  # type: ignore
    loss_val.backward()
    optim.step()

    return loss_val.detach().numpy().item()


def freeze(*, parameters: Iterable[torch.Tensor]) -> bool:
    """Freeze the given parameters.

    Parameters
    ----------
    parameters: Iterable[torch.Tensor]
        The parameters to freeze.

    Returns
    -------
    frozen: bool
        True. Used to track whether the network is currently frozen.
    """
    for param in parameters:
        param.requires_grad = False

    frozen = True
    return frozen


def unfreeze(*, parameters: Iterable[torch.Tensor]) -> bool:
    """Unfreeze the given parameters.

    Parameters
    ----------
    parameters: Iterable[torch.Tensor]
        The parameters to unfreeze.

    Returns
    -------
    frozen: bool
        False. Used to track whether the network is currently frozen.
    """
    for param in parameters:
        param.requires_grad = True

    frozen = False
    return frozen
