"""Implements functions related to networks."""

import torch.nn as nn


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
