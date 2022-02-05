"""Implement classes related to agents."""

import numpy
import torch
import torch.nn as nn


class Actor(nn.Module):
    """A policy network.

    Parameters
    ----------
        action_scale (float): The positive bound of the action space.
        pi (nn.Module): The policy network of the agent.

    Attributes
    ----------
        action_scale (float): The positive bound of the action space.
        pi (nn.Module): The policy network of the agent.

    Examples
    --------
    >>> from cadr.core import Actor
    >>> from cadr.network import mlp
    >>> import torch.nn as nn
    >>> pi = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[2, 256, 256, 4])
    >>> actor = Actor(action_scale=1.0, pi=pi)
    """

    def __init__(self, *, action_scale: float, pi: nn.Module):
        super().__init__()
        self.action_scale = action_scale
        self.pi = pi

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass of the actor.

        Parameters
        ----------
        observation: torch.Tensor
            The observation of the agent.

        Returns
        -------
        action: torch.Tensor
            The action of the agent.
        """
        action = self.action_scale * self.pi(observation)
        return action


class Critic(nn.Module):
    """A critic network.

    Parameters
    ----------
        q (nn.Module): The value network of the agent.

    Attributes
    ----------
        q (nn.Module): The value network of the agent.

    Examples
    --------
    >>> from cadr.core import Critic
    >>> from cadr.network import mlp
    >>> import torch.nn as nn
    >>> q = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[2, 256, 256, 4])
    >>> critic = Critic(q=q)
    """

    def __init__(self, *, q: nn.Module):
        super().__init__()
        self.q = q

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass of the critic.

        Parameters
        ----------
        observation: torch.Tensor
            The observation of the agent.

        Returns
        -------
        value: torch.Tensor
            The value of the action.
        """
        value = self.q(observation)
        return value


class ActorCritic(nn.Module):
    """An actor-critic.

    Parameters
    ----------
        action_scale (float): The positive bound of the action space.
        pi (nn.Module): The policy network of the agent.
        q (nn.Module): The value network of the agent.

    Attributes
    ----------
        actor (nn.Module): The policy network of the agent.
        critic (nn.Module): The value network of the agent.

    Examples
    --------
    >>> from cadr.core import ActorCritic
    >>> from cadr.network import mlp
    >>> import torch.nn as nn
    >>> pi = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[2, 256, 256, 4])
    >>> q = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[2, 256, 256, 4])
    >>> agent = ActorCritic(action_scale=1.0, pi=pi, q=q)
    """

    def __init__(self, *, action_scale: float, pi: nn.Module, q: nn.Module):
        super().__init__()
        self.actor = Actor(action_scale=action_scale, pi=pi)
        self.critic = Critic(q=q)

    def action(self, *, observation: torch.Tensor) -> np.ndarray:
        """Get the action for the agent to take.

        Parameters
        ----------
        observation: torch.Tensor
            The agent's observation.

        Returns
        -------
        action: np.ndarray
            The action the agent chooses to take.
        """
        with torch.no_grad():
            action = self.actor(observation).numpy()
        return action
