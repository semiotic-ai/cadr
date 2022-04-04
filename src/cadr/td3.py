# Copyright 2022 Semiotic AI, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements the Twin-Delayed DDPG Algorithm."""

import copy
import functools
import itertools
from collections import deque
from typing import Callable, Iterable, Type

import numpy as np
import torch
import torch.nn as nn

import cadr.buffer as cbuff
import cadr.network as cnet


class Actor(nn.Module):
    """TD3's actor.

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
    >>> from cadr.td3 import Actor
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
    """TD3's critic.

    Parameters
    ----------
        q: nn.Module
            The value network of the agent.

    Attributes
    ----------
        q: nn.Module
            The value network of the agent.

    Examples
    --------
    >>> from cadr.td3 import Critic
    >>> from cadr.network import mlp
    >>> import torch.nn as nn
    >>> q = mlp(
            activations=[nn.ReLU, nn.ReLU, nn.Identity], layer_sizes=[6, 256, 256, 1]
        )
    >>> critic = Critic(q=q)
    """

    def __init__(self, *, q: nn.Module):
        super().__init__()
        self.q = q

    def forward(self, action: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass of the critic.

        Parameters
        ----------
        action: torch.Tensor
            The action of the agent.
        observation: torch.Tensor
            The observation of the agent.

        Returns
        -------
        value: torch.Tensor
            The value of the action.
        """
        _inp = torch.cat([observation, action], dim=-1)
        unsqueezed_value = self.q(_inp)
        value = torch.squeeze(unsqueezed_value, dim=-1)  # Squeeze for correct shape
        return value


class ActorCritic(nn.Module):
    """TD3's actor-critic.

    Parameters
    ----------
        action_scale: float
            The positive bound of the action space.
        pi: nn.Module
            The policy network of the agent.
        q: nn.Module
            The value network of the agent.
        qt: nn.Module
            The twin value network of the agent.

    Attributes
    ----------
        actor: Actor
            The policy network of the agent.
        critic: Critic
            The value network of the agent.
        twin_critic: Critic
            The twin value network of the agent.

    Examples
    --------
    >>> from cadr.td3 import ActorCritic
    >>> from cadr.network import mlp
    >>> import torch.nn as nn
    >>> pi = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[2, 256, 256, 4])
    >>> q = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[6, 256, 256, 1])
    >>> qt = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[6, 256, 256, 1])
    >>> agent = ActorCritic(action_scale=1.0, pi=pi, q=q, qt=qt)
    """

    def __init__(
        self, *, action_scale: float, pi: nn.Module, q: nn.Module, qt: nn.Module
    ):
        super().__init__()
        self.actor = Actor(action_scale=action_scale, pi=pi)
        self.critic = Critic(q=q)
        self.twin_critic = Critic(q=qt)

    def action(self, *, observation: torch.Tensor) -> np.ndarray:
        """Get the action for the agent to take.

        Parameters
        ----------
        observation: torch.Tensor
            The agent's observation.

        Returns
        -------
        action: np.ndarray
            The action the agent chooses to take as a numpy array (no gradient).
        """
        with torch.no_grad():
            action = self.actor(observation).numpy()
        return action


def _q_parameters(*, agent: ActorCritic) -> Iterable[torch.Tensor]:
    """The parameters of the critic.

    Parameters
    ----------
    agent: ActorCritic
        The agent whose critic's parameters to get.

    Returns
    -------
    params: Iterable[torch.Tensor]
        The parameters of the agent's critic.

    Examples
    --------
    >>> from cadr.td3 import ActorCritic, _q_parameters
    >>> from cadr.network import mlp
    >>> import torch.nn as nn
    >>> pi = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[2, 256, 256, 4])
    >>> q = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[6, 256, 256, 1])
    >>> qt = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[6, 256, 256, 1])
    >>> agent = ActorCritic(action_scale=1.0, pi=pi, q=q, qt=qt)
    >>> for param in _q_parameters(agent=agent):
    >>>     print(param)
    torch.Tensor(...)
    torch.Tensor(...)
    torch.Tensor(...)
    ...
    """
    params = itertools.chain(agent.critic.parameters(), agent.twin_critic.parameters())
    return params


def _pi_loss(*, agent: ActorCritic, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """The policy loss for TD3.

    Parameters
    ----------
    agent: ActorCritic
        The TD3 agent.
    batch: dict[str, torch.Tensor]
        A batch of samples from the buffer.

    Returns
    -------
    loss: torch.Tensor
        The policy loss.

    Examples
    --------
    >>> from cadr.td3 import ActorCritic, _pi_loss
    >>> from cadr.network import mlp
    >>> import torch.nn as nn
    >>> pi = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[2, 256, 256, 4])
    >>> q = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[6, 256, 256, 1])
    >>> qt = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[6, 256, 256, 1])
    >>> agent = ActorCritic(action_scale=1.0, pi=pi, q=q, qt=qt)
    >>> batch = {"observation": torch.Tensor([[0.0, 1.0], [1.0, 1.0]])}
    >>> _pi_loss(agent=agent, batch=batch)
    -0.004
    """
    obs = batch["observation"]
    act = agent.actor(observation=obs)
    qval = agent.critic(observation=obs, action=act)
    loss = -qval.mean()
    return loss


def _q_loss(
    *,
    agent: ActorCritic,
    batch: dict[str, torch.Tensor],
    gamma: float,
    noise_clip: float,
    target: ActorCritic,
    target_noise: float,
) -> torch.Tensor:
    """The value loss for TD3.

    Parameters
    ----------
    agent: ActorCritic
        The TD3 agent.
    batch: dict[str, torch.Tensor]
        A batch of samples from the buffer.
    gamma: float
        The discount factor.
    noise_clip: float
        The value to which to clip the noise.
    target: ActorCritic
        The target TD3 agent.
    target_noise: float
        The scale factor of the target's noise.

    Returns
    -------
    loss: torch.Tensor
        The value loss

    Examples
    --------
    >>> from cadr.td3 import td3_agent, _q_loss
    >>> from cadr.network import mlp
    >>> import torch.nn as nn
    >>> pi = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[2, 256, 256, 4])
    >>> q = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[6, 256, 256, 1])
    >>> qt = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[6, 256, 256, 1])
    >>> agent, target = td3_agent(action_scale=1.0, pi=pi, q=q, qt=qt)
    >>> batch = {
            "observation": torch.Tensor([[0.0, 1.0], [1.0, 1.0]]),
            "next_observation": torch.Tensor([[0.0, 1.0], [1.0, 1.0]]),
            "action": torch.Tensor([0.2], [0.3]),
            "reward": torch.Tensor([-4.2], [-12.0]),
            "done": torch.Tensor([0.0], [0.0]),
        }
    >>> _q_loss(
            agent=agent,
            batch=batch,
            gamma=0.995,
            noise_clip=0.5,
            target=target,
            target_noise=0.2
        )
    12002
    """
    obs = batch["observation"]
    nobs = batch["next_observation"]
    act = batch["action"]
    rew = batch["reward"]
    done = batch["done"]

    qval = agent.critic(observation=obs, action=act)
    twin_qval = agent.twin_critic(observation=obs, action=act)

    # Bellman Backup
    with torch.no_grad():
        tar_nact = target.actor(observation=nobs)

        # Add noise to target
        eps = torch.randn_like(tar_nact) * target_noise
        clipped_eps = torch.clamp(eps, -noise_clip, noise_clip)
        noisy_nact = tar_nact + clipped_eps
        clipped_nact = torch.clamp(
            noisy_nact, -target.actor.action_scale, target.actor.action_scale
        )

        # Target Q
        tar_qval = target.critic(observation=nobs, action=clipped_nact)
        tar_tqval = target.twin_critic(observation=nobs, action=clipped_nact)
        tar_q = torch.min(tar_qval, tar_tqval)

        # Backup
        backup = rew + gamma * (1 - done) * tar_q

    # MSE Loss against backup
    loss_qval = ((qval - backup) ** 2).mean()
    loss_tqval = ((twin_qval - backup) ** 2).mean()
    loss_q = loss_qval + loss_tqval

    return loss_q


def td3_agent(
    *, action_scale: float, pi: nn.Module, q: nn.Module, qt: nn.Module
) -> tuple[ActorCritic, ActorCritic]:
    """Return the Twin-Delayed DDPG agent.

    Parameters
    ----------
    action_scale: float
        The positive bound of the action space.
    pi: nn.Module
        The policy network of the agent.
    q: nn.Module
        The value network of the agent.
    qt: nn.Module
        The twin value network of the agent.

    Returns
    -------
    agent: ActorCritic
        The TD3 agent.
    target: ActorCritic
        The frozen TD3 agent.

    Examples
    --------
    >>> from cadr.td3 import td3_agent
    >>> from cadr.network import mlp
    >>> import torch.nn as nn
    >>> pi = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[2, 256, 256, 4])
    >>> q = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[6, 256, 256, 1])
    >>> qt = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[6, 256, 256, 1])
    >>> agent, target = td3_agent(action_scale=1.0, pi=pi, q=q, qt=qt)
    """
    agent = ActorCritic(action_scale=action_scale, pi=pi, q=q, qt=qt)
    target = copy.deepcopy(agent)

    # Freeze target
    for param in target.parameters():
        param.requires_grad = False

    return agent, target


def td3_optimizer(
    *,
    actor_lr: float,
    actor_optimizer: Type[torch.optim.Optimizer],
    agent: ActorCritic,
    critic_lr: float,
    critic_optimizer: Type[torch.optim.Optimizer],
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """Return the TD3 actor and critic optimizers.

    Parameters
    ----------
    actor_lr: float
        The actor's learning rate.
    actor_optimizer: torch.optim.Optimizer
        The actor's optimizer class.
    agent: ActorCritic
        The TD3 agent.
    critic_lr: float
        The critic's learning rate.
    critic_optimizer: torch.optim.Optimizer
        The critic's optimizer class.

    Returns
    -------
    actor_optim: Type[torch.optim.Optimizer]
        The actor's optimizer.
    critic_optim: Type[torch.optim.Optimizer]
        The critic's optimizer.

    Examples
    --------
    >>> from cadr.td3 import td3_optimizer
    >>> from cadr.network import mlp
    >>> import torch.nn as nn
    >>> import torch.optim as optim
    >>> pi = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[2, 256, 256, 4])
    >>> q = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[6, 256, 256, 1])
    >>> qt = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[6, 256, 256, 1])
    >>> agent = ActorCritic(action_scale=1.0, pi=pi, q=q, qt=qt)
    >>> actor_optim, critic_optim = td3_optimizer(
            actor_lr=1e-3,
            actor_optimizer=optim.Adam,
            agent=agent,
            critic_lr=1e-4,
            critic_optimizer=optim.Adam
        )
    """
    # Ignore because lr not in base class `Optimizer`, but it always in the concretion.
    actor_optim = actor_optimizer(agent.actor.parameters(), lr=actor_lr)  # type: ignore
    critic_optim = actor_optimizer(
        _q_parameters(agent=agent), lr=critic_lr
    )  # type: ignore

    return actor_optim, critic_optim


def td3_loss(
    *,
    agent: ActorCritic,
    gamma: float,
    noise_clip: float,
    target: ActorCritic,
    target_noise: float,
) -> tuple[
    Callable[[dict[str, torch.Tensor]], torch.Tensor],
    Callable[[dict[str, torch.Tensor]], torch.Tensor],
]:
    """Return the partial loss functions of TD3.

    To run these loss functions, do ``loss_fn(batch=batch)``.

    Parameters
    ----------
    agent: ActorCritic
        The TD3 agent.
    gamma: float
        The discount factor.
    noise_clip: float
        The value to which to clip the noise.
    target: ActorCritic
        The target TD3 agent.
    target_noise: float
        The scale factor of the target's noise.

    Returns
    -------
    pi_loss: Callable[[dict[str, torch.Tensor]], torch.Tensor]
        The policy loss function.
    q_loss: Callable[[dict[str, torch.Tensor]], torch.Tensor]
        The value loss function.

    Examples
    --------
    >>> from cadr.td3 import td3_agent, td3_loss
    >>> from cadr.network import mlp
    >>> import torch.nn as nn
    >>> pi = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[2, 256, 256, 4])
    >>> q = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[6, 256, 256, 1])
    >>> qt = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[6, 256, 256, 1])
    >>> agent, target = td3_agent(action_scale=1.0, pi=pi, q=q, qt=qt)
    >>> batch = {
            "observation": torch.Tensor([[0.0, 1.0], [1.0, 1.0]]),
            "next_observation": torch.Tensor([[0.0, 1.0], [1.0, 1.0]]),
            "action": torch.Tensor([0.2], [0.3]),
            "reward": torch.Tensor([-4.2], [-12.0]),
            "done": torch.Tensor([0.0], [0.0]),
        }
    >>> pi_loss, q_loss = td3_loss(
            agent=agent, gamma=0.995, noise_clip=0.5, target=target, target_noise=0.2
        )
    >>> pi_loss(batch=batch)
    -0.004
    >>> q_loss(batch=batch)
    12002
    """
    pi_loss = functools.partial(_pi_loss, agent=agent)
    q_loss = functools.partial(
        _q_loss,
        agent=agent,
        gamma=gamma,
        noise_clip=noise_clip,
        target=target,
        target_noise=target_noise,
    )
    return pi_loss, q_loss


def td3_update(
    *,
    agent: ActorCritic,
    actor_loss: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    actor_optim: torch.optim.Optimizer,
    actor_update_frequency: int,
    batch_size: int,
    buffer: deque[dict[str, np.ndarray]],
    critic_loss: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    critic_optim: torch.optim.Optimizer,
    device: str,
    gradient_steps: int,
    rho: float,
    target: ActorCritic,
) -> tuple[list[float], list[float]]:
    """Update the TD3 agent's weights.

    Parameters
    ----------
    agent: ActorCritic
        The TD3 agent.
    actor_loss: Callable[[dict[str, torch.Tensor]], torch.Tensor]
        The actor's loss function.
    actor_optim: torch.optim.Optimizer
        The actor's optimizer.
    actor_update_frequency: int
        How often to update the actor network and target agent weights. In TD3, the
        critic is updated more often than the actor.
    batch_size: int
        The number of samples to sample from the buffer.
    buffer: deque[dict[str, np.ndarray]]
        The buffer containing the samples.
    critic_loss: Callable[[dict[str, torch.Tensor]], torch.Tensor]
        The critic's loss function.
    critic_optim: torch.optim.Optimizer
        The critic's optimizer.
    device: str
        "cpu" or "cuda"
    gradient_steps: int
        The number of iterations for which to update the critic.
    rho: float
        The interpolation factor in polyak averaging. Target networks are updated
        according to:

        ..math:: \\theta_{\\text{target}} \\leftarrow
            \\rho \\theta_{\\text{target}} + (1 - \\rho) \\theta

    target: ActorCritic
        The target TD3 agent.

    Returns
    -------
    q_losses: list[float]
        A list of the critic's losses.
    pi_losses: list[float]
        A list of the actor's losses.

    Examples
    --------
    >>> from cadr.buffer import buffer
    >>> from cadr.td3 import td3_agent, td3_loss, td3_optimizer, td3_update
    >>> from cadr.network import mlp
    >>> import torch.nn as nn
    >>> import torch.optim as optim
    >>> buff = buffer(max_length=int(1e6))
    >>> sample = {"observation": np.array([0, 1])}
    >>> buff.append(sample)
    >>> buff.append(sample)
    >>> buff.append(sample)
    >>> pi = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[2, 256, 256, 4])
    >>> q = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[6, 256, 256, 1])
    >>> qt = mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[6, 256, 256, 1])
    >>> agent, target = td3_agent(action_scale=1.0, pi=pi, q=q, qt=qt)
    >>> pi_loss, q_loss = td3_loss(
            agent=agent, gamma=0.995, noise_clip=0.5, target=target, target_noise=0.2
        )
    >>> pi_optim, q_optim = td3_optimizer(
            actor_lr=1e-3,
            actor_optimizer=optim.Adam,
            agent=agent,
            critic_lr=1e-4,
            critic_optimizer=optim.Adam
        )
    >>> td3_update(
            agent=agent,
            actor_loss=pi_loss,
            actor_optim=pi_optim,
            actor_update_frequency=2,
            batch_size=1,
            buffer=buff,
            critic_loss=q_loss,
            critic_optim=q_optim,
            device="cpu",
            gradient_steps=32,
            rho=0.995,
            target=target
        )
    """
    q_losses = []
    pi_losses = []
    for i in range(gradient_steps):
        batch = cbuff.batch(batch_size=batch_size, buffer=buffer, device=device)

        # Critic loss
        q_loss = cnet.backpropagation(batch=batch, loss=critic_loss, optim=critic_optim)
        q_losses.append(q_loss)

        if i % actor_update_frequency == 0:
            # Freeze critic to prevent gradient during actor backprop
            _ = cnet.freeze(parameters=_q_parameters(agent=agent))

            # Actor loss
            pi_loss = cnet.backpropagation(
                batch=batch, loss=actor_loss, optim=actor_optim
            )
            pi_losses.append(pi_loss)

            # Unfreeze critic to enable gradient for critic updates.
            _ = cnet.unfreeze(parameters=_q_parameters(agent=agent))

            # # Update target weights
            cnet.polyak(agent=agent, target=target, rho=rho)

    return q_losses, pi_losses
