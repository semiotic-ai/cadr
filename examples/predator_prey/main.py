"""Implements an RL training loop using cadR and cadCAD for predator-prey."""

from cadCAD.configuration.utils import config_sim
from cadCAD.configuration import Experiment
from cadCAD import configs
from cadCAD.engine import ExecutionContext, Executor
import numpy as np
import torch
from torch import nn
from torch import optim

import cadr.buffer as cbuff
import cadr.network as cnet
import cadr.td3 as ctd3

import action
import agent
import condition
import dynamics
import observation
import playback
import reward
import state


def main():
    # Seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # RL setup
    num_episodes = 1000
    buffer = cbuff.buffer(max_length=int(1e6))
    pi = cnet.mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[4, 256, 256, 2])
    q = cnet.mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[6, 256, 256, 1])
    qt = cnet.mlp(activations=[nn.ReLU, nn.ReLU, nn.Tanh], layer_sizes=[6, 256, 256, 1])
    pursuer, target_pursuer = ctd3.td3_agent(action_scale=1.0, pi=pi, q=q, qt=qt)
    pi_optim, q_optim = ctd3.td3_optimizer(
        actor_lr=1e-4,
        actor_optimizer=optim.Adam,
        agent=pursuer,
        critic_lr=1e-4,
        critic_optimizer=optim.Adam,
    )
    pi_loss, q_loss = ctd3.td3_loss(
        agent=pursuer,
        gamma=0.99,
        noise_clip=0.5,
        target=target_pursuer,
        target_noise=0.2,
    )

    # Environment setup (using cadCAD)
    del configs[:]
    sim_config = config_sim({"T": range(100), "N": 1})
    partial_state_update_blocks = [
        {
            "policies": {
                "pursuer_agent": agent.pursuer_agent,
            },
            "variables": {
                "pursuer_state": dynamics.update_pursuer_state,
                "target_state": dynamics.static_target,
                "pursuer_reward": reward.pursuer_distance_reward,
                "pursuer_done": condition.collision_condition,
                "pursuer_observation": observation.pursuer_observation,
                "pursuer_action": action.pursuer_action,
            },
        }
    ]

    # Run training loop
    for ep in range(num_episodes):
        if ep % 100 == 0:
            print(ep)

        # Create initial state. Must be in loop for random initialisation
        p0 = state.State(position=np.random.uniform(-1, 1, 2), velocity=np.zeros((2,)))
        t0 = state.State(position=np.zeros((2,)), velocity=np.zeros((2,)))
        _, obs = observation.pursuer_observation(
            params=None,
            substep=None,
            history=None,
            state={"pursuer_state": p0, "target_state": t0},
            actions=None,
        )
        initial_state = {
            "pursuer_state": p0,
            "target_state": t0,
            "pursuer_agent": pursuer,
            "pursuer_reward": 0.0,
            "pursuer_done": 0,
            "delta_t": 0.1,
            "pursuer_observation": obs,
            "pursuer_action": np.zeros((2,)),
        }

        # Set up cadcad run
        experiment = Experiment()
        experiment.append_configs(
            sim_configs=sim_config,
            initial_state=initial_state,
            partial_state_update_blocks=partial_state_update_blocks,
        )
        exec_context = ExecutionContext()
        run = Executor(exec_context=exec_context, configs=experiment.configs)

        # TRAINING LOOP:

        # COLLECT ROLLOUTS
        (system_events, tensor_field, sessions) = run.execute()

        # Push to buffer
        for se, nse in zip(system_events[:-1], system_events[1:]):
            sample = {
                "observation": se["pursuer_observation"],
                "next_observation": nse["pursuer_observation"],
                "action": se["pursuer_action"],
                "reward": se["pursuer_reward"],
                "done": se["pursuer_done"],
            }
            buffer.append(sample)

        # UPDATE ALGORITHM
        ctd3.td3_update(
            agent=pursuer,
            actor_loss=pi_loss,
            actor_optim=pi_optim,
            actor_update_frequency=2,
            batch_size=32,
            buffer=buffer,
            critic_loss=q_loss,
            critic_optim=q_optim,
            device="cpu",
            gradient_steps=32,
            rho=0.995,
            target=target_pursuer,
        )

    playback.playback(system_events=system_events)


if __name__ == "__main__":
    main()
