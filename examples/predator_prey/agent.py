"""Implements the RL agent's adapter for cadCAD."""

import numpy as np
import torch

import action


def pursuer_agent(params, substep, history, state) -> dict[str, action.Action]:
    """The RL agent representing the pursuer.

    Parameters
    ----------
    params: dict
        System parameters that can be swept.
    substep: int
        The sub-timestep in which the updated states of agents don't
        affect other agents.
    history: list[list[dict]]
        The history of states.
    state: dict
        The current state of the system.

    Returns
    -------
    value: dict[str, action.Action]
        The new value of the state variable.
    """
    obs = state["pursuer_observation"]
    agent = state["pursuer_agent"]

    with torch.no_grad():
        act = agent.action(observation=torch.tensor(obs, dtype=torch.float32))

    paction = action.Action(acceleration=act)
    taction = action.Action(acceleration=np.zeros_like(act))

    return {"pursuer_action": paction, "target_action": taction}
