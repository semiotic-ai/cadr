"""Implements actions."""

from typing import NamedTuple

import numpy as np


class Action(NamedTuple):
    """The action of agents.

    Attributes
    ----------
    acceleration: np.ndarray
        The acceleration of the agent.
    """

    acceleration: np.ndarray


def pursuer_action(params, substep, history, state, actions) -> tuple[str, Action]:
    """Save the action as an array so that it can be added to our buffer.

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
    actions: dict
        Actions of the agents.

    Returns
    -------
    name: str
        "pursuer_action"
    value: np.ndarray
        The pursuer's action as an array.
    """
    return "pursuer_action", np.array(*actions["pursuer_action"])
