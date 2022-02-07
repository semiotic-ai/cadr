"""Implements reward functions."""

import numpy as np


def pursuer_distance_reward(params, substep, history, state, actions):
    """A reward based on the distance between a pursuer and a target.

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
        "pursuer_reward"
    value: float
        The negative of the distance between the predator and prey. If the distance is
        0, this value is a positive reward.
    """

    pstate = state["pursuer_state"]
    tstate = state["target_state"]

    neg_dist = -10.0 * np.linalg.norm(pstate.position - tstate.position)

    if neg_dist == 0:
        neg_dist = 10.0

    return "pursuer_reward", neg_dist
