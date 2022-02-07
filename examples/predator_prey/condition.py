"""Implements conditions."""

import math

import numpy as np


def collision_condition(params, substep, history, state, actions):
    """Return 1 if the pursuer has collided with the target.

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
        "pursuer_done"
    value: int
        1 if the pursuer has collided with the target. Else 0.
    """
    pstate = state["pursuer_state"]
    tstate = state["target_state"]

    dist = np.linalg.norm(pstate.position - tstate.position)
    done = int(math.isclose(dist, 0.0, abs_tol=1e-2))

    return "pursuer_done", done
