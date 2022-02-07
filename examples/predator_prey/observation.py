"""Implement functions related to observations."""

import numpy as np

import state


def pursuer_observation(
    params, substep, history, state, actions
) -> tuple[str, state.State]:
    """The pursuer's relative distance to the target and its absolute velocity.

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
        "pursuer_observation"
    value: state.State
        The new value of the state variable.
    """

    pstate = state["pursuer_state"]
    tstate = state["target_state"]

    rel_pos = tstate.position - pstate.position
    value = np.array([rel_pos[0], rel_pos[1], *pstate.velocity])

    return "pursuer_observation", value
