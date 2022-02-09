"""Implements state update functions (dynamics)."""

import copy

import numpy as np

import state as _state
import helper


def update_pursuer_state(
    params, substep, history, state, actions
) -> tuple[str, _state.State]:
    """Update the pursuer's state.

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
        "pursuer_state"
    value: _state.State
        The new value of the state variable.
    """
    pstate = state["pursuer_state"]
    paction = actions["pursuer_action"]
    delta_t = [params["delta_t"]] * len(pstate.position)

    updated_state = tuple(
        map(
            helper.kinematics,
            pstate.position,
            pstate.velocity,
            paction.acceleration,
            delta_t,
        )
    )
    updated_s = np.array(tuple(map(lambda x: x[0], updated_state)))
    updated_v = np.array(tuple(map(lambda x: x[1], updated_state)))

    value = _state.State(position=updated_s, velocity=updated_v)
    return "pursuer_state", value


def static_target(params, substep, history, state, actions) -> tuple[str, _state.State]:
    """Persist target state.

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
        "target_state"
    value: _state.State
        The new value of the state variable.
    """

    tstate = state["target_state"]
    value = copy.deepcopy(tstate)

    return "target_state", value
