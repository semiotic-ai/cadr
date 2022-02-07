"""Implements states."""

from typing import NamedTuple

import numpy as np


class State(NamedTuple):
    """The state of an entity.

    Attributes
    ----------
    position: np.ndarray
        The position of the entity.
    velocity: np.ndarray
        The velocity of the entity.
    """

    position: np.ndarray
    velocity: np.ndarray
