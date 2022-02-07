"""Implements helper functions."""

from typing import NamedTuple

import numpy as np


def kinematics(s: float, v: float, a: float, t: float) -> tuple[float, float]:
    """Compute a kinematics update to the position and velocity.

    Parameters
    ----------
    s: float
        The position.
    v: float
        The velocity.
    a: float
        The acceleration.
    t: float
        Delta time.

    Returns
    -------
    _s: float
        The updated position.
    _v: float
        The updated velocity.
    """
    _s = s + (v * t) + (0.5 * a * t ** 2)
    _v = v + (a * t)

    return _s, _v


def unit_vector(x: np.ndarray) -> np.ndarray:
    """Compute the unit vector of a vector."""
    return x / np.linalg.norm(x)
