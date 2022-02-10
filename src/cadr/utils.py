"""Implements utility functions."""

from typing import Any

import numpy as np


def list_to_dict(value: list[dict[Any, np.ndarray]]) -> dict[Any, list[np.ndarray]]:
    """Convert a list of dicts to a dict of lists.

    Parameters
    ----------
    value: list[dict[Any, list[Any]]]
        The list to convert.

    Returns
    -------
    converted: dict[Any, list]
        The converted dict of lists.

    Examples
    --------
    >>> from cadr.utils import list_to_dict
    >>> value = [{"foo": [0, 1], "bar": [2, 3]}, {"foo": [2, 3], "bar": [0, 1]}]
    >>> list_to_dict(value)
    {"foo": [[0, 1], [2, 3]], "bar": [[2, 3], [0, 1]]}
    """
    converted = {k: [v[k] for v in value] for k in value[0]}
    return converted
