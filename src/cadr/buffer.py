"""Implements buffers."""

from collections import deque
import random

import numpy as np
import torch

import cadr.utils as utils


def buffer(*, max_length: int) -> deque[dict[str, np.ndarray]]:
    """Create a buffer.

    Parameters
    ----------
    max_length: int
        The maximum length of the buffer.

    Returns
    -------
    buffer: deque[dict[str, np.ndarray]]
        The buffer.

    Raises
    ------
    ValueError:
        If max_length < 0
        If max_length is not an integer

    Examples
    --------
    >>> from cadr.buffer import buffer
    >>> my_buffer = buffer(max_length=int(1e6))
    >>> sample = {"observation": np.array([0, 1]), "next_observation": np.array([2, 3])}
    >>> my_buffer.append(sample)
    >>> len(my_buffer)
    1
    """
    try:
        buffer = deque(maxlen=max_length)
    except (ValueError, TypeError):
        raise ValueError("max_length must be a positive integer.")

    return buffer


def batch(
    *, batch_size: int, buffer: deque[dict[str, np.ndarray]], device: str
) -> dict[str, torch.Tensor]:
    """Return a batch of samples from the buffer.

    Parameters
    ----------
    batch_size: int
        The number of samples in each batch.
    buffer: deque[dict[str, np.ndarray]]
        The buffer from which to sample the batch.
    device: str
        "cpu" or "cuda".

    Returns
    -------
    batch: dict[str, torch.Tensor]
        The batch of samples from the buffer.

    Raises
    ------
    ValueError:
        If batch_size <= 0
    TypeError:
        If batch_size is not an integer

    Examples
    --------
    >>> from cadr.buffer import buffer, batch
    >>> my_buffer = buffer(max_length=5)
    >>> sample = {"observation": np.array([0, 1])}
    >>> my_buffer.append(sample)
    >>> my_buffer.append(sample)
    >>> my_buffer.append(sample)
    >>> batch(batch_size=2, buffer=my_buffer, device="cpu")
    {"observation": torch.Tensor([[0, 1], [0, 1]])}
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    list_batch = random.choices(buffer, k=batch_size)
    dict_batch = utils.list_to_dict(list_batch)
    # Convert to numpy before tensor as this is faster than straight to tensor
    batch = {
        k: torch.tensor(np.array(v), dtype=torch.float32, device=torch.device(device))
        for k, v in dict_batch.items()
    }
    return batch
