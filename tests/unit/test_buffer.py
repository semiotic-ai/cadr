"""Test the cadr.buffer module"""

import numpy as np
import pytest

import cadr.buffer as buffer


def test_buffer():
    with pytest.raises(ValueError):
        replay_buffer = buffer.buffer(max_length=-1)

    with pytest.raises(ValueError):
        replay_buffer = buffer.buffer(max_length=1.0)


def test_buffer_batch():
    replay_buffer = buffer.buffer(max_length=int(1e6))
    # Push sample to replay buffer multiple times
    for _ in range(10):
        sample = {
            "observation": np.array([1, 2, 3, 4]),
            "next_observation": np.array([5, 6, 7, 8]),
            "action": np.array([9, 10]),
            "reward": np.array([1.0]),
            "done": np.array([0]),
        }
        replay_buffer.append(sample)

    batch = buffer.batch(batch_size=4, buffer=replay_buffer, device="cpu")
    assert len(batch["observation"]) == 4

    with pytest.raises(ValueError):
        batch = buffer.batch(batch_size=0, buffer=replay_buffer, device="cpu")
