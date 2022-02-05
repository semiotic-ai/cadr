"""Test the cadr.buffer module"""

import numpy as np
import pytest

import cadr.buffer


@pytest.mark.skip("Not Implemented")
@pytest.fixture
def replay_buffer():
    return cadr.buffer.ReplayBuffer(
        action_size=2,
        observation_size=4,
        max_length=int(1e6),
    )


@pytest.mark.skip("Not Implemented")
def test_replaybuffer_push(replay_buffer):
    assert len(replay_buffer) == 0

    # Push sample to replay buffer
    sample = {
        "observation": np.array([1, 2, 3, 4]),
        "next_observation": np.array([5, 6, 7, 8]),
        "action": np.array([9, 10]),
        "reward": np.array([1.0]),
        "done": np.array([0]),
    }
    replay_buffer.push(sample=sample)

    assert len(replay_buffer) == 1


@pytest.mark.skip("Not Implemented")
def test_replaybuffer_batch(replay_buffer):
    # Push sample to replay buffer multiple times
    for _ in range(10):
        sample = {
            "observation": np.array([1, 2, 3, 4]),
            "next_observation": np.array([5, 6, 7, 8]),
            "action": np.array([9, 10]),
            "reward": np.array([1.0]),
            "done": np.array([0]),
        }
        replay_buffer.push(sample=sample)

    batch = buffer.batch(batch_size=4)
    assert len(batch["observation"]) == 4
