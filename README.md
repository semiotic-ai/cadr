# cadR

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![pipeline status](https://gitlab.semiotic.ai/cadcad-experiments/cadr/badges/main/pipeline.svg)](https://gitlab.semiotic.ai/cadcad-experiments/cadr/-/commits/main)
[![coverage report](https://gitlab.semiotic.ai/cadcad-experiments/cadr/badges/main/coverage.svg)](https://gitlab.semiotic.ai/cadcad-experiments/cadr/-/commits/main)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`cadR` is a reinforcement library compatible with [`cadCAD`](https://cadcad.org/).


## Motivation

`cadCAD`'s design philosophy inverts the dependency between *environment* and *algorithm* as compared with a standard reinforcement learning setup.
In a standard reinforcement learning paradigm, using an environment built around [OpenAI Gym](https://gym.openai.com/), the agent has a training loop that calls the `environment.step()` and `environment.reset()` methods.
The training loop collects samples from the environment and stores them in a buffer until it is ready to update the algorithm's parameters.
This process is called *rollout collection*.

```python
def rollout_collection(
    algorithm: object,
    environment: object,
    buffer: object,
    num_episodes_per_update: int,
    num_timesteps_per_episode: int
):
    for ep in range(num_episodes_per_update):
        observation = environment.reset() for t in range(num_timesteps_per_episode):
            action = algorithm.forward(observation)
            buffer.push(observation, action, **other_params)
            observation = environment.step()
```

**Note 1:** The above is a highly simplified rollout collection function.
Typically, you would not run this for a set number of episodes.
Rather, rollout collection would return when some condition about when-to-update was met.
For example, rollout collection might return based on `total_timesteps % rollouts_per_update == 0`.

The training loop then enters the algorithm update step, in which the collected rollouts are used to update the algorithm's parameters.

```python
def update(algorithm: object, buffer: object, gradient_steps: int, batch_size: int):
    for _ in range(gradient_steps):
        batch = buffer.batch(batch_size)
        loss = algorithm.loss(batch)
        algorithm.backprop(loss)
```

**Note 2:** Again, this is highly oversimplified, not least because the update step would work differently for on- and off-policy algorithms.

The overall training loop in a typical RL flow looks like:

```python
for _ in range(num_training_iters):
    rollout_collection()
    update()
```

Let's focus on rollout collection again to understand why we need a novel RL implementation for use with `cadCAD`.
When using `cadCAD`, our `environment` is replaced with a `cadCAD` simulation.
However, `cadCAD` is meant as a Monte Carlo simulation for heuristic agents, meaning it does not need to expose an `environment.step` or `environment.reset` method.
In this sense, it is incompatible with the OpenAI Gym API.
Most existing RL implementations depend on a Gym-like interface.
Their `rollout_collection` step calls `environment.step` and `environment.reset`.
On the other hand, `cadCAD` effectively replaces the entire `rollout_collection` method, with the exception of pushing samples to the buffer.
As such, `cadCAD` is incompatible with most popular RL libraries.
If we want to use RL within the `cadCAD` ecosystem, we will need to develop a new RL library.
Hence, `cadR`.

## Installation

There are several installation options.
Choose the one that best suits your need.

### Pip

```bash
$ pip install -r requirements.txt
$ pip install -e src/
```

**Note:** The `-e` flag installs `cadR` as a developer.
If you only care about using the library, use `pip install src/`

To test:

```bash
$ python -m pytest tests/
```

### Minimal (Not Recommended)

```bash
$ pip install -e src/ --find-links https://download.pytorch.org/whl/torch_stable.html
```

**Note:** The `-e` flag installs `cadR` as a developer.
If you only care about using the library, use `pip install src/ --find-links https://download.pytorch.org/whl/torch_stable.html`

To test:

```bash
$ python -m pytest tests/
```

### Docker (Not yet implemented)

```bash
$ make build
```

To test:

```bash
$ make test
```
