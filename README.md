# cadR

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![pipeline status](https://gitlab.semiotic.ai/cadcad-experiments/cadr/badges/main/pipeline.svg)](https://gitlab.semiotic.ai/cadcad-experiments/cadr/-/commits/main)
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

## Usage

**Note:** Please see our [examples (TODO)]() to help you get started beyond this usage guide.

`cadR` is written in a somewhat functional style, though it is impossible to fully avoid side effects when using `PyTorch`.
As such, the general training process involves calling various functions.
We'll write pseudocode below and expand it using later.

**Note:** `cadr` almost always follows the convention of using keyword-only arguments.
Whilst maybe being somewhat tedious, this is much safer than positional arguments, in which you could get the positions wrong.
If you try to provide arguments to a function or method by position, your code will likely error out.

### Set Up
As mentioned in the **Motivation** section, the typical RL training loop looks like

```python
# Setup
for ep in range(num_episodes):
    # Collect rollouts
    # Update agent(s)
```

With `cadR`, we expand the setup step as follows.

```python
# Setup
# Set up buffer
# Set up networks
# Set up algorithm
# ...
```

The relevant `cadR` modules are somewhat aptly named.
To set up the buffer, use `cadr.buffer`.
To set up the networks, use `cadr.network`.
To set up the algorithm, use the name of the relevant algorithm (e.g., `cadr.td3`).
An RL algorithm needs several components to learn.
In addition to the buffer, we'll need loss functions, optimizers, and the agents themselves.
We'll also need an update function, but we'll come to that later.
The algorithm module (e.g., `cadr.td3`) contains helper functions to quickly set up these different components for you.
They follow the pattern: `cadr.[alg].[alg]_loss`, `cadr.[alg].[alg]_optimizer`, and `cadr.[alg].[alg]_agent`, respectively.
As an example, let's set up the Twin-Delayed DDPG (TD3) algorithm.

```python
# Setup
# Set up buffer
buffer = cadr.buffer.buffer(max_length=int(1e6))
# Set up networks
# TD3 involves and actor-critic with two critics. Hence the word "twin" in its name.
obs_dim = 4  # Size of the agent's observation vector
act_dim = 2  # Size of the agent's action vector
pi = cadr.network.mlp(  # actor
    activations=[nn.ReLU, nn.Tanh], layer_sizes=[obs_dim, 256, act_dim]
)
q = cadr.network.mlp(  # critic
    activations=[nn.ReLU, nn.Identity], layer_sizes=[obs_dim+act_dim, 256, 1]
)
qt = cadr.network.mlp(  # twin critic
    activations=[nn.ReLU, nn.Identity], layer_sizes=[obs_dim+act_dim, 256, 1]
)
# Set up algorithm
# Get the agent. TD3 uses a target agent for training stability.
agent, target = cadr.td3.td3_agent(action_scale=1.0, pi=pi, q=q, qt=qt)
actor_optim, critic_optim = cadr.td3.td3_optimizer(
    actor_lr=1e-4,
    actor_optimizer=optim.Adam,
    agent=agent,
    critic_lr=1e-4,
    critic_optimizer=optim.Adam,
)
actor_loss, critic_loss = cadr.td3.td3_loss(
    agent=agent, gamma=0.99, noise_clip=0.5, target=target, target_noise=0.2
)
```

With the RL algorithm set up, we now turn to setting up our simulation, which is handled by `cadCAD`.
We won't get too far into the details of `cadCAD` here as we assume that if you want to use `cadR`, it's because you have access to a `cadCAD` environment, but not a `gym` environment.
If you have access to a `gym` environment, we'd encourage you to use a different library, as other RL libraries are much more mature than `cadR`.
That being said, we will need to set up certain special components for using `cadCAD` in an RL setting.
The first thing to know is that the `"T"` parameter in `cadCAD` refers to the `episode_length` in an RL setting.
The `"N"` parameter is for Monte-Carlo simulation, and probably should be set to `1` for most RL algorithms.
We can also use the `"M"` parameter to specify any constants we may want to provide.
For example, we could provide a timestep size as `"M": {"delta_t": [0.1]}`.
Just note that since `"M"` is technically used for Monte Carlo simulations for sweeping parameters, you'll want to push any constants into a list of size 1.
We'll also want to set up the `partial_state_update_blocks`.
This is where we'll start to pass in some stuff explicitly related to RL.
Firstly, your policies need to translate the output of your agents to the input `cadCAD` expects.
For example:

```python
def cadCAD_rl_policy_example(params, substep, history, state):
    obs = state["agent_obseration"]
    agent = state["agent"]

    with torch.no_grad():
        act = agent.action(observation=torch.tensor(obs, dtype=torch.float32))

    return {"agent_action": act}
```

Within the `partial_state_update_blocks`, we will also need to provide functions to compute everything that the buffer will will store.
For example, for TD3, the buffer will store `("reward", "done", "observation", "action")`.
For example, our observation might just be the state of the agent, as given below.

```python
def agent_observation(params, substep, history, state, actions):
    agent_state = state["agent_state"]
    return "agent_observation", agent_state
```

You'd create one or more of these functions for each other key in the buffer.
Putting the pieces together then, an example `partial_state_update_blocks` might look like:

```python
partial_state_update_blocks = [
    {
        "policies": {
            "agent": cadCAD_rl_policy_example
        }
        "variables": {
            "agent_state": environment_dynamics,
            "agent_reward": agent_reward,
            "agent_done": agent_gameover,
            "agent_observation": agent_observation,
            "agent_action": agent_action
        }
    },
]
```

That's all of our setup that occurs outside of the training loop!
We still have some more setup within the training loop though, so we'll get to that next.


### The Training Loop

Typically, in a `gym` style training loop, the loop would continue while certain conditions haven't been met, be it the number of episodes or some sort of performance-based early stopping or something else.
`cadCAD` was not developed with any sort of early stopping in mind.
Instead, since `cadCAD` handles the entire simulation in one go, rather than breaking for updates like RL typically does, we'll have to loop over the number of episodes.
We can still have some sort of early stopping by using an if statement in the training loop

```python
for ep in range(num_episodes):
    # Rollout collection

    # Update

    # Early stopping
    if condition:
        break
```

`cadCAD` doesn't allow us to randomly initialise the states of our agents.
This is crucial for RL.
As a result, we actually finialise setting up our agents within this `for` loop so that we can use `numpy.random` or something similar.
Remember that for all of the variables you defined in `partial_state_update_blocks` before, you'll now need to specify some initial value.
You then execute the normal `cadCAD` workflow to run the simulation for 1 episode.

```python
for ep in range(num_episodes):
    # Rollout collection
    _state = np.random.uniform(-1, 1, 2),
    initial_state = {
        "agent_state": _state,
        "agent_reward": 0.0,
        "agent_done": 0,
        "agent_observation": _state,
        "agent_action": np.zeros((2,))
        "agent": agent,
    }
    experiment = Experiment()
    experiment.append_configs(
        sim_configs=sim_config,
        initial_state=initial_state,
        partial_state_update_blocks=partial_state_update_blocks,
    )
    exec_context = ExecutionContext()
    run = Executor(exec_context=exec_context, configs=experiment.configs)
    (system_events, tensor_field, sessions) = run.execute()

    # Update

    # Early stopping
    if condition:
        break
```

`system_events` contains all of the variables we decided to track over time, so we'll now need to use it to populate our buffer, which we do using `buffer.append`.
Depending on the algorithm, `cadR` will expect different things out of the buffer, so pay attention to the naming convention for the batch described by `cadR`'s agent documentation.

```python
for se, nse in zip(system_events[:-1], system_events[1:]):
    sample = {
        "observation": se["agent_observation"],
        "next_observation": nse["agent_observation"],
        "action": se["agent_action"],
        "reward": se["agent_reward"],
        "done": se["agent_done"],
    }
    buffer.append(sample)
```

Finally, we can use the populated buffer to update the algorithm's weights using `cadr.[alg].[alg]_update`.
For example, for TD3, the full training loop might look like:

```python
for ep in range(num_episodes):
    # Rollout collection
    _state = np.random.uniform(-1, 1, 2),
    initial_state = {
        "agent_state": _state,
        "agent_reward": 0.0,
        "agent_done": 0,
        "agent_observation": _state,
        "agent_action": np.zeros((2,))
        "agent": agent,
    }
    experiment = Experiment()
    experiment.append_configs(
        sim_configs=sim_config,
        initial_state=initial_state,
        partial_state_update_blocks=partial_state_update_blocks,
    )
    exec_context = ExecutionContext()
    run = Executor(exec_context=exec_context, configs=experiment.configs)
    (system_events, tensor_field, sessions) = run.execute()

    # Append to buffer
    for se, nse in zip(system_events[:-1], system_events[1:]):
        sample = {
            "observation": se["agent_observation"],
            "next_observation": nse["agent_observation"],
            "action": se["agent_action"],
            "reward": se["agent_reward"],
            "done": se["agent_done"],
        }
        buffer.append(sample)

    # Update
    cadr.td3.td3_update(
        agent=agent,
        actor_loss=actor_loss,
        actor_optim=actor_optim,
        actor_update_frequency=2,
        batch_size=32,
        buffer=buffer,
        critic_loss=critic_loss,
        critic_optim=critic_optim,
        device="cpu",
        gradient_steps=32,
        rho=0.995,
        target=target,
    )

    # Early stopping
    if condition:
        break
```

**Note:** The `cadr.[alg].[alg]_update` function will return the loss values as lists of floats, so if you wanted to plot the actor and critic losses for TD3, for example, grab these lists and plot them.
`cadR` doesn't currently have native support for a dashboard like `tensorboard`.
