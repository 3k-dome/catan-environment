import gc
import time
from itertools import chain

import tensorflow as tf  # type: ignore
import tqdm
from tf_agents.agents import TFAgent  # type: ignore
from tf_agents.environments.tf_py_environment import TFPyEnvironment  # type: ignore
from tf_agents.policies.tf_policy import TFPolicy  # type: ignore
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer  # type: ignore
from tf_agents.trajectories import trajectory  # type: ignore

from agent.parameters import AgentParams  # type: ignore


def play_episode(policy: TFPolicy, tf_environment: TFPyEnvironment) -> tuple[list[float], int, float]:
    """Deploys a given policy within the given environment for exactly one episode.

    :param environment: The environment to deploy the policy in.
    :param policy: The policy used for decision making.
    """
    reward: list[float] = []
    steps = 0
    start = time.time()

    time_step = tf_environment.reset()
    while not time_step.is_last():  # type: ignore
        action_step = policy.action(time_step)  # type: ignore
        time_step = tf_environment.step(action_step.action)  # type: ignore

        reward.append(float(time_step.reward))  # type: ignore
        steps += 1

    return reward, steps, (time.time() - start)


def play_episodes(
    policy: TFPolicy, tf_environment: TFPyEnvironment, no_episodes: int, omit_results: bool = False
) -> tuple[list[list[float]], list[int], list[float]]:
    """Deploys a given policy within the given environment until a set number of episodes passed.

    :param environment: The environment to deploy the policy in.
    :param policy: The policy used for decision making.
    :param no_episodes: The number of episodes to run.
    """
    episode_rewards: list[list[float]] = []
    episode_steps: list[int] = []
    episode_lengths: list[float] = []

    for _ in tqdm.tqdm(range(no_episodes), desc="Playing"):
        reward, steps, length = play_episode(policy, tf_environment)

        if omit_results:
            continue

        episode_rewards.append(reward)
        episode_steps.append(steps)
        episode_lengths.append(length)

    # some clean up, may help with constantly increasing ram usage
    tf.keras.backend.clear_session()
    gc.collect()

    return episode_rewards, episode_steps, episode_lengths


def collect_episode(policy: TFPolicy, tf_environment: TFPyEnvironment, replay_buffer: TFUniformReplayBuffer) -> int:
    steps = 0
    time_step = tf_environment.reset()
    while not time_step.is_last():  # type: ignore
        action_step = policy.action(time_step)  # type: ignore
        next_time_step = tf_environment.step(action_step.action)  # type: ignore

        transition = trajectory.from_transition(time_step, action_step, next_time_step)  # type: ignore
        replay_buffer.add_batch(transition)  # type: ignore

        time_step = next_time_step
        steps += 1

    return steps


def collect_episodes(policy: TFPolicy, tf_environment: TFPyEnvironment, replay_buffer: TFUniformReplayBuffer, no_episodes: int) -> None:
    steps = 0
    for _ in tqdm.tqdm(range(no_episodes), desc="Collecting"):
        steps += collect_episode(policy, tf_environment, replay_buffer)

    print(steps)


def train_episode(agent: TFAgent, environment: TFPyEnvironment, buffer: TFUniformReplayBuffer, parameters: AgentParams) -> list[float]:
    steps = 0
    loss_info: list[float] = []

    time_step = environment.reset()
    while not time_step.is_last():  # type: ignore
        # select action
        action_step = agent.collect_policy.action(time_step)  # type: ignore
        next_time_step = environment.step(action_step.action)  # type: ignore

        # add transition to buffer and reset
        transition = trajectory.from_transition(time_step, action_step, next_time_step)  # type: ignore
        buffer.add_batch(transition)  # type: ignore
        time_step = next_time_step

        # train each n-th step
        steps += 1
        if steps % parameters.network_update_frequency == 0:
            batch, _ = buffer.get_next(parameters.batchsize, parameters.n_steps + 1)  # type: ignore
            loss = agent.train(batch)  # type: ignore
            loss_info.append(float(loss.loss))  # type: ignore

    return loss_info


def train_episodes(agent: TFAgent, environment: TFPyEnvironment, buffer: TFUniformReplayBuffer, parameters: AgentParams, no_episodes: int) -> list[float]:
    loss_info: list[list[float]] = []
    for _ in tqdm.tqdm(range(no_episodes), desc="Training"):
        loss = train_episode(agent, environment, buffer, parameters)
        loss_info.append(loss)

    return [*chain(*loss_info)]


def train_episodes_async(
    agent: TFAgent, environment: TFPyEnvironment, buffer: TFUniformReplayBuffer, parameters: AgentParams, no_episodes: int
) -> list[float]:
    steps = 0
    for _ in tqdm.tqdm(range(no_episodes), desc="Collecting"):
        steps += collect_episode(agent.collect_policy, environment, buffer)

    loss_info: list[float] = []
    dataset = buffer.as_dataset(
        num_parallel_calls=4,
        sample_batch_size=parameters.batchsize,
        num_steps=parameters.n_steps + 1,
    )

    batch_iterator = iter(dataset)
    for _ in tqdm.tqdm(range(steps // parameters.network_update_frequency), desc="Training"):
        batch, _ = next(batch_iterator)
        loss = agent.train(batch)
        loss_info.append(float(loss.loss))

    return loss_info
