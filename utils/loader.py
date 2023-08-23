import random
from pathlib import Path

import tensorflow as tf  # type: ignore
from tf_agents import agents  # type: ignore
from tf_agents.environments import tf_py_environment  # type: ignore
from tf_agents.policies import policy_saver  # type: ignore
from tf_agents.policies import tf_policy  # type: ignore
from tf_agents.replay_buffers import tf_uniform_replay_buffer  # type: ignore
from tf_agents.utils import common  # type: ignore

from agent.agent import build_agent
from environment.environment import CatanEnvironment


def save_agent(checkpointer: common.Checkpointer, saver: policy_saver.PolicySaver, agent: agents.TFAgent, policy_cache: Path):
    """Saves an agent using the given checkpointer and policy saver.

    :param checkpointer: The checkpointer to use.
    :param saver: The policy saver to use.
    :param agent: The agent to save.
    :param policy_cache: The directory to save the policy to.
    """
    checkpointer.save(agent.train_step_counter)  # type: ignore
    save_policy(saver, agent, policy_cache)


def save_policy(saver: policy_saver.PolicySaver, agent: agents.TFAgent, policy_cache: Path) -> None:
    """Saves the given agents policy using the given policy saver.

    :param saver: The policy saver to use.
    :param agent: The agent to save.
    :param policy_cache: The directory to save the policy to.
    """
    saver.save(policy_cache / f"{agent.train_step_counter.value()}/")  # type: ignore


def load_policy(policy: Path) -> tf_policy.TFPolicy:
    """Loads the policy from the given directory.

    :param policy: The directory to load the policy from.
    :return: The loaded policy.
    """
    return tf.saved_model.load(policy)  # type: ignore


def load_recent_policy(policies: Path, window_width: int, seed: int = 0) -> tf_policy.TFPolicy:
    """Loads a random recent policy from the given directory.

    :param policy: The directory to sample the policy from.
    :param window_width: The number of recent policies to sample from.
    :param seed: The seed used for the random selection, defaults to 0
    :return: The loaded policy.
    """
    # wait for the first policy if this is the first call

    sub_directories = [directory for directory in policies.iterdir() if directory.is_dir()]
    sub_directories.sort(reverse=False)

    # random.seed(seed)
    window_width = min(len(sub_directories), window_width)
    choice = random.choice([*range(window_width)])

    print(choice)

    return load_policy(sub_directories[choice])


SlaveComponents = tuple[
    tf_policy.TFPolicy,
    tf_py_environment.TFPyEnvironment,
]


def get_slave(port: int, policy_cache: Path, window_width: int, seed: int = 0) -> SlaveComponents:
    """Loads the initial components to run the script in slave mode.

    :param port: The port used by the environment.
    :param policy: The directory to sample the policy from.
    :param window_width: The number of recent policies to sample from.
    :param seed: The seed used for the random selection, defaults to 0
    :return: A loaded policy and the initialized environment.
    """
    py_environment = CatanEnvironment(port)
    tf_environment = tf_py_environment.TFPyEnvironment(py_environment)
    policy = load_recent_policy(policy_cache, window_width, seed)
    return policy, tf_environment


MasterComponents = tuple[
    agents.TFAgent,
    tf_py_environment.TFPyEnvironment,
    tf_uniform_replay_buffer.TFUniformReplayBuffer,
    common.Checkpointer,
    policy_saver.PolicySaver,
]


def get_master(port: int, agent_cache: Path, policy_cache: Path) -> MasterComponents:
    """Loads the initial components to run the script in master mode.

    :param port: The port used by the environment.
    :param policy: The directory to save/load the agent to/from.
    :param policy: The directory to save the initial policy to.
    :return: The loaded agent, the initialized environment, the restored
        replay buffer and a checkpointer and policy saver.
    """
    agent_cache.mkdir(parents=True, exist_ok=True)
    policy_cache.mkdir(parents=True, exist_ok=True)

    py_environment = CatanEnvironment(port)
    agent, tf_environment = build_agent(py_environment)
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,  # type: ignore
        batch_size=tf_environment.batch_size,
        max_length=1_000_000,
    )

    checkpointer = common.Checkpointer(
        agent_cache,
        max_to_keep=4,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=agent.train_step_counter,
    )

    if [*agent_cache.iterdir()]:
        checkpointer.initialize_or_restore()  # type: ignore

    saver = policy_saver.PolicySaver(agent.policy)

    if not [*policy_cache.iterdir()]:
        save_policy(saver, agent, policy_cache)

    return agent, tf_environment, replay_buffer, checkpointer, saver
