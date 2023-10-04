import random
from pathlib import Path

import tensorflow as tf  # type: ignore
from tf_agents import agents  # type: ignore
from tf_agents.agents import TFAgent  # type: ignore
from tf_agents.environments import tf_py_environment  # type: ignore
from tf_agents.environments.tf_py_environment import TFPyEnvironment  # type: ignore
from tf_agents.policies import policy_saver  # type: ignore
from tf_agents.policies import tf_policy  # type: ignore
from tf_agents.policies.policy_saver import PolicySaver  # type: ignore
from tf_agents.policies.random_tf_policy import RandomTFPolicy  # type: ignore
from tf_agents.policies.tf_policy import TFPolicy  # type: ignore
from tf_agents.replay_buffers import tf_uniform_replay_buffer  # type: ignore
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer  # type: ignore
from tf_agents.utils import common  # type: ignore
from tf_agents.utils.common import Checkpointer  # type: ignore

from agent import AgentParams
from agent.agent import get_initialized_agent
from environment import CatanSocketEnvironment, EnvironmentParams

MasterComponents = tuple[
    agents.TFAgent,
    tf_py_environment.TFPyEnvironment,
    tf_uniform_replay_buffer.TFUniformReplayBuffer,
    common.Checkpointer,
    policy_saver.PolicySaver,
]


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


def load_recent_policy(policies: Path, window_width: int, offset: int) -> tf_policy.TFPolicy | None:
    """Loads a random recent policy from the given directory.

    Window width is always increase by one, if the last one is selected None
    will be returned. I.e. if ``window_width`` is 9 then there is a 10% to fail to
    load a new policy.

    :param policies: The directory to sample the policies from.
    :param window_width: The number of recent policies to sample from.
    :param offset: An offset to prevent loading of the offset-newest policies.
    :return: The loaded policy or none.
    """

    window = [*range(window_width + 1)]

    # sort policies by time of creation
    sub_directories = [directory for directory in policies.iterdir() if directory.is_dir()]
    sub_directories.sort(reverse=True, key=lambda path: path.stat().st_mtime)

    # set sample window to min amount of policies, add one as random sample
    window_width = min(len(sub_directories), window_width)
    chosen_index = random.choice(window)

    if chosen_index == max(window):
        print(f"No policy was chosen.")
        return None
    try:
        # correct by one to skip the latest, set an offset to use older policies
        chosen_index += 1
        chosen_index += offset

        print(f"Loaded policy from {sub_directories[chosen_index]}.")
        return load_policy(sub_directories[chosen_index])
    except:
        print(f"No policy was chosen.")
        return None


def load_latest_policy(policies: Path) -> tf_policy.TFPolicy:
    sub_directories = [directory for directory in policies.iterdir() if directory.is_dir()]
    sub_directories.sort(reverse=True, key=lambda path: int(path.name))
    print(f"loading {sub_directories[0]}")
    return load_policy(sub_directories[0])


def load_random_policy(policies: Path, random_chance: float = 0.1) -> tf_policy.TFPolicy | None:
    """Loads a random policy from all policies within a directory.

    :param policies: Directory where all potential policies are stored.
    :param random_chance: A random change between 0 and 1 to fail to load a policy, defaults to 0.1
    :return: A random chosen policy or none if the random check failed.
    """
    if random_chance > random.random():
        print(f"No policy was chosen.")
        return None

    sub_directories = [directory for directory in policies.iterdir() if directory.is_dir()]
    choice = random.choice([*range(len(sub_directories))])
    print(f"Loaded policy from {sub_directories[choice]}.")
    load_policy(sub_directories[choice])


def get_master(agent_params: AgentParams, environment_params: EnvironmentParams, agent_dir: Path, policy_dir: Path, buffer_dir: Path) -> MasterComponents:
    agent_dir.mkdir(parents=True, exist_ok=True)
    policy_dir.mkdir(parents=True, exist_ok=True)
    buffer_dir.mkdir(parents=True, exist_ok=True)

    agent, environment = _get_agent_and_environment(agent_params, environment_params)
    replay_buffer = restore_replay_buffer(agent_params.buffer_size, buffer_dir, agent, environment)
    checkpointer, saver = _setup_checkpointer(agent, replay_buffer, agent_dir, policy_dir)

    return agent, environment, replay_buffer, checkpointer, saver

def _get_agent_and_environment(agent_parameters: AgentParams, environment_parameters: EnvironmentParams) -> tuple[TFAgent, TFPyEnvironment]:
    py_environment = CatanSocketEnvironment(environment_parameters)
    agent, tf_environment = get_initialized_agent(py_environment, agent_parameters)
    return agent, tf_environment


def restore_replay_buffer(buffer_size: int, buffer_cache: Path, policy: TFPolicy | TFAgent, environment: TFPyEnvironment) -> TFUniformReplayBuffer:
    if not [*buffer_cache.iterdir()]:
        raise Exception("Buffer directory is empty, therefore not buffer to restore exists.")

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=policy.collect_data_spec,  # type: ignore
        batch_size=environment.batch_size,
        max_length=buffer_size,
    )
    checkpointer = common.Checkpointer(buffer_cache, max_to_keep=1, replay_buffer=replay_buffer)
    checkpointer.initialize_or_restore()  # type: ignore
    return replay_buffer


def _setup_checkpointer(agent: TFAgent, replay_buffer: TFUniformReplayBuffer, agent_cache: Path, policy_cache: Path) -> tuple[Checkpointer, PolicySaver]:
    checkpointer = common.Checkpointer(
        agent_cache,
        max_to_keep=1,
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

    return checkpointer, saver


def get_initial_random_policy(parameters: EnvironmentParams) -> tuple[RandomTFPolicy, TFPyEnvironment]:
    py_environment = CatanSocketEnvironment(parameters)
    tf_environment = TFPyEnvironment(py_environment)

    policy = RandomTFPolicy(
        tf_environment.time_step_spec(),  # type: ignore
        tf_environment.action_spec(),  # type: ignore
        observation_and_action_constraint_splitter=CatanSocketEnvironment.constraint_splitter,  # type: ignore
    )

    return policy, tf_environment


def get_initial_replay_buffer(
    buffer_size: int, buffer_cache: Path, policy: TFPolicy, environment: TFPyEnvironment
) -> tuple[TFUniformReplayBuffer, Checkpointer]:
    replay_buffer = TFUniformReplayBuffer(
        data_spec=policy.collect_data_spec,  # type: ignore
        batch_size=environment.batch_size,
        max_length=buffer_size,
    )
    checkpointer = common.Checkpointer(buffer_cache, max_to_keep=1, replay_buffer=replay_buffer)

    return replay_buffer, checkpointer
