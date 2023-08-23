import tensorflow as tf  # type: ignore
from tf_agents import agents  # type: ignore
from tf_agents.environments.tf_py_environment import TFPyEnvironment  # type: ignore
from tf_agents.policies.tf_policy import TFPolicy  # type: ignore
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer  # type: ignore
from tf_agents.trajectories import trajectory  # type: ignore


def play_episodes(environment: TFPyEnvironment, policy: TFPolicy, no_episodes: int = 10) -> None:
    """Deploys a given policy within the given environment until a set number of episodes passed.

    :param environment: The environment to act in.
    :param policy: The policy used for decision making.
    :param no_episodes: The number of episodes to run, defaults to 10
    """
    for _ in range(no_episodes):
        time_step = environment.reset()

        while not time_step.is_last():  # type: ignore
            action_step = policy.action(time_step)  # type: ignore
            time_step = environment.step(action_step.action)  # type: ignore


def collect_episodes(environment: TFPyEnvironment, policy: TFPolicy, replay_buffer: TFUniformReplayBuffer, no_episodes: int = 10):
    """Samples a given policy within the given environment until a set number of episodes passed.

    `tf_agents` seams to prefer the usage of drives to collect a set number of steps. This
    method collects all steps until a set number of episodes have passed instead. 
    
    Collecting for whole episodes should increase the diversity of the collected data since the
    layout of the games board is fixed for the whole episode and will not change until a new episode
    is started. Therefore collecting only low amounts of steps could possibly result in a biased
    dataset.
    
    :param environment: The environment to act in.
    :param policy: The policy used for decision making.
    :param replay_buffer: The buffer to collect into.
    :param no_episodes: The number of episodes to run, defaults to 10
    """
    for _ in range(no_episodes):
        time_step = environment.reset()

        while not time_step.is_last():  # type: ignore
            action_step = policy.action(time_step)  # type: ignore
            next_time_step = environment.step(action_step.action)  # type: ignore

            transition = trajectory.from_transition(time_step, action_step, next_time_step)  # type: ignore
            replay_buffer.add_batch(transition)  # type: ignore

            time_step = next_time_step


def train_agent(agent: agents.TFAgent, replay_buffer: TFUniformReplayBuffer, batch_size: int, no_batches: int) -> None:
    """Trains an agent using the given replay buffer.

    :param agent: The agent to train.
    :param replay_buffer: The buffer to sample data from.
    :param batch_size: The batchsize for training.
    :param no_batches: The number of batches to train on.
    """
    dataset: tf.data.Dataset = replay_buffer.as_dataset(num_parallel_calls=4, sample_batch_size=batch_size, num_steps=2).prefetch(4)  # type: ignore
    batches = iter(dataset)  # type: ignore

    for _ in range(no_batches):
        batch, _ = next(batches)  # type: ignore
        _ = agent.train(batch)  # type: ignore
