import tensorflow as tf  # type: ignore
from tf_agents import agents  # type: ignore
from tf_agents.agents.dqn import dqn_agent  # type: ignore
from tf_agents.environments import tf_py_environment  # type: ignore

from environment.environment import ACTION_SPEC, OBSERVATION_SPEC, CatanEnvironment

from .network import build_network


def build_agent(environment: CatanEnvironment) -> tuple[agents.TFAgent, tf_py_environment.TFPyEnvironment]:
    tf_environment = tf_py_environment.TFPyEnvironment(environment)

    optimizer = tf.keras.optimizers.Adam()

    agent = dqn_agent.DqnAgent(
        tf_environment.time_step_spec(),  # type: ignore
        tf_environment.action_spec(),  # type: ignore
        q_network=build_network(OBSERVATION_SPEC, ACTION_SPEC),
        optimizer=optimizer,
        observation_and_action_constraint_splitter=CatanEnvironment.constraint_splitter,  # type: ignore
        train_step_counter=tf.compat.v1.train.get_or_create_global_step(),  # type: ignore
    )

    agent.initialize()

    return agent, tf_environment
