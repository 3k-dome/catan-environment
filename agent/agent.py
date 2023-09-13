from typing import Callable

import tensorflow as tf  # type: ignore
from tensorflow.python.ops.variables import Variable  # type: ignore
from tf_agents import agents  # type: ignore
from tf_agents.agents.dqn import dqn_agent  # type: ignore
from tf_agents.environments import tf_py_environment  # type: ignore

from environment.environment import ACTION_SPEC, OBSERVATION_SPEC, CatanRemoteEnvironment

from agent.network import build_network
from agent.parameters import AgentParameters


def _setup_epsilon_decay_callback(train_step: Variable, parameters: AgentParameters) -> Callable[..., float]:
    epsilon_decay: Callable[[int], float] = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=parameters.epsilon_start,
        decay_steps=parameters.epsilon_steps,
        end_learning_rate=parameters.epsilon_end,
    )

    return lambda: epsilon_decay(train_step)  # type: ignore


def get_initialized_agent(environment: CatanRemoteEnvironment, parameters: AgentParameters) -> tuple[agents.TFAgent, tf_py_environment.TFPyEnvironment]:
    tf_environment = tf_py_environment.TFPyEnvironment(environment)

    optimizer = tf.keras.optimizers.Adam()
    train_step: Variable = tf.compat.v1.train.get_or_create_global_step()  # type: ignore
    epsilon_decay_callback = _setup_epsilon_decay_callback(train_step, parameters)

    agent = dqn_agent.DqnAgent(
        tf_environment.time_step_spec(),  # type: ignore
        tf_environment.action_spec(),  # type: ignore
        q_network=build_network(OBSERVATION_SPEC, ACTION_SPEC),
        n_step_update=parameters.n_steps,
        optimizer=optimizer,
        epsilon_greedy=epsilon_decay_callback,
        gamma=parameters.gamma,
        observation_and_action_constraint_splitter=CatanRemoteEnvironment.constraint_splitter,  # type: ignore
        train_step_counter=train_step,
        target_update_period=parameters.target_update_frequency,
        target_update_tau=parameters.target_update_tau,
    )

    agent.initialize()

    return agent, tf_environment
