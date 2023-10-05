import argparse
import gc
import pprint
import subprocess
from pathlib import Path
from typing import Any

import tensorflow as tf  # type: ignore
import absl.logging  # type: ignore
import psutil
import silence_tensorflow.auto  # type: ignore

import agent
import catan_engine
import environment
import metrics
from scripts import SlaveParameters
from utils import loader, player

absl.logging.set_verbosity(absl.logging.ERROR)  # type: ignore

process = psutil.Process()
process.cpu_affinity([*range(16)])
process.nice(psutil.HIGH_PRIORITY_CLASS)

parser = argparse.ArgumentParser()

# general parameters
parser.add_argument("--single", action=argparse.BooleanOptionalAction)
parser.add_argument("--train_async", action=argparse.BooleanOptionalAction)
parser.add_argument("--port", type=int)
parser.add_argument("--initial_name", type=str)
parser.add_argument("--name", type=str)

# additional catan engine parameters
parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
parser.add_argument("--assisted", action=argparse.BooleanOptionalAction)
parser.add_argument("--init", action=argparse.BooleanOptionalAction)
parser.add_argument("--seed", type=int)

# additional environment parameters
parser.add_argument("--reward_mode", type=str)
parser.add_argument("--use_end_signal", action=argparse.BooleanOptionalAction)

# additional training parameters
parser.add_argument("--training_intervals", type=int)
parser.add_argument("--training_episodes", type=int)
parser.add_argument("--evaluation_episodes", type=int)
parser.add_argument("--batchsize", type=int)

# additional agent parameters
parser.add_argument("--gamma", type=float)
parser.add_argument("--n_steps", type=int)
parser.add_argument("--soft_updates", action=argparse.BooleanOptionalAction)
parser.add_argument("--epsilon_steps", type=int)
parser.add_argument("--epsilon_end", type=float, default=0.1)
parser.add_argument("--buffer_size", type=int, default=100_000)

# additional slave parameters
parser.add_argument("--adaptive", action=argparse.BooleanOptionalAction)
parser.add_argument("--swap_start", type=int, default=0)
parser.add_argument("--swap_interval", type=int, default=0)
parser.add_argument("--window_width", type=int, default=0)
parser.add_argument("--window_offset", type=int, default=0)

args = parser.parse_args()

NUMBER_OF_EPISODES = args.evaluation_episodes + (args.training_episodes + args.evaluation_episodes) * args.training_intervals

folder = "single" if args.single else "dynamic"

BUFFER_CACHE_DIRECTORY = Path(f"./cache/buffers/{folder}/{args.initial_name}/")
POLICY_CACHE_DIRECTORY = Path(f"./cache/policies/{folder}/{str(args.name).replace('s13', 's7')}/")
AGENT_CACHE_DIRECTORY = Path(f"./cache/agents/{folder}/{str(args.name).replace('s13', 's7')}/")
METRICS_FILE_PATH = Path(f"./cache/metrics/{folder}/")

agent_parameters = agent.AgentParams(
    args.gamma, args.n_steps, args.batchsize, args.soft_updates, args.epsilon_steps, epsilon_end=args.epsilon_end, buffer_size=args.buffer_size
)

engine_parameters = catan_engine.EngineParameters(
    args.single,
    args.verbose,
    args.assisted,
    args.init,
    NUMBER_OF_EPISODES,
    args.port,
    args.seed,
)

environment_parameters = environment.EnvironmentParams(
    args.reward_mode,
    args.use_end_signal,
    args.port,
)

slave_parameters = SlaveParameters(
    args.port,
    NUMBER_OF_EPISODES,
    args.adaptive,
    str(POLICY_CACHE_DIRECTORY),
    args.swap_start,
    args.swap_interval,
    args.window_width,
    args.window_offset,
)

pprint.pprint(agent_parameters, indent=4)
pprint.pprint(engine_parameters, indent=4)
pprint.pprint(environment_parameters, indent=4)
pprint.pprint(slave_parameters, indent=4)

eval_writer = metrics.EvaluationWriter(METRICS_FILE_PATH / f"{args.name}.csv")
loss_writer = metrics.LossWriter(METRICS_FILE_PATH / f"{args.name}.loss.csv")

start_catan_engine = catan_engine.get_launch_callback(engine_parameters)
tf_agent, tf_environment, buffer, checkpointer, saver = loader.get_master(
    agent_parameters,
    environment_parameters,
    AGENT_CACHE_DIRECTORY,
    POLICY_CACHE_DIRECTORY,
    BUFFER_CACHE_DIRECTORY,
)

# set up slave agents
if not args.init:

    def get_slave_command(offset: int, parameters: SlaveParameters) -> str:
        mode = "single" if args.single else "dynamic"
        return rf"title Training Slave ({mode}, {args.reward_mode}, {parameters.port + offset}) && python slave.py {parameters.as_args(offset)}"

    slaves: list[subprocess.Popen[Any]] = [
        subprocess.Popen(["start", "cmd", "/c", get_slave_command(1, slave_parameters)], shell=True),
        subprocess.Popen(["start", "cmd", "/c", get_slave_command(2, slave_parameters)], shell=True),
        subprocess.Popen(["start", "cmd", "/c", get_slave_command(3, slave_parameters)], shell=True),
    ]


# define evaluation callback
def evaluation() -> None:
    rewards, steps, lengths = player.play_episodes(tf_agent.policy, tf_environment, args.evaluation_episodes)
    evaluation = metrics.EvaluationMetrics(rewards, steps, lengths, float(tf_agent._epsilon_greedy()))  # type: ignore
    eval_writer.add(evaluation)


start_catan_engine()
evaluation()

i = 0
for _ in range(args.training_intervals):
    if args.train_async:
        loss = player.train_episodes_async(tf_agent, tf_environment, buffer, agent_parameters, args.training_episodes)
    else:
        loss = player.train_episodes(tf_agent, tf_environment, buffer, agent_parameters, args.training_episodes)
    loss_writer.add(loss)
    evaluation()

    i += 1
    if i % 10 == 0:
        loader.save_agent(checkpointer, saver, tf_agent, POLICY_CACHE_DIRECTORY)
        i = 0

        # some clean up, may help with constantly increasing ram usage
        tf.keras.backend.clear_session()
        gc.collect()

    else:
        loader.save_policy(saver, tf_agent, POLICY_CACHE_DIRECTORY)

loader.save_agent(checkpointer, saver, tf_agent, POLICY_CACHE_DIRECTORY)
tf_environment.close()
