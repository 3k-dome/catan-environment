import argparse
import pprint
from pathlib import Path

import absl.logging  # type: ignore
import silence_tensorflow.auto  # type: ignore

import catan_engine
import environment
from utils import loader, player

absl.logging.set_verbosity(absl.logging.ERROR)  # type: ignore


parser = argparse.ArgumentParser()

# general parameters
parser.add_argument("--single", action=argparse.BooleanOptionalAction)
parser.add_argument("--port", type=int)
parser.add_argument("--name", type=str)
parser.add_argument("--buffer_size", type=int, default=100_000)

# additional catan engine parameters
parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
parser.add_argument("--assisted", action=argparse.BooleanOptionalAction)
parser.add_argument("--episodes", type=int)
parser.add_argument("--seed", type=int)

# additional environment parameters
parser.add_argument("--reward_mode", type=str)
parser.add_argument("--use_end_signal", action=argparse.BooleanOptionalAction)

args = parser.parse_args()

BUFFER_CACHE_DIRECTORY = Path(f"./cache/buffers/{'single' if args.single else 'dynamic'}/{args.name}")

engine_parameters = catan_engine.EngineParameters(args.single, args.verbose, args.assisted, True, args.episodes, args.port, args.seed)
environment_parameters = environment.EnvironmentParams(args.reward_mode, args.use_end_signal, args.port)

pprint.pprint(engine_parameters, indent=4)
pprint.pprint(environment_parameters, indent=4)

start_catan_engine = catan_engine.get_launch_callback(engine_parameters)
policy, tf_environment = loader.get_initial_random_policy(environment_parameters)
buffer, checkpointer = loader.get_initial_replay_buffer(args.buffer_size, BUFFER_CACHE_DIRECTORY, policy, tf_environment)

start_catan_engine()

player.collect_episodes(policy, tf_environment, buffer, args.episodes)

checkpointer.save(0)  # type: ignore
tf_environment.close()
