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
parser.add_argument("--port", type=int)

# additional catan engine parameters
parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
parser.add_argument("--assisted", action=argparse.BooleanOptionalAction)
parser.add_argument("--episodes", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument("--rolls", type=int)
parser.add_argument("--path", type=str)
parser.add_argument("--prefix", type=str)

args = parser.parse_args()

PATH = Path(args.path)

engine_parameters = catan_engine.EngineParameters(False, args.verbose, args.assisted, True, args.episodes, args.port, args.seed, args.rolls, True, args.prefix)
environment_parameters = environment.EnvironmentParams("naive", False, args.port)

pprint.pprint(engine_parameters, indent=4)
pprint.pprint(environment_parameters, indent=4)

start_catan_engine = catan_engine.get_launch_callback(engine_parameters)

policy, tf_environment = loader.get_initial_random_policy(environment_parameters)
policy = loader.load_policy(PATH)

start_catan_engine()

_, _, _ = player.play_episodes(policy, tf_environment, args.episodes, True)
tf_environment.close()
