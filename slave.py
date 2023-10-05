import argparse
import gc
import random
from pathlib import Path

import tensorflow as tf  # type: ignore

import environment
from utils import loader, player

parser = argparse.ArgumentParser()

# additional environment parameters
parser.add_argument("--port", type=int)

# slave parameters
parser.add_argument("--episodes", type=int)

# adaptive slave parameters
parser.add_argument("--adaptive", action=argparse.BooleanOptionalAction, help="Whether to use random policy sampling or not.")
parser.add_argument("--seed", type=int, default=-1, help="Radom seed.")
parser.add_argument("--name", type=str, default="", help="Name of the folder to load policies from.")
parser.add_argument("--swap_start", type=int, default=0, help="An amount of episodes after which the random sampling starts.")
parser.add_argument("--swap_interval", type=int, default=0, help="The interval [episodes] in which the policy is re-chosen.")
parser.add_argument("--window_width", type=int, default=0, help="Sample window size i.e. the amount of possible old polices to choose from.")
parser.add_argument("--window_offset", type=int, default=0, help="Sample window size i.e. the amount of possible old polices to choose from.")

args = parser.parse_args()

if args.seed >= 0:
    random.seed(args.seed)


POLICY_CACHE_DIRECTORY = Path(args.name)
environment_parameters = environment.EnvironmentParams("naive", False, args.port)

if args.adaptive:
    # run a random policy for the first few episodes
    random_policy, tf_environment = loader.get_initial_random_policy(environment_parameters)
    _, _, _ = player.play_episodes(random_policy, tf_environment, args.swap_start, True)

    print("\nDone playing initial episodes, using adaptive policies going forward.\n")

    left_episodes = args.episodes - args.swap_start

    for _ in range(left_episodes // args.swap_interval):
        if policy:= loader.load_recent_policy(POLICY_CACHE_DIRECTORY, args.window_width, args.window_offset):
            _, _, _ = player.play_episodes(policy, tf_environment, args.swap_interval, True)
        else:
            _, _, _ = player.play_episodes(random_policy, tf_environment, args.swap_interval, True)


        # some clean up, may help with constantly increasing ram usage
        tf.keras.backend.clear_session()
        gc.collect()

    tf_environment.close()

else:
    # run a single random policy

    policy, tf_environment = loader.get_initial_random_policy(environment_parameters)
    _, _, _ = player.play_episodes(policy, tf_environment, args.episodes, True)
    tf_environment.close()
