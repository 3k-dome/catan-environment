import logging
from threading import Thread
from typing import Literal, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from tf_agents import trajectories  # type: ignore
from tf_agents.environments.py_environment import PyEnvironment  # type: ignore
from tf_agents.specs.array_spec import BoundedArraySpec  # type: ignore
from tf_agents.trajectories.time_step import TimeStep  # type: ignore

from environment.enums import MessageType, PlayerNumber
from environment.models import SubmittedActionModel
from environment.parameters import EnvironmentParameters
from environment.queues import ENVIRONMENT_ACTION, ENVIRONMENT_STATE
from environment import server

LOGGER = logging.getLogger("catan-environment")

ACTION_SPEC = 218
OBSERVATION_SPEC = 841


T = TypeVar("T")


class CatanRemoteEnvironment(PyEnvironment):
    def __init__(self, parameters: EnvironmentParameters):
        super().__init__(False)

        self.reward_mode: float | Literal["naive"] = parameters.reward_mode
        self.use_episode_end_signal = parameters.episode_end_signal

        self.player_number = PlayerNumber.ONE

        self._action_spec = BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=ACTION_SPEC - 1,
            name="action",
        )

        self._observation_spec = {
            "observation": BoundedArraySpec(
                shape=(OBSERVATION_SPEC,),
                dtype=np.float32,
                minimum=0,
                maximum=1,
                name="observation",
            ),
            "mask": BoundedArraySpec(
                shape=(ACTION_SPEC,),
                dtype=np.int32,
                minimum=0,
                maximum=1,
                name="mask",
            ),
        }

    @staticmethod
    def constraint_splitter(state: dict[str, T]) -> tuple[T, T]:
        return state["observation"], state["mask"]

    def _reset(self) -> TimeStep:
        """Resets the current environment and starts a new episode.

        This will reset the `_episode_ended` flag to `false`, load a new
        state into `_state` and return a transition that marks the start
        of a new episode.

        This implementation will block until a reset from the external
        environment is received.

        :return: The start transition of the new episode.
        """
        model = ENVIRONMENT_STATE.get()

        if model.type != MessageType.EPISODE_STARTS:
            raise Exception("This episode has not yet ended, something must have gone wrong!")

        self.player_number = model.player_number
        self._episode_ended = False
        self._state = model.to_observation()

        return trajectories.restart(self._state)  # type: ignore

    def _perform_action(self, action: NDArray[np.int32]) -> tuple[dict[str, NDArray[np.float32] | NDArray[np.int32]], MessageType]:
        """Submits the chosen action to the environment and returns the new state.

        The chosen action is pushed into the response queue of the underlying server,
        and then we wait until a new state is received and pushed into the request queue.

        :param action: The chosen action passed to the `_step` method.
        :return: The new observation of the environment.
        """
        action_model = SubmittedActionModel(self.player_number, int(action))
        ENVIRONMENT_ACTION.put(action_model)
        state_model = ENVIRONMENT_STATE.get()

        return state_model.to_observation(), state_model.type

    def _perform_dummy_action(self) -> None:
        """Send a dummy action back the environment.

        The current environment is setup using a simple HTTP server and therefore
        uses POST requests to communicate states.

        A state is send from the environment to the agent and the agent answers
        the request with ist chosen action.

        If the episode ends with the current state the agent is no longer required
        to select and send an action, therefore a dummy action is send.
        """
        action_model = SubmittedActionModel(self.player_number, -1)
        ENVIRONMENT_ACTION.put(action_model)

    @staticmethod
    def episode_end_signal(current_observation: NDArray[np.float32]) -> float | None:
        if current_observation[0] < 1:
            reward = -1.0
            LOGGER.info(f"Reward := {reward}, Episode lost!")
            return reward

        LOGGER.info(f"Agent did not lose, using default reward.")
        return None

    @staticmethod
    def naive_additive_reward(old_observation: NDArray[np.float32], new_observation: NDArray[np.float32]) -> float:
        """Naive reward function that simply assigns rewards based on the victory points gained."""
        delta_vp = new_observation[0] - old_observation[0]
        LOGGER.info(f"Reward := {delta_vp}")
        return delta_vp

    @staticmethod
    def polynomial_distance_reward(current_observation: NDArray[np.float32], degree: float) -> float:
        """Reward function based on a negative target distance with polynomial weighting."""
        corrected_target_offset = min(current_observation[0], 1)
        reward = (corrected_target_offset**degree) - 1
        LOGGER.info(f"Reward := {reward}")
        return reward

    def _calculate_rewards(self, message_type: MessageType, old_observation: NDArray[np.float32], new_observation: NDArray[np.float32]) -> float:
        """Calculate rewards based on the old and new state after any taken action.

        :param old_observation: The old observation before the latest action was taken.
        :param new_observation: The new observation after the latest action was taken.
        :return: A value representing the reward for this action.
        """

        if message_type == MessageType.EPISODE_ENDS and self.use_episode_end_signal:
            if reward := self.episode_end_signal(new_observation):
                return reward

        return (
            CatanRemoteEnvironment.naive_additive_reward(old_observation, new_observation)
            if self.reward_mode == "naive"
            else CatanRemoteEnvironment.polynomial_distance_reward(new_observation, self.reward_mode)
        )

    def _step(self, action: NDArray[np.int32]) -> TimeStep:  # type: ignore
        """Updates the environment and returns the next transition.

        :param action: The action the agent chose.
        :return: A new transition that either continues or ends the current episode.
        """
        if self._episode_ended:
            return self.reset()

        observation, message_type = self._perform_action(action)
        reward = self._calculate_rewards(
            message_type,
            cast(NDArray[np.float32], CatanRemoteEnvironment.constraint_splitter(self._state)[0]),
            cast(NDArray[np.float32], CatanRemoteEnvironment.constraint_splitter(observation)[0]),
        )

        self._state = observation

        match message_type:
            case MessageType.EPISODE_CONTINUES:
                return trajectories.transition(self._state, reward)  # type: ignore

            case MessageType.EPISODE_ENDS:
                self._episode_ended = True
                self._perform_dummy_action()
                return trajectories.termination(self._state, reward)  # type: ignore

            case _:
                raise Exception("This episode has not yet ended, something must have gone wrong!")

    def action_spec(self) -> BoundedArraySpec:
        return self._action_spec

    def observation_spec(self) -> dict[str, BoundedArraySpec]:
        return self._observation_spec


# this could be moved into the base class using generics but for now this is much simpler


class CatanHttpEnvironment(CatanRemoteEnvironment):
    def __init__(self, parameters: EnvironmentParameters):
        super().__init__(parameters)
        self.server, self.start_callback, self.stop_callback = server.EnvironmentHttpServer.server_factory(parameters.host, parameters.port)
        self.server_thread = Thread(target=self.start_callback)
        self.server_thread.daemon = True
        self.server_thread.start()

    def close(self) -> None:
        """Closes the environment by stopping the underlying http server."""
        self.stop_callback()
        return super().close()


class CatanSocketEnvironment(CatanRemoteEnvironment):
    def __init__(self, parameters: EnvironmentParameters):
        super().__init__(parameters)
        self.server, self.start_callback, self.stop_callback = server.EnvironmentSocketServer.server_factory(parameters.host, parameters.port)
        self.server_thread = Thread(target=self.start_callback)
        self.server_thread.daemon = True
        self.server_thread.start()

    def close(self) -> None:
        """Closes the environment by stopping the underlying socket server."""
        self.stop_callback()
        return super().close()
