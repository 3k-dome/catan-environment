import builtins
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from environment.enums import MessageType, PlayerNumber

from .base_model import BaseModel


@dataclass
class ReceivedStateModel(BaseModel):
    player_number: PlayerNumber
    type: MessageType
    step: int
    state: list[float]
    mask: list[float]

    def __post_init__(self):
        """Handle datatype conversions for enum types.

        Depending on the used enum serialization and/or deserialization
        the passed values could be of type `int` i.e. the enums value
        or of type `str` i.e. the name of the value.

        Conversion is not handled automatically.
        """

        match type(self.player_number):
            case builtins.str:
                self.player_number = PlayerNumber[cast(str, self.type)]
            case builtins.int:
                self.player_number = PlayerNumber(self.type)
            case _:
                pass

        match type(self.type):
            case builtins.str:
                self.type = MessageType[cast(str, self.type)]
            case builtins.int:
                self.type = MessageType(self.type)
            case _:
                pass

    def to_observation(self) -> dict[str, NDArray[np.float32] | NDArray[np.int32]]:
        """Converts this model to an observation spec.

        :return: A dictionary containing both the 'observation' and 'mask'
            as numpy arrays to be used within the environment.
        """
        return {
            "observation": np.array(self.state, dtype=np.float32),
            "mask": np.array(self.mask, dtype=np.int32),
        }
