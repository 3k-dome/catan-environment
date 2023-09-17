from dataclasses import dataclass
from typing import Literal


@dataclass
class EnvironmentParameters:
    reward_mode: float | Literal["naive"]
    episode_end_signal: bool
    port: int
    host: str = ""

    def __post_init__(self) -> None:
        """Convert `reward_type` from given `argparse` string."""
        if self.reward_mode == "naive":
            return

        try:
            self.reward_mode = float(self.reward_mode)
        except:
            raise Exception(f"{self.__class__}, reward_mode does not match specified type.")
