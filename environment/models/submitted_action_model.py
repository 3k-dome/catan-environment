from dataclasses import dataclass

from environment.enums import PlayerNumber

from .base_model import BaseModel


@dataclass
class SubmittedActionModel(BaseModel):
    player_number: PlayerNumber
    index: int
