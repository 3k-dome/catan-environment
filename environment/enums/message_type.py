from enum import Enum, auto


class MessageType(Enum):
    EPISODE_STARTS = 0
    EPISODE_CONTINUES = auto()
    EPISODE_ENDS = auto()
