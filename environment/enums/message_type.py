from enum import IntEnum, auto


class MessageType(IntEnum):
    EPISODE_STARTS = 0
    EPISODE_CONTINUES = auto()
    EPISODE_ENDS = auto()
