from enum import IntEnum, auto


class Phase(IntEnum):
    FoundingFirstPass = 0
    FoundingSecondPass = auto()
    RobberDiscard = auto()
    RobberPlace = auto()
    RobberSteal = auto()
    Trading = auto()
    Building = auto()
