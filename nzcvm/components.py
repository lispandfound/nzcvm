from enum import StrEnum, auto()

class Component(StrEnum):
    RHO = auto()
    VP = auto()
    VS = auto()
    QP = auto()
    QS = auto()
    Z = auto()
    ALPHA = auto()


class Coordinate(StrEnum):
    X = auto()
    Y = auto()
    Z = auto()
    COMPONENT = auto()
