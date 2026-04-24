from enum import StrEnum, auto


class Component(StrEnum):
    RHO = auto()
    VP = auto()
    VS = auto()
    QP = auto()
    QS = auto()
    ALPHA = auto()


class Coordinate(StrEnum):
    X = auto()
    Y = auto()
    Z = auto()
    I = auto()  # noqa: E741
    J = auto()
    K = auto()
    COMPONENT = auto()
