"""Physical-property components produced by the velocity model.

Examples
--------
>>> from nzcvm.components import Component
>>> str(Component.VP)
'vp'
"""

from enum import StrEnum, auto


class Component(StrEnum):
    """Seismic material property label used as a dataset variable name.

    Each member's string value is the variable name written into output
    datasets, so ``Component.VP == "vp"`` and it can be used directly as
    an xarray coordinate or dimension name.

    Examples
    --------
    >>> Component.RHO == "rho"
    True
    >>> list(Component)  # doctest: +NORMALIZE_WHITESPACE
    [<Component.RHO: 'rho'>, <Component.VP: 'vp'>, <Component.VS: 'vs'>, <Component.QP: 'qp'>, <Component.QS: 'qs'>, <Component.ALPHA: 'alpha'>]
    """
    RHO = auto()
    VP = auto()
    VS = auto()
    QP = auto()
    QS = auto()
    ALPHA = auto()


class Coordinate(StrEnum):
    """Grid coordinate axis labels used as dimension names in datasets.

    These labels are used as dimension names for xarray datasets. The
    spatial axes ``X``, ``Y``, ``Z`` hold projected coordinates after
    transformation; ``I``, ``J``, ``K`` are logical grid indices.

    Examples
    --------
    >>> Coordinate.I == "i"
    True
    """
    X = auto()
    Y = auto()
    Z = auto()
    I = auto()  # noqa: E741
    J = auto()
    K = auto()
    COMPONENT = auto()
