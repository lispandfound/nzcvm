from enum import Enum


class ModelRange(Enum):
    """Priority ranges for bounded velocity-model queries.

    Priority values are ``u8`` ordered so that ``0`` is the highest priority
    and ``255`` is the lowest.  The ranges below reflect the NZCVM convention:

    * ``0–127``  — basin models (higher priority, evaluated first).
    * ``129–255`` — tomography models (lower priority, blended in afterwards).

    Priority 128 is intentionally excluded from both named ranges and may be
    used as a separator value by model authors.

    Parameters
    ----------
    value :
        A ``(priority_lo, priority_hi)`` tuple (both inclusive) passed to
        :meth:`~nzcvm.model.ModelTree.query_bounded`.
    """

    BASINS = (0, 127)
    TOMOGRAPHY = (128, 255)
    ALL = (0, 255)
