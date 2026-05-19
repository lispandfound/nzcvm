from dataclasses import dataclass
from pathlib import Path
from .core import LayerConfig


@dataclass
class ElyLayerConfig(LayerConfig):
    """Configuration DTO for an :class:`~nzcvm.layers.ely.ElyTaperLayer`.

    Parameters
    ----------
    vs30 :
        Path to the Vs30 surface file.
    depth_t :
        Taper depth in metres (default ``450.0``).

    Examples
    --------
    TOML::

        [[layers]]
        type = "ely"
        vs30 = "path/to/vs30.h5"
        z_t = 450.0
    """

    vs30: Path
    depth_t: float = 450.0
    type: str = "ely"
