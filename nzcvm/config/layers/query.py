from dataclasses import dataclass
from .core import LayerConfig
from pathlib import Path


@dataclass
class QueryLayerConfig(LayerConfig):
    """Configuration DTO for a :class:`~nzcvm.layers.query.ModelLayer`.

    Specifies where to find the velocity-model mesh files.  *model_path*
    and *model_glob* together identify the set of ``*.vtkhdf`` files to load.

    Parameters
    ----------
    model_path : Path | None
        Directory containing the mesh files.
    model_glob : str
        Glob pattern used to find mesh files under *model_path*
        (default ``"*.vtkhdf"``).

    Examples
    --------
    TOML::

        [[layers]]
        type = "model"
        model_path = "path/to/models"
        model_glob = "*.vtkhdf"
    """

    model_path: Path
    model_glob: str = "*.vtkhdf"
    type: str = "query"
