from dataclasses import dataclass

from nzcvm.config.validation import ExistingDir

from .core import LayerConfig


@dataclass
class QueryLayerConfig(LayerConfig):
    """Configuration DTO for a :class:`~nzcvm.layers.query.QueryLayer`.

    Specifies where to find the velocity-model mesh files.  *model_path*
    and *model_glob* together identify the set of ``*.vtkhdf`` files to load.

    Parameters
    ----------
    model_path :
        Directory containing the mesh files.
    model_glob :
        Glob pattern used to find mesh files under *model_path*
        (default ``"*.vtkhdf"``).

    Examples
    --------
    TOML::

        [[layers]]
        type = "query"
        model_path = "path/to/models"
        model_glob = "*.vtkhdf"
    """

    model_path: ExistingDir
    model_glob: str = "*.vtkhdf"
    type: str = "query"
