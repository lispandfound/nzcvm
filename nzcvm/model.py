"""High-level Python wrappers around the compiled Rust velocity-model backend.

The primary public interface is :class:`Model`, which loads one or more
tetrahedral mesh files and exposes spatial quality queries.
:class:`Quality` and the other dataclasses mirror their Rust counterparts
and are returned from query methods.

See Also
--------
nzcvm.layers : Pipeline layers for coordinate transforms and model queries.
nzcvm.mesh : Mesh I/O utilities used by :meth:`Model.load_models`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import pyvista as pv
import rich
import xarray as xr
from mashumaro.mixins.dict import DataClassDictMixin
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm import nzcvm  # ty: ignore[unresolved-import]
from nzcvm.mesh import read_vtkhdf

from .nzcvm import PyModel  # ty: ignore[unresolved-import]

MB = 1 / (1024 * 1024)


@dataclass
class Quality(DataClassDictMixin):
    """Seismic material properties at a single point in the velocity model.

    Parameters
    ----------
    rho :
        Density in kg m⁻³.
    vp :
        P-wave velocity in m s⁻¹.
    vs :
        S-wave velocity in m s⁻¹.
    qp :
        P-wave quality factor (attenuation).
    qs :
        S-wave quality factor (attenuation).
    alpha :
        Opacity weight in [0, 1] used for alpha blending when multiple
        models overlap.  A value of 1.0 means the model is fully opaque
        and no lower-priority models contribute.

    Examples
    --------
    >>> q = Quality(rho=2700.0, vp=6000.0, vs=3500.0, qp=200.0, qs=100.0, alpha=1.0)
    >>> str(q)
    '(ρ=2700.00, Vp=6000.00, Vs=3500.00, Qp=200.00, Qs=100.00, ɑ=1.00)'
    """
    rho: float
    vp: float
    vs: float
    qp: float
    qs: float
    alpha: float

    def __str__(self):
        """Return a compact string like ``(ρ=…, Vp=…, Vs=…, Qp=…, Qs=…, ɑ=…)``."""
        return (
            f"(ρ={self.rho:.2f}, Vp={self.vp:.2f}, Vs={self.vs:.2f},"
            f" Qp={self.qp:.2f}, Qs={self.qs:.2f}, ɑ={self.alpha:.2f})"
        )


@dataclass
class Point(DataClassDictMixin):
    """A 3-D point returned by some query methods.

    Examples
    --------
    >>> p = Point(x=1.5, y=2.5, z=-100.0)
    >>> str(p)
    '(1.5, 2.5, -100)'
    """
    x: float
    y: float
    z: float

    def __str__(self) -> str:
        """Return ``(x, y, z)`` formatted to six significant figures."""
        return f"({self.x:.6g}, {self.y:.6g}, {self.z:.6g})"


@dataclass
class QueryStats(DataClassDictMixin):
    """Diagnostic counters for a single model query.

    Useful for profiling BVH traversal efficiency. Returned by
    :meth:`Model.query_stats`.

    Parameters
    ----------
    aabb_tests :
        Number of axis-aligned bounding-box intersection tests performed.
    simplex_tests :
        Number of simplex (tetrahedron) containment tests performed.
    hit_count :
        Number of simplices that contained the query point.
    output :
        Final blended quality, or ``None`` if the point is outside the model.
    elapsed :
        Wall-clock time for the query in nanoseconds.
    """
    aabb_tests: int
    simplex_tests: int
    hit_count: int
    output: Quality | None
    elapsed: int


@dataclass
class ModelContribution(DataClassDictMixin):
    """A single model's contribution to a blended quality result.

    Parameters
    ----------
    priority :
        Integer priority of this model (lower number = higher priority).
    quality :
        Raw (un-blended) quality returned by this model for the query point.
    """
    priority: int
    quality: Quality

    def __str__(self) -> str:
        """Return ``priority=<n>, quality=<Quality>``."""
        return f"priority={self.priority}, quality={str(self.quality)}"


@dataclass
class Explanation(DataClassDictMixin):
    """Full audit trail for how a query result was produced.

    Returned by :meth:`Model.get_explanation`. Each element in
    ``contributions`` shows the raw quality from one model; ``output`` is
    the final blended result.

    Parameters
    ----------
    contributions :
        Per-model contributions in priority order.
    output :
        Final blended quality, or ``None`` if no model covered the point.
    termination :
        Index into ``contributions`` at which alpha saturation was reached.
        All contributions at or after this index were ignored.

    Notes
    -----
    If ``termination`` is ``None`` all contributions were used.
    """
    contributions: list[ModelContribution]
    output: Quality | None
    termination: int | None

    def __rich__(self) -> Tree:
        """Return a :class:`rich.tree.Tree` showing per-model contributions."""
        if not self.output:
            return Tree("[red]No model coverage for query point.[/red]")

        root = Tree(f"[bold white]{self.output}[/bold white]")

        for i, contribution in enumerate(self.contributions):
            is_active = self.termination is None or i < self.termination
            colour = "green" if is_active else "red"
            node = root.add(
                f"[{colour}]Model {i} (priority = {contribution.priority})[/{colour}]"
            )
            node.add(f"Quality: {contribution.quality}")

        return root


class Model:
    """A velocity model backed by a Rust BVH tree of tetrahedral meshes.

    Wraps one or more VTKHDF mesh files into a priority-ordered spatial
    index. Queries return blended :class:`Quality` values at arbitrary
    3-D coordinates.

    Notes
    -----
    Lower priority numbers take precedence. When multiple models cover the
    same point their qualities are alpha-composited until the cumulative
    alpha reaches 1.0.

    See Also
    --------
    Model.load_models : Load from VTKHDF files or a directory.
    Model.from_mesh : Build from an in-memory PyVista mesh.
    Model.query : Single-point quality query.
    Model.query_many : Vectorised multi-point query returning an xarray Dataset.
    """

    def __init__(self, internal_py_model: PyModel, model_map: dict | None = None):
        """
        Parameters
        ----------
        internal_py_model :
            Compiled Rust ``PyModel`` object.
        model_map :
            Optional mapping from integer model index to a human-readable name,
            used when displaying the model tree.
        """
        self._raw = internal_py_model
        self.model_map = model_map or {}

    @classmethod
    def load_models(cls, *models: Path | str) -> Self:
        """Load a velocity model from one or more VTKHDF files or a directory.

        Parameters
        ----------
        *models :
            Paths to individual ``.vtkhdf`` files, or a single directory
            path.  When a directory is given every ``*.vtkhdf`` file it
            contains is loaded.

        Returns
        -------
        Model

        Examples
        --------
        Load all mesh files in a directory (requires data files to exist):

        >>> from pathlib import Path
        >>> Model.load_models(Path("/path/to/models"))  # doctest: +SKIP
        """
        if len(models) == 1 and Path(models[0]).is_dir():
            mesh_paths = list(Path(models[0]).glob("*.vtkhdf"))
        else:
            mesh_paths = [Path(p) for p in models]

        mesh_models = [_mesh_model_from_pyvista(read_vtkhdf(p)) for p in mesh_paths]
        model_map = {i: p.stem for i, p in enumerate(mesh_paths)}
        raw = nzcvm.model_tree(mesh_models)
        return cls(raw, model_map)

    @classmethod
    def from_mesh(
        cls, mesh_model: pv.UnstructuredGrid, model_map: dict | None = None
    ) -> Self:
        """Build a :class:`Model` from a single in-memory PyVista mesh.

        Parameters
        ----------
        mesh_model :
            An ``UnstructuredGrid`` with the NZCVM cell and field data
            layout (see :func:`nzcvm.mesh.make_mesh`).
        model_map :
            Optional mapping from integer model ID to a display name.

        Returns
        -------
        Model

        See Also
        --------
        Model.load_models : Load from VTKHDF files on disk.
        nzcvm.mesh.make_mesh : Create a compatible ``UnstructuredGrid``.
        """
        raw_mesh_model = _mesh_model_from_pyvista(mesh_model)
        raw = nzcvm.model_tree([raw_mesh_model])
        return cls(raw, model_map or {})

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box of all meshes in the model.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            A pair ``(min_xyz, max_xyz)`` of shape-``(3,)`` float32 arrays
            in the model's coordinate system.
        """
        return self._raw.aabb()

    def query(self, x: Any, y: Any, z: Any) -> Quality | None:
        """Query material properties at a single point.

        Parameters
        ----------
        x, y, z :
            Coordinates in the model's projected CRS (metres).

        Returns
        -------
        Quality or None
            Blended quality, or ``None`` if the point lies outside all
            mesh models.

        See Also
        --------
        Model.query_many : Vectorised query for arrays of coordinates.
        Model.query_stats : Query with BVH traversal diagnostics.
        Model.get_explanation : Query with per-model contribution details.
        """
        quality_dict = self._raw.query(x, y, z)
        return Quality.from_dict(quality_dict) if quality_dict is not None else None

    def query_stats(self, x: Any, y: Any, z: Any) -> QueryStats:
        """Query a single point and return traversal diagnostics.

        Parameters
        ----------
        x, y, z :
            Coordinates in the model's projected CRS (metres).

        Returns
        -------
        QueryStats

        See Also
        --------
        Model.query : Query without diagnostics.
        """
        return QueryStats.from_dict(self._raw.query_stats(x, y, z))

    def get_explanation(self, x: Any, y: Any, z: Any) -> Explanation:
        """Return a full :class:`Explanation` for a single-point query.

        Parameters
        ----------
        x, y, z :
            Coordinates in the model's projected CRS (metres).

        Returns
        -------
        Explanation

        See Also
        --------
        Model.explain : Pretty-print the explanation to the terminal.
        """
        return Explanation.from_dict(self._raw.explain(x, y, z))

    def explain(self, x: float, y: float, z: float) -> None:
        """Pretty-print the blending explanation for a query point.

        Prints a rich-formatted tree to stdout showing each model's
        contribution and whether it was included in the final blend.

        Parameters
        ----------
        x, y, z :
            Coordinates in the model's projected CRS (metres).

        See Also
        --------
        Model.get_explanation : Return the explanation as a Python object.
        """
        explanation = self.get_explanation(x, y, z)
        rich.print(explanation)
        if len(explanation.contributions) > 1:
            rich.print(
                "Qualities alpha-blended together until exhaustion "
                "or combined quality alpha ~ 1.0"
            )

    def query_many_raw(self, x: Any, y: Any, z: Any) -> np.ndarray:
        """Vectorised query returning a raw float32 array.

        Parameters
        ----------
        x, y, z :
            Arrays of coordinates; must be broadcastable to the same shape.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(*x.shape, 6)`` with columns ordered
            as ``[rho, vp, vs, qp, qs, alpha]``.

        See Also
        --------
        Model.query_many : Same query returning a labelled xarray Dataset.
        """
        return self._raw.query_many(
            x.astype(np.float32, copy=False).ravel(),
            y.astype(np.float32, copy=False).ravel(),
            z.astype(np.float32, copy=False).ravel(),
        ).reshape(x.shape + (6,))

    def query_many(self, x: Any, y: Any, z: Any) -> xr.Dataset:
        """Vectorised query returning a labelled :class:`xarray.Dataset`.

        Parameters
        ----------
        x, y, z :
            Arrays of coordinates; broadcastable to a common shape.

        Returns
        -------
        xarray.Dataset
            Dataset with variables ``rho``, ``vp``, ``vs``, ``qp``, ``qs``
            and coordinates ``x``, ``y``, ``z``.

        See Also
        --------
        Model.query_many_raw : Same query as an unlabelled float32 array.
        """
        x, y, z = np.broadcast_arrays(x, y, z)

        ndim = x.ndim
        dims = tuple(f"d{i}" for i in range(ndim))
        quality_array = self.query_many_raw(x, y, z)
        var_names = ["rho", "vp", "vs", "qp", "qs"]
        data_vars = {}
        for i, name in enumerate(var_names):
            data_vars[name] = (dims, quality_array[..., i])

        dset = xr.Dataset(
            data_vars=data_vars,
            coords={
                "x": (dims, x),
                "y": (dims, y),
                "z": (dims, z),
            },
        )

        return dset

    def view(self) -> Tree:
        """Generates a rich.tree.Tree representation of the model structure."""
        data = self._raw.view()

        total_size_mb = round(data["size"] * MB)
        tree = Tree(f"Model Tree (Total Size: {total_size_mb:,} MB)")

        for m in data["models"]:
            m_id = m["id"]
            name = self.model_map.get(m_id, f"Model {m_id}")

            branch = tree.add(f"{name} (ID: {m_id})")
            size_mb = round(m["size"] * MB)
            branch.add(f"Priority: {m['priority']}")

            if size_mb > 1024:
                branch.add(f"[red]Size: {size_mb:,} MB[/red]")
            else:
                branch.add(f"Size: {size_mb:,} MB")

            b = m["bounds"]
            branch.add(
                f"Bounds: [X: {b[0]:.0f}-{b[3]:.0f}, Y: {b[1]:.0f}-{b[4]:.0f}, Z: {b[2]:.0f}-{b[5]:.0f}]"
            )

            transform_str = "None" if m["transform"] is None else "Active"
            branch.add(f"Transform: {transform_str}")

        return tree

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render the model tree as a rich tree for ``rich.print``."""
        yield self.view()


def _mesh_model_from_pyvista(mesh_model: pv.UnstructuredGrid):
    """Build a PyMeshModel from a pyvista UnstructuredGrid.

    Priority is a model-level scalar stored in ``field_data["priority"]``.
    """
    # Extract connectivity and wrap in asarray
    connectivity = np.asarray(mesh_model.cells_dict[np.uint8(pv.CellType.TETRA)])

    # Convert cell_data attributes
    types = np.asarray(mesh_model.cell_data["model_type"])
    model_idx = np.asarray(mesh_model.cell_data["models"]).ravel().astype(np.uint64)

    # Convert field_data attributes
    rho = np.asarray(mesh_model.field_data["rho"])
    vp = np.asarray(mesh_model.field_data["vp"])
    vs = np.asarray(mesh_model.field_data["vs"])
    qp = np.asarray(mesh_model.field_data["qp"])
    qs = np.asarray(mesh_model.field_data["qs"])
    alpha = np.asarray(mesh_model.field_data["alpha"])

    # Stack into qualities matrix
    qualities = np.c_[rho, vp, vs, qp, qs, alpha]

    # Handle priority scalar
    if "priority" not in mesh_model.field_data:
        priority = np.uint8(255)
    else:
        # field_data is often stored as a single-element array in PyVista
        priority = np.uint8(np.asarray(mesh_model.field_data["priority"])[0])

    # Handle transform if it exists
    transform = mesh_model.field_data.get("transform")
    if transform is not None:
        transform = np.asarray(transform)

    return nzcvm.mesh_model(
        np.asarray(mesh_model.points).astype(np.float32),
        connectivity.astype(np.uint64),
        types,
        model_idx,
        qualities.astype(np.float32),
        priority,
        transform,
    )
