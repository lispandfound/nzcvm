"""High-level Python wrappers around the compiled Rust velocity-model backend.

The primary public interfaces are :class:`MeshModel` (a single tetrahedral
mesh) and :class:`ModelTree` (a priority-ordered collection of meshes with
alpha-blended queries).  Both implement the :class:`QueryableModel` protocol,
which requires a :meth:`~QueryableModel.query` method.

:class:`Quality` and the other dataclasses mirror their Rust counterparts
and are returned from query methods.

See Also
--------
nzcvm.layers : Pipeline layers for coordinate transforms and model queries.
nzcvm.mesh : Mesh I/O utilities used by :meth:`ModelTree.load_models`.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Self

import numpy as np
import pyvista as pv
import rich
import xarray as xr
from mashumaro.mixins.dict import DataClassDictMixin
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm import nzcvm  # ty: ignore[unresolved-import]
from nzcvm.mesh import read_vtkhdf

from .nzcvm import PyModelTree  # ty: ignore[unresolved-import]

MB = 1 / (1024 * 1024)


class ModelRange(Enum):
    """Priority ranges for bounded velocity-model queries.

    Priority values are ``u8`` ordered so that ``0`` is the highest priority
    and ``255`` is the lowest.  The ranges below reflect the NZCVM convention:

    * ``0–127``  — tomography models (higher priority, evaluated first).
    * ``128–255`` — basin models (lower priority, blended in afterwards).

    Parameters
    ----------
    value :
        A ``(priority_lo, priority_hi)`` tuple (both inclusive) passed to
        :meth:`~nzcvm.model.ModelTree.query_bounded`.

    Examples
    --------
    >>> ModelRange.TOMOGRAPHY.value
    (0, 127)
    >>> ModelRange.BASINS.value
    (129, 255)
    >>> ModelRange.ALL.value
    (0, 255)
    """

    TOMOGRAPHY = (0, 127)
    BASINS = (129, 255)
    ALL = (0, 255)


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
    :meth:`ModelTree.query_stats`.

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

    Returned by :meth:`ModelTree.get_explanation`. Each element in
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


class QueryableModel(Protocol):
    """Protocol satisfied by both :class:`MeshModel` and :class:`ModelTree`.

    Any object exposing a ``query`` method with this signature can be used
    wherever a queryable velocity model is expected — for example, as the
    argument to :class:`~nzcvm.layers.query.ModelLayer`.
    """

    def query(self, x: Any, y: Any, z: Any) -> Quality | None:
        """Return the blended quality at ``(x, y, z)``, or ``None`` outside coverage."""
        ...

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box as ``(min_xyz, max_xyz)`` float32 arrays."""
        ...


class MeshModel:
    """A single tetrahedral mesh velocity model.

    Wraps a compiled Rust :class:`PyMeshModel` and exposes spatial quality
    queries together with a rich display interface.

    Notes
    -----
    A ``MeshModel`` becomes *consumed* once it has been passed to
    :class:`ModelTree`.  Calling :meth:`query` or :attr:`aabb` on a consumed
    instance raises :class:`ValueError`.

    See Also
    --------
    ModelTree : Combines multiple ``MeshModel`` instances with priority blending.
    nzcvm.mesh.make_mesh : Build a compatible ``UnstructuredGrid``.
    """

    def __init__(self, raw: Any) -> None:
        """
        Parameters
        ----------
        raw :
            Compiled Rust ``PyMeshModel`` object returned by
            :func:`nzcvm.nzcvm.mesh_model`.
        """
        self._raw = raw

    @classmethod
    def from_mesh(
        cls,
        mesh: pv.UnstructuredGrid,
        name: str | None = None,
    ) -> "MeshModel":
        """Build a :class:`MeshModel` from a PyVista ``UnstructuredGrid``.

        Parameters
        ----------
        mesh :
            An ``UnstructuredGrid`` with the NZCVM cell and field data
            layout (see :func:`nzcvm.mesh.make_mesh`).
        name :
            Optional human-readable name.  Takes precedence over any name
            stored in ``mesh.field_data["name"]``.

        Returns
        -------
        MeshModel

        See Also
        --------
        nzcvm.mesh.make_mesh : Create a compatible ``UnstructuredGrid``.
        ModelTree : Combine multiple ``MeshModel`` instances for priority-blended queries.
        """
        return cls(_mesh_model_from_pyvista(mesh, name=name))

    @property
    def name(self) -> str:
        """Human-readable name assigned at construction time."""
        return self._raw.name  # type: ignore[no-any-return]

    @property
    def priority(self) -> int:
        """Model priority — lower number = higher priority in a :class:`ModelTree`."""
        return self._raw.priority  # type: ignore[no-any-return]

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box of this mesh.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            A pair ``(min_xyz, max_xyz)`` of shape-``(3,)`` float32 arrays.
        """
        return self._raw.aabb()  # type: ignore[no-any-return]

    def query(self, x: Any, y: Any, z: Any) -> Quality | None:
        """Query material properties at a single point.

        Parameters
        ----------
        x, y, z :
            Coordinates in the model's projected CRS (metres).

        Returns
        -------
        Quality or None
            Quality at ``(x, y, z)``, or ``None`` if outside this mesh.

        Raises
        ------
        ValueError
            If this ``MeshModel`` has been consumed by a :class:`ModelTree`.
        """
        quality_dict = self._raw.query(x, y, z)
        return Quality.from_dict(quality_dict) if quality_dict is not None else None

    def view(self) -> Tree:
        """Return a :class:`rich.tree.Tree` summary of this mesh model."""
        data = self._raw.view()
        label = data.get("name") or f"MeshModel {data['id']}"
        tree = Tree(f"[bold]MeshModel[/bold] [cyan]{label!r}[/cyan]")
        tree.add(f"Priority: {data['priority']}")
        size_mb = round(data["size"] * MB)
        if size_mb > 1024:
            tree.add(f"[red]Size: {size_mb:,} MB[/red]")
        else:
            tree.add(f"Size: {size_mb:,} MB")
        b = data["bounds"]
        tree.add(
            f"Bounds: [X: {b[0]:.0f}–{b[3]:.0f}, "
            f"Y: {b[1]:.0f}–{b[4]:.0f}, "
            f"Z: {b[2]:.0f}–{b[5]:.0f}]"
        )
        transform_str = "None" if data["transform"] is None else "Active"
        tree.add(f"Transform: {transform_str}")
        return tree

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render this mesh model as a rich tree for ``rich.print``."""
        yield self.view()


class ModelTree:
    """A velocity model backed by a Rust BVH tree of tetrahedral meshes.

    Wraps one or more :class:`MeshModel` instances (or VTKHDF mesh files)
    into a priority-ordered spatial index.  Queries return blended
    :class:`Quality` values at arbitrary 3-D coordinates.

    Notes
    -----
    Lower priority numbers take precedence. When multiple models cover the
    same point their qualities are alpha-composited until the cumulative
    alpha reaches 1.0.

    See Also
    --------
    ModelTree.load_models : Load from VTKHDF files or a directory.
    ModelTree.from_mesh : Build from an in-memory PyVista mesh.
    ModelTree.query : Single-point quality query.
    ModelTree.query_many : Vectorised multi-point query returning an xarray Dataset.
    """

    def __init__(
        self,
        internal: PyModelTree | list[MeshModel],
        model_map: dict | None = None,
    ):
        """
        Parameters
        ----------
        internal :
            Either a compiled Rust ``PyModelTree`` object (legacy path) or a
            list of :class:`MeshModel` instances that will be combined into a
            tree.
        model_map :
            Optional mapping from integer model index to a human-readable
            name, used as a fallback when a model has no embedded name.
        """
        self._raw: PyModelTree
        if isinstance(internal, list):
            raw_list = [m._raw for m in internal]
            self._raw = nzcvm.model_tree(raw_list)
        else:
            self._raw = internal
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
        ModelTree

        Examples
        --------
        Load all mesh files in a directory (requires data files to exist):

        >>> from pathlib import Path
        >>> ModelTree.load_models(Path("/path/to/models"))  # doctest: +SKIP
        """
        if len(models) == 1 and Path(models[0]).is_dir():
            mesh_paths = list(Path(models[0]).glob("*.vtkhdf"))
        else:
            mesh_paths = [Path(p) for p in models]

        mesh_models = [
            _mesh_model_from_pyvista(read_vtkhdf(p), name=p.stem)
            for p in mesh_paths
        ]
        raw = nzcvm.model_tree(mesh_models)
        return cls(raw)

    @classmethod
    def from_mesh(
        cls, mesh_model: pv.UnstructuredGrid, model_map: dict | None = None
    ) -> Self:
        """Build a :class:`ModelTree` from a single in-memory PyVista mesh.

        Parameters
        ----------
        mesh_model :
            An ``UnstructuredGrid`` with the NZCVM cell and field data
            layout (see :func:`nzcvm.mesh.make_mesh`).
        model_map :
            Optional mapping from integer model ID to a display name.

        Returns
        -------
        ModelTree

        See Also
        --------
        ModelTree.load_models : Load from VTKHDF files on disk.
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
        ModelTree.query_many : Vectorised query for arrays of coordinates.
        ModelTree.query_stats : Query with BVH traversal diagnostics.
        ModelTree.get_explanation : Query with per-model contribution details.
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
        ModelTree.query : Query without diagnostics.
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
        ModelTree.explain : Pretty-print the explanation to the terminal.
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
        ModelTree.get_explanation : Return the explanation as a Python object.
        """
        explanation = self.get_explanation(x, y, z)
        rich.print(explanation)
        if len(explanation.contributions) > 1:
            rich.print(
                "Qualities alpha-blended together until exhaustion "
                "or combined quality alpha ~ 1.0"
            )

    def query_bounded(
        self,
        x: Any,
        y: Any,
        z: Any,
        model_range: ModelRange = ModelRange.ALL,
    ) -> Quality | None:
        """Query material properties at a single point within a priority range.

        Parameters
        ----------
        x, y, z :
            Coordinates in the model's projected CRS (metres).
        model_range :
            Restricts the query to models whose priority falls within this
            range.  Defaults to :attr:`ModelRange.ALL`.

        Returns
        -------
        Quality or None
            Blended quality from models in the given range, or ``None`` if no
            matching model covers the point.

        See Also
        --------
        ModelRange : Priority range constants.
        ModelTree.query : Query across all priorities.
        ModelTree.query_many_bounded : Vectorised bounded query.
        """
        lo, hi = model_range.value
        quality_dict = self._raw.query_bounded(x, y, z, lo, hi)
        return Quality.from_dict(quality_dict) if quality_dict is not None else None

    def query_many_raw_bounded(
        self, x: Any, y: Any, z: Any, model_range: ModelRange = ModelRange.ALL
    ) -> np.ndarray:
        """Vectorised bounded query returning a raw float32 array.

        Parameters
        ----------
        x, y, z :
            Arrays of coordinates; must be broadcastable to the same shape.
        model_range :
            Restricts the query to models whose priority falls within this
            range.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(*x.shape, 6)`` with columns ordered
            as ``[rho, vp, vs, qp, qs, alpha]``.  Points not covered by any
            matching model are returned as zeros.

        See Also
        --------
        ModelTree.query_many_bounded : Same query returning a labelled Dataset.
        """
        lo, hi = model_range.value
        return self._raw.query_many_bounded(
            x.astype(np.float32, copy=False).ravel(),
            y.astype(np.float32, copy=False).ravel(),
            z.astype(np.float32, copy=False).ravel(),
            lo,
            hi,
        ).reshape(x.shape + (6,))

    def query_many_bounded(
        self, x: Any, y: Any, z: Any, model_range: ModelRange = ModelRange.ALL
    ) -> xr.Dataset:
        """Vectorised bounded query returning a labelled :class:`xarray.Dataset`.

        Parameters
        ----------
        x, y, z :
            Arrays of coordinates; broadcastable to a common shape.
        model_range :
            Restricts the query to models whose priority falls within this
            range.

        Returns
        -------
        xarray.Dataset
            Dataset with variables ``rho``, ``vp``, ``vs``, ``qp``, ``qs``
            and coordinates ``x``, ``y``, ``z``.  Points not covered by any
            matching model are returned as zeros.

        See Also
        --------
        ModelRange : Priority range constants.
        ModelTree.query_bounded : Single-point bounded query.
        """
        x, y, z = np.broadcast_arrays(x, y, z)
        ndim = x.ndim
        dims = tuple(f"d{i}" for i in range(ndim))
        quality_array = self.query_many_raw_bounded(x, y, z, model_range)
        var_names = ["rho", "vp", "vs", "qp", "qs"]
        data_vars = {
            name: (dims, quality_array[..., i]) for i, name in enumerate(var_names)
        }
        return xr.Dataset(
            data_vars=data_vars,
            coords={"x": (dims, x), "y": (dims, y), "z": (dims, z)},
        )

    def query_many_raw_into(self, existing: np.ndarray, x: Any, y: Any, z: Any) -> np.ndarray:
        """Vectorised query that alpha-blends into an existing quality buffer.

        Parameters
        ----------
        existing :
            Float32 array of shape ``(*x.shape, 6)`` containing the current
            quality values (same layout as returned by
            :meth:`query_many_raw`).  The array is **not** modified in-place;
            a new array with the blended result is returned.
        x, y, z :
            Arrays of coordinates; must be broadcastable to ``existing.shape[:-1]``.

        Returns
        -------
        numpy.ndarray
            Float32 array of the same shape as ``existing`` with each row
            blended with the new query result using the Porter-Duff "over"
            operator.

        See Also
        --------
        ModelTree.query_many_bounded_into : Bounded variant.
        """
        return self._raw.query_many_into(
            existing.astype(np.float32, copy=False).reshape(-1, 6),
            x.astype(np.float32, copy=False).ravel(),
            y.astype(np.float32, copy=False).ravel(),
            z.astype(np.float32, copy=False).ravel(),
        ).reshape(existing.shape)

    def query_many_raw_bounded_into(
        self,
        existing: np.ndarray,
        x: Any,
        y: Any,
        z: Any,
        model_range: ModelRange = ModelRange.ALL,
    ) -> np.ndarray:
        """Bounded vectorised query that alpha-blends into an existing buffer.

        Parameters
        ----------
        existing :
            Float32 array of shape ``(*x.shape, 6)`` containing the current
            quality values.
        x, y, z :
            Arrays of coordinates; must be broadcastable to ``existing.shape[:-1]``.
        model_range :
            Restricts the query to models whose priority falls within this
            range.

        Returns
        -------
        numpy.ndarray
            Float32 array with the same shape as ``existing``.

        See Also
        --------
        ModelRange : Priority range constants.
        ModelTree.query_many_raw_into : Unbounded variant.
        """
        lo, hi = model_range.value
        return self._raw.query_many_bounded_into(
            existing.astype(np.float32, copy=False).reshape(-1, 6),
            x.astype(np.float32, copy=False).ravel(),
            y.astype(np.float32, copy=False).ravel(),
            z.astype(np.float32, copy=False).ravel(),
            lo,
            hi,
        ).reshape(existing.shape)

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
        ModelTree.query_many : Same query returning a labelled xarray Dataset.
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
        ModelTree.query_many_raw : Same query as an unlabelled float32 array.
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
        """Return a :class:`rich.tree.Tree` representation of the model tree."""
        data = self._raw.view()

        total_size_mb = round(data["size"] * MB)
        tree = Tree(f"Model Tree (Total Size: {total_size_mb:,} MB)")

        for m in data["models"]:
            m_id = m["id"]
            embedded_name = m.get("name") or ""
            name = embedded_name or self.model_map.get(m_id, f"Model {m_id}")

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


#: Backward-compatible alias — ``Model`` was renamed to :class:`ModelTree`.
Model = ModelTree


def _mesh_model_from_pyvista(
    mesh_model: pv.UnstructuredGrid, name: str | None = None
) -> Any:
    """Build a PyMeshModel from a pyvista UnstructuredGrid.

    Priority is a model-level scalar stored in ``field_data["priority"]``.
    If *name* is not supplied the function falls back to
    ``field_data["name"]`` when present.
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

    # Resolve name: caller-supplied > field_data["name"] > None
    if name is None and "name" in mesh_model.field_data:
        raw_names = mesh_model.field_data["name"]
        if len(raw_names) > 0:
            name = str(raw_names[0])

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
        name,
    )
