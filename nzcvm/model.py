from dataclasses import dataclass, fields
from pathlib import Path


import numpy as np
import xarray as xr

from typing import Self, Any, get_type_hints
from rich.tree import Tree
from rich.console import Console, ConsoleOptions, RenderResult
import rich


import pyvista as pv

from nzcvm import nzcvm  # ty: ignore[unresolved-import]
from nzcvm.mesh import read_vtkhdf

from .nzcvm import PyModel  # ty: ignore[unresolved-import]

MB = 1 / (1024 * 1024)


class RSBase:
    @classmethod
    def _from_rs(cls, rs_obj: Any) -> Self | None:
        """
        Automagically maps attributes from a Rust-backed object to this dataclass.
        Handles recursive deserialization for fields that inherit from RSBase.
        """
        if rs_obj is None:
            return None

        # Get type hints to handle the recursive case
        hints = get_type_hints(cls)
        init_kwargs = {}

        for field in fields(cls):
            field_name = field.name
            field_type = hints[field_name]

            val = getattr(rs_obj, field_name)

            try:
                if issubclass(field_type, RSBase):
                    init_kwargs[field_name] = field_type._from_rs(val)
                else:
                    init_kwargs[field_name] = val
            except TypeError:
                init_kwargs[field_name] = val

        return cls(**init_kwargs)


@dataclass
class Quality(RSBase):
    rho: float
    vp: float
    vs: float
    qp: float
    qs: float
    alpha: float

    def __str__(self):
        return (
            f"(ρ={self.rho:.2f}, Vp={self.vp:.2f}, Vs={self.vs:.2f},"
            f" Qp={self.qp:.2f}, Qs={self.qs:.2f}, ɑ={self.alpha:.2f})"
        )


@dataclass
class Point(RSBase):
    x: float
    y: float
    z: float

    def __str__(self) -> str:
        return f"({self.x:.6g}, {self.y:.6g}, {self.z:.6g})"


@dataclass
class QueryStats(RSBase):
    aabb_tests: int
    simplex_tests: int
    hit_count: int
    output: Quality | None
    elapsed: int


@dataclass
class ModelContribution(RSBase):
    priority: int
    quality: Quality

    def __str__(self) -> str:
        return f"priority={self.priority}, quality={str(self.quality)}"


@dataclass
class Explanation(RSBase):
    contributions: list[ModelContribution]
    output: Quality | None
    termination: int | None

    @classmethod
    def _from_rs(cls, rs_obj: Any) -> Self:
        return cls(
            contributions=[ModelContribution._from_rs(c) for c in rs_obj.contributions],  # ty: ignore[invalid-argument-type]
            output=Quality._from_rs(rs_obj.output),
            termination=rs_obj.termination,
        )

    def __rich__(self) -> Tree:
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
    """A high-level wrapper for the Rust ModelTree."""

    def __init__(self, internal_py_model: PyModel, model_map: dict | None = None):
        self._raw = internal_py_model
        self.model_map = model_map or {}

    @classmethod
    def load_models(cls, *models: Path | str) -> Self:
        if len(models) == 1 and Path(models[0]).is_dir():
            mesh_paths = list(Path(models[0]).glob("*.vtkhdf"))
        else:
            mesh_paths = [Path(p) for p in models]

        mesh_models = [
            _mesh_model_from_pyvista(read_vtkhdf(p)) for p in mesh_paths
        ]
        model_map = {i: p.stem for i, p in enumerate(mesh_paths)}
        raw = nzcvm.model_tree(mesh_models)
        return cls(raw, model_map)

    @classmethod
    def from_mesh(cls, mesh_model: pv.UnstructuredGrid, model_map: dict | None = None) -> Self:
        raw_mesh_model = _mesh_model_from_pyvista(mesh_model)
        raw = nzcvm.model_tree([raw_mesh_model])
        return cls(raw, model_map or {})

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        aabb = self._raw.aabb()
        min = np.array([aabb.min.x, aabb.min.y, aabb.min.z])
        max = np.array([aabb.max.x, aabb.max.y, aabb.max.z])
        return min, max

    def query(self, x, y, z) -> Quality | None:
        quality_rs = self._raw.query(x, y, z)
        return Quality._from_rs(quality_rs)

    def query_stats(self, x, y, z) -> QueryStats | None:
        quality_rs = self._raw.query_stats(x, y, z)
        return QueryStats._from_rs(quality_rs)

    def get_explanation(self, x, y, z) -> Explanation:
        explanation_rs = self._raw.explain(x, y, z)
        return Explanation._from_rs(explanation_rs)

    def explain(self, x: float, y: float, z: float) -> None:
        explanation = self.get_explanation(x, y, z)
        rich.print(explanation)
        if len(explanation.contributions) > 1:
            rich.print(
                "Qualities alpha-blended together until exhaustion "
                "or combined quality alpha ~ 1.0"
            )

    def query_many_raw(self, x, y, z) -> np.ndarray:
        return self._raw.query_many(
            x.astype(np.float32, copy=False).ravel(),
            y.astype(np.float32, copy=False).ravel(),
            z.astype(np.float32, copy=False).ravel(),
        ).reshape(x.shape + (6,))

    def query_many(self, x, y, z) -> xr.Dataset:
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
        """Allows direct usage of rich.print(model)"""
        yield self.view()


def _mesh_model_from_pyvista(mesh_model: pv.UnstructuredGrid):
    """Build a PyMeshModel from a pyvista UnstructuredGrid.

    Priority is a model-level scalar stored in ``field_data["priority"]``.
    """
    connectivity = mesh_model.cells_dict[int(pv.CellType.TETRA)]  # ty: ignore[invalid-argument-type]
    types = mesh_model.cell_data["model_type"]
    model_idx = mesh_model.cell_data["models"]
    rho = mesh_model.field_data["rho"]
    vp = mesh_model.field_data["vp"]
    vs = mesh_model.field_data["vs"]
    qp = mesh_model.field_data["qp"]
    qs = mesh_model.field_data["qs"]
    alpha = mesh_model.field_data["alpha"]

    qualities = np.c_[rho, vp, vs, qp, qs, alpha]
    priority = np.uint8(mesh_model.field_data["priority"][0])
    transform = mesh_model.field_data.get("transform")
    return nzcvm.mesh_model(
        mesh_model.points.astype(np.float32),
        connectivity.astype(np.uint64),
        types,
        model_idx,
        qualities.astype(np.float32),
        priority,
        transform,
    )
