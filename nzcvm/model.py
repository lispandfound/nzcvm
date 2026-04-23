from nzcvm.components import Component
from nzcvm.coordinates import Coordinate
from dataclasses import dataclass, fields
from pathlib import Path


import numpy as np
import xarray as xr
import xoak
from typing import Self, Any, get_type_hints
from rich.tree import Tree
import rich


from nzcvm import nzcvm, mesh

from .nzcvm import PyModel


class RSBase:
    @classmethod
    def _from_rs(cls, rs_obj: Any) -> Self:
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
class Simplex(RSBase):
    c0: Point
    c1: Point
    c2: Point
    c3: Point
    priority: int

    def __str__(self) -> str:
        c0 = str(self.c0)
        c1 = str(self.c1)
        c2 = str(self.c2)
        c3 = str(self.c3)
        return f"Tetrahedron with corners:\nc0={c0}\nc1={c1}\nc2={c2}\nc3={c3}"


@dataclass
class ConstantModel(RSBase):
    quality: Quality

    def __str__(self) -> str:
        return f"Constant model with quality = {str(self.quality)}"


@dataclass
class InterpolatedModel(RSBase):
    x: Quality
    y: Quality
    z: Quality
    w: Quality

    def __str__(self) -> str:
        return (
            f"Barycentric interpolation between:\n"
            f"x={str(self.x)}\ny={str(self.y)}\nz={str(self.z)}\nw={str(self.w)}"
        )


@dataclass
class QueryStats(RSBase):
    aabb_tests: int
    simplex_tests: int
    hit_count: int
    output: Quality | None
    elapsed: int


SimplexModel = ConstantModel | InterpolatedModel


def _from_rs_simplex_model(model: Any) -> SimplexModel:
    try:
        return ConstantModel._from_rs(model)
    except AttributeError:
        return InterpolatedModel._from_rs(model)


@dataclass
class Explanation(RSBase):
    simplices: list[Simplex]
    qualities: list[Quality]
    models: list[SimplexModel]
    output: Quality | None
    termination: int | None

    @classmethod
    def _from_rs(cls, rs_obj: Any) -> Self:
        return cls(
            simplices=[Simplex._from_rs(simplex) for simplex in rs_obj.simplices],
            qualities=[Quality._from_rs(quality) for quality in rs_obj.qualities],
            models=[_from_rs_simplex_model(model) for model in rs_obj.models],
            output=Quality._from_rs(rs_obj.output),
            termination=rs_obj.termination,
        )

    def __rich__(self) -> Tree:
        if not self.output:
            return Tree("[red]No model coverage for query point.[/red]")

        root = Tree(f"[bold white]{self.output}[/bold white]")

        for i, (simplex, model, quality) in enumerate(
            zip(self.simplices, self.models, self.qualities)
        ):
            is_active = self.termination is None or i < self.termination
            colour = "green" if is_active else "red"

            simplex_node = root.add(
                f"[{colour}]Simplex {i} (priority = {simplex.priority})[/{colour}]"
            )

            simplex_node.add(f"Model: {model}")
            simplex_node.add(f"Simplex quality: {quality}")
            simplex_node.add(f"Geometry: {simplex}")

        return root


class Model:
    """A high-level wrapper for the Rust ModelTree."""

    def __init__(self, internal_py_model: PyModel):
        self._raw = internal_py_model

    @classmethod
    def load_models(cls, *models: Path | str) -> Self:
        if len(models) == 1 and Path(models[0]).is_dir():
            meshes = [
                mesh.Mesh.read_vtkhdf(mesh_path)
                for mesh_path in Path(models[0]).glob("*.vtkhdf")
            ]
        else:
            meshes = [mesh.Mesh.read_vtkhdf(mesh_path) for mesh_path in models]
        all = mesh.Mesh.union(*meshes)
        return cls.from_mesh(all)

    @classmethod
    def from_mesh(cls, mesh_model: mesh.Mesh):
        # Determine the model type from the available data
        # Point data => tomography model
        # Cell data => Basin model
        # TODO: Map this explicitly to field data in the vtkhdf

        types = mesh_model.cell_data["model_type"]
        model_idx = mesh_model.cell_data["models"]
        rho = mesh_model.field_data["rho"]
        vp = mesh_model.field_data["vp"]
        vs = mesh_model.field_data["vs"]
        qp = mesh_model.field_data["qp"]
        qs = mesh_model.field_data["qs"]
        alpha = mesh_model.field_data["alpha"]

        qualities = np.c_[rho, vp, vs, qp, qs, alpha]
        raw = nzcvm.mesh_model(
            mesh_model.points.astype(np.float32),
            mesh_model.connectivity.astype(np.uint64),
            types,
            model_idx,
            qualities.astype(np.float32),
            mesh_model.cell_data["priority"],
        )
        return cls(raw)

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        aabb = self._raw.aabb()
        min = np.array([aabb.min.x, aabb.min.y, aabb.min.z])
        max = np.array([aabb.max.x, aabb.max.y, aabb.max.z])
        return min, max

    def query(self, x, y, z) -> Quality:
        quality_rs = self._raw.query(x, y, z)
        return Quality._from_rs(quality_rs)

    def query_stats(self, x, y, z) -> QueryStats:
        quality_rs = self._raw.query_stats(x, y, z)
        return QueryStats._from_rs(quality_rs)

    def get_explanation(self, x, y, z) -> Explanation:
        explanation_rs = self._raw.explain(x, y, z)
        return Explanation._from_rs(explanation_rs)

    def explain(self, x: float, y: float, z: float) -> None:
        explanation = self.get_explanation(x, y, z)
        rich.print(explanation)
        if len(explanation.qualities) > 1:
            rich.print(
                "Qualities alpha-blended together until exhaustion "
                "or combined quality alpha ~ 1.0"
            )

    def query_many(self, x, y, z) -> xr.Dataset:
        x, y, z = np.broadcast_arrays(x, y, z)

        original_shape = x.shape
        ndim = x.ndim
        dims = tuple(f"d{i}" for i in range(ndim))

        quality_array = self._raw.query_many(
            x.astype(np.float32, copy=False).ravel(),
            y.astype(np.float32, copy=False).ravel(),
            z.astype(np.float32, copy=False).ravel(),
        )

        var_names = ["rho", "vp", "vs", "qp", "qs"]
        data_vars = {}
        for i, name in enumerate(var_names):
            data_vars[name] = (dims, quality_array[:, i].reshape(original_shape))

        dset = xr.Dataset(
            data_vars=data_vars,
            coords={
                "x": (dims, x),
                "y": (dims, y),
                "z": (dims, z),
            },
        )

        dset = dset.set_xindex(
            ["x", "y", "z"],
            xr.indexes.NDPointIndex,
            tree_adapter_cls=xoak.SklearnBallTreeAdapter,
        )

        return dset

    def _query_chunk_wrapper(self, x, y, z):
        """
        Internal helper to bridge Dask chunks to the Rust backend.
        This function receives NumPy arrays (chunks) from Dask.
        """
        quality_array = self._raw.query_many(
            x.ravel(),
            y.ravel(),
            z.ravel(),
        )

        return quality_array.reshape((*x.shape, 6))

    def assign_qualities(self, model: xr.DataTree) -> xr.DataTree:
        var_names = list(Component)
        coords = [Coordinate.X.value, Coordinate.Y.value, Coordinate.Z.value]

        def process_node(ds: xr.Dataset) -> xr.Dataset:
            ds = ds.copy()
            if not all(c in ds for c in coords):
                return ds
            qualities = xr.apply_ufunc(
                self._query_chunk_wrapper,  # The function to apply to each chunk
                ds[Coordinate.X],
                ds[Coordinate.Y],
                ds[Coordinate.Z],
                input_core_dims=[[], [], []],  # Treat inputs as scalars per-element
                output_core_dims=[["quality_dim"]],  # We are adding a new dimension
                dask="parallelized",
                output_dtypes=[np.float32],
                dask_gufunc_kwargs={"output_sizes": {"quality_dim": len(var_names)}},
            )

            for i, name in enumerate(var_names):
                ds[name] = qualities.isel(quality_dim=i)

            return ds

        model["block"] = model["block"].map_over_datasets(process_node)
        return model
