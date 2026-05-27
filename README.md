# NZCVM

New Zealand Community Velocity Model — tools for building and querying
tetrahedral velocity models.

A velocity model is a collection of tetrahedral meshes stored in VTKHDF
format.  Each mesh carries seismic velocity (Vp, Vs), density (rho), and
quality-factor (Qp, Qs) cell data, plus a priority value that controls
blending when meshes overlap.  Queries return alpha-composited properties at
arbitrary 3-D coordinates.

---

## Build

The core query engine is a Rust extension built with
[maturin](https://github.com/PyO3/maturin).  Python ≥ 3.13 is required.

```sh
pip install maturin
python3.13 -m maturin build --release --interpreter python3.13
pip install target/wheels/*.whl --force-reinstall
```

### Non-pip dependencies

| Dependency | Purpose |
|---|---|
| Rust toolchain (stable) | Compiling the extension |
| HDF5 ≥ 1.12 | Runtime requirement of h5py |

All Python dependencies are declared in `pyproject.toml` and installed via pip
or uv.

---

## Usage

```python
from nzcvm.model import ModelTree

tree = ModelTree.load_models("/path/to/models")
quality = tree.query(x=1_000.0, y=2_000.0, z=-500.0)
print(quality.vp, quality.vs)
```

To generate a full 3-D velocity grid from a TOML config:

```sh
nzcvm generate config.toml output/
```

---

## Testing

```sh
python3.13 -m pytest tests/
python3.13 -m pytest --doctest-modules nzcvm/   # doctests
python3.13 -m ruff check nzcvm/ tests/          # linting
python3.13 -m ty check nzcvm/                   # type checking
```

---

## Code architecture

The package is structured in three layers, each with a narrowly defined
responsibility:

**`nzcvm.model`** — Rust wrappers.  `MeshModel` holds a single tetrahedral
mesh; `ModelTree` combines many meshes into a priority-ordered BVH tree and
handles alpha-composited queries.  Both live close to the metal: no grid
logic here.

**`nzcvm.layers`** — Query pipeline.  A `Layer` is a composable unit that
accepts a grid (xarray Dataset) and a model range and returns a `Qualities`
dataset.  Layers chain via constructor injection (`next_layer`), so each
layer wraps the next without coupling them.  Built-in layers:

- `QueryLayer` — performs the actual model queries against a `ModelTree`.
- `ElyLayer` — applies the Ely et al. (2010) near-surface GTL taper.
- `OffshoreBasinLayer` — fills offshore and coastal regions with a 1-D
  velocity profile.
- `ClampLayer` — clamps velocity components to physical bounds.

New layers can be registered with the `@functional_layer` decorator.

**`nzcvm.grids`** — Grid construction.  Builds the 3-D curvilinear mesh
(SW4 format) as an xarray DataTree.  Grids are chunked lazily with Dask and
assembled from a `VelocityModelConfig`.

`Qualities` (in `nzcvm.qualities`) is an `xr.Dataset` subclass that carries
the typed velocity, density, and quality-factor arrays returned by every layer.

### Design notes

- Layers are decoupled: `QueryLayer` does not know about `ElyLayer` and vice
  versa.  Adding a new transformation or model source means writing one class,
  not modifying existing ones.
- The Rust extension owns all spatial indexing.  Python only orchestrates
  grid construction and layer chaining.
- Config objects (`nzcvm.config`) are plain dataclasses deserialised by
  mashumaro.  They carry no behaviour — their only job is to convey
  user intent to the Python layer.
