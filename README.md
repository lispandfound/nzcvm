# NZCVM

New Zealand Community Velocity Model — tools for building and querying
tetrahedral velocity models.

A velocity model is a collection of tetrahedral meshes stored in VTKHDF
format. Each mesh carries seismic velocity (Vp, Vs), density (rho), and
quality-factor (Qp, Qs) cell data, plus a priority value that controls
blending when meshes overlap. Queries return alpha-composited properties at
arbitrary 3-D coordinates.

---

## Build

The core query engine is a Rust extension built with
[maturin](https://github.com/PyO3/maturin). Python ≥ 3.13 is required.

```sh
pip install maturin
python3.13 -m maturin build --release --interpreter python3.13
pip install target/wheels/*.whl --force-reinstall
```

Alternatively, `uv` is the preferred build tool for this repo

``` sh
uv sync # Creates a venv and builds the rust extension
```

### Non-pip dependencies

| Dependency              | Purpose                                           |
|-------------------------|---------------------------------------------------|
| Rust toolchain (stable) | Compiling the extension (if building from source) |
| HDF5 ≥ 1.12             | Runtime requirement of h5py                       |

All Python dependencies are declared in `pyproject.toml` and installed via
pip. PyVista is an optional visualisation dependency:

```sh
pip install nzcvm[vis]   # also installs pyvista
```

---

## Usage

### Python API

```python
from nzcvm.model import ModelTree

tree = ModelTree.load_models("/path/to/models")
quality = tree.query(x=1_000.0, y=2_000.0, z=-500.0)
print(quality.vp, quality.vs)
```

### CLI — generating a velocity grid

```sh
nzcvm generate config.toml output/
```

### Supported grid formats

| Format          | `type` value | Key parameters                              |
|-----------------|--------------|---------------------------------------------|
| SW4 curvilinear | `"sw4"`      | `extent_x/y`, `orientation`, `refinements`  |
| EMOD3D          | `"emod3d"`   | `nx`, `ny`, `nz`, `resolution`, `topo_type` |

Both grids are specified under the `[grid]` section of a TOML config file.

### Supported layers

Layers are processed in the order they appear in the config. Each layer
wraps the next, so the first listed layer runs last (outermost wrapper).

| Type        | Description                                                            |
|-------------|------------------------------------------------------------------------|
| `query`     | Queries the tetrahedral model tree (always required)                   |
| `ely`       | Ely et al. (2010) near-surface Vs taper using a Vs30 map               |
| `offshore`  | 1-D offshore/coastal velocity profile                                  |
| `coastline` | Coastline-distance fill for offshore grid cells                        |
| `clamp`     | Clamps Vp, Vs, and Vp/Vs ratios to physical bounds                     |
| custom      | Any layer registered with `@functional_layer` or as a `Layer` subclass |

### Supported output formats

| Format   | Description                                                               |
|----------|---------------------------------------------------------------------------|
| `zarr`   | Chunked array store (Useful for debugging outputs, contains all metadata) |
| `netcdf` | NetCDF4/HDF5 via xarray                                                   |
| `sfile`  | NZVM sfile binary format                                                  |
| `emod3d` | EMOD3D binary velocity model directory                                    |

### Example configuration

The following config generates a ~50 km SW4 grid centred near Wellington,
queries the EP2020 tomography model, and applies the Ely GTL taper on top.

```toml
[metadata]
title = "Wellington domain"

[grid]
type = "sw4"
surface = "./resources/dem.vtkhdf"
extent_x = 50000.0
extent_y = 50000.0

[grid.orientation]
azimuth = 39.0
origin_crs = 2193
origin_x = 1749030.0
origin_y = 5428152.0

[grid.refinements.surface_layer]
resolution = 100.0
bottom = 3500.0

[grid.refinements.deep_layer]
resolution = 400.0
bottom = 42000.0

[grid.chunks]
i = 256
j = 256
k = 128

[[layers]]
type = "clamp"
min_vp_vs_ratio = 1.73
max_vp_vs_ratio = 4.0

[layers.clamps.vs]
min = 500.0

[[layers]]
type = "ely"
vs30 = "./resources/vs30.vtkhdf"
depth_t = 450.0

[[layers]]
type = "query"
model_path = "./models"
model_glob = "ep2020.vtkhdf"
```

---

## Visualisation

Model output in Zarr or NetCDF format is a standard xarray DataTree. Each
grid is a Dataset with coordinates `x`, `y`, `z` and data variables `vp`,
`vs`, `rho`, `qp`, `qs`, `alpha`.

```python
import xarray as xr
import matplotlib.pyplot as plt

# Open the output DataTree
dt = xr.open_datatree("output.zarr", engine="zarr")

# Select the first grid
ds = dt["surface_layer"].ds

# Plot a horizontal Vs slice at depth index 10
ds["vs"].isel(k=10).plot(x="x", y="y", cmap="viridis")
plt.title("Vs at k=10")
plt.show()
```

Depth slices along any axis:

```python
# Vertical cross-section along j = 128
ds["vs"].isel(j=128).plot(x="x", y="z", cmap="viridis", yincrease=False)
plt.title("Vs — N–S vertical cross-section")
plt.show()
```

For interactive 3-D visualisation with PyVista (requires `nzcvm[visualization]`):

```sh
nzcvm view output.zarr --scalar vs
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

The package is structured in four subpackages, each with a narrowly defined
responsibility:

### **`nzcvm.model`** 
Contains geospatial rust wrappers. `MeshModel` holds a single tetrahedral mesh;
`ModelTree` combines many meshes into a priority-ordered BVH tree and handles
alpha-composited queries. The `Surface` class interpolates values from a 2d
triangular surface mesh.

### **`nzcvm.layers`**.
A `Layer` is a composable unit that accepts a `Grid` (xarray
Dataset of 3-D coordinates) and a `ModelRange` and returns a `Qualities`
dataset. Layers chain via constructor injection (`next_layer`), so each layer
wraps the next without coupling them. Layers are registered by subclass hooks
into the `Layer` superclass.

Built-in layers:

- `QueryLayer` performs the actual model queries against a `ModelTree`.
- `ElyLayer` applies the Ely et al. (2010) near-surface GTL taper.
- `OffshoreBasinLayer` fills offshore and coastal regions with a 1-D
  velocity profile.
- `ClampLayer` clamps velocity components to physical bounds.

### **`nzcvm.config`** 
Contains grid and layer configuration. Every layer has a companion
**`LayerConfig`** dataclass that carries its parameters. Every grid has a
companion **`GridConfig`** dataclass. Config objects are plain dataclasses
deserialised from TOML, JSON or YAML by mashumaro. Being dataclasses, they can
also be instantiated in pure Python. Configuration objects contain lightweight
schema validation for things like: `float` bounds checking, layer order and
requirements.

### **`nzcvm.grids`** 
Defines meshgrids for interpolation. Builds 3-D curvilinear
meshes (SW4 or EMOD3D format) as xarray DataTrees. Grids are chunked lazily with
Dask and assembled from a `GridConfig`.

`Qualities` (in `nzcvm.qualities`) is an `xr.Dataset` subclass that carries
the typed velocity, density, and quality-factor arrays returned by every layer.

## Extending with custom grids and layers

This package is designed to make extension easy. To do this, the code is
carefully designed to allow plug-and-play of both 3rd party grids and layers.

### Functional layers (simple case)

The `@functional_layer` decorator turns a plain function into a registered
layer. The function receives the current `Grid` and the `next_layer`
callable, applies its transformation, and returns `Qualities`.

```python
from nzcvm.layers.functional import functional_layer
from nzcvm.grids import Grid
from nzcvm.layers.core import Layer
from nzcvm.query import ModelRange

@functional_layer
def scale_vs(
    grid: Grid,
    model_range: ModelRange,
    *,
    next_layer: Layer,
    factor: float,
):
    """Multiply Vs throughout the model by *factor*."""
    qualities = next_layer(grid, model_range)
    qualities["vs"] = qualities["vs"] * factor
    return qualities
```

Add it to a config:

```toml
[[layers]]
type = "scale_vs"
factor = 0.9
```

See `examples/near_fault.py` for a more complete example that uses a spatial
distance mask to perturb Vs near a fault zone.

### Class-based layers

For layers that need state, caching, or a non-trivial config, subclass `Layer`
and provide a matching `LayerConfig`.

```python
from dataclasses import dataclass
import numpy as np
from nzcvm.layers.core import Layer
from nzcvm.config.layers.core import LayerConfig
from nzcvm.grids import Grid
from nzcvm.qualities import Qualities
from nzcvm.query import ModelRange


@dataclass
class BackusAveragingConfig(LayerConfig):
    type: str = "backus_averaging"
    window_size: int = 5


class BackusAveragingLayer(Layer, config_cls=BackusAveragingConfig):
    """Backus averaging: a moving-window harmonic mean over depth slices.

    This smooths the model's velocity structure over a sliding window of
    *window_size* depth cells, reducing the effective resolution to improve
    waveform accuracy for long-period simulations.
    """

    def __init__(self, config: BackusAveragingConfig, next_layer: Layer):
        self._window = config.window_size
        self._next = next_layer

    def __call__(
        self, grid: Grid, model_range: ModelRange = ModelRange.ALL
    ) -> Qualities:
        qualities = self._next(grid, model_range)
        w = self._window
        for comp in ("vp", "vs"):
            arr = qualities[comp].values
            # Harmonic mean along the k (depth) axis using a sliding window
            slow = 1.0 / np.maximum(arr, 1e-6)
            kernel = np.ones(w) / w
            smoothed_slow = np.apply_along_axis(
                lambda x: np.convolve(x, kernel, mode="same"), axis=-1, arr=slow
            )
            qualities[comp] = qualities[comp].copy(data=1.0 / smoothed_slow)
        return qualities
```

Use it in TOML:

```toml
[[layers]]
type = "backus_averaging"
window_size = 7
```

### Custom grid types

A grid is any xarray Dataset with coordinates `x`, `y`, `z` (in metres,
projected CRS). Implement `build_grid(spec, surface)` and register it via
`GridSchema`.

The simplest custom grid is a borehole — a single vertical column of points:

```python
import numpy as np
import xarray as xr
from nzcvm.grids.grid import Grid, GridSchema


def borehole_grid(
    x: float,
    y: float,
    z_top: float,
    z_bottom: float,
    dz: float,
) -> Grid:
    """A single vertical column of query points (a synthetic borehole).

    Parameters
    ----------
    x, y:
        Horizontal position in the model CRS (metres).
    z_top:
        Elevation of the top of the borehole (metres, negative = below sea level).
    z_bottom:
        Elevation of the base of the borehole (metres).
    dz:
        Vertical sample spacing (metres).
    """
    z = np.arange(z_top, z_bottom, -dz, dtype=np.float32)
    return GridSchema.new(
        x=('k', np.full_like(z, x)),
        y=('k', np.full_like(z, y)),
        z=('k', z)
    )
```

Pass the resulting Dataset directly to any layer pipeline:

```python
grid = borehole_grid(x=1_749_030, y=5_428_152, z_top=0.0, z_bottom=-5000.0, dz=50.0)
qualities = pipeline(grid)
```

Grids can be registered with a configuration using `@build_grids_from_config.register` and creating a `GridConfig` dataclass for configuration. 

``` python
import numpy as np
from nzcvm.config.grids import GridConfig
from nzcvm.grids.grid import Grid, GridSchema
from nzcvm.grids import build_grid_from_config

class BoreholeConfig(GridConfig):
    x: float
    y: float
    z_top: float
    z_bottom: float
    dz: float
    type: Literal['borehole'] = 'borehole'


@build_grid_from_config.register
def borehole_grid(config: BoreholeConfig) -> Grid:
    """A single vertical column of query points (a synthetic borehole).

    Parameters
    ----------
    x, y:
        Horizontal position in the model CRS (metres).
    z_top:
        Elevation of the top of the borehole (metres, negative = below sea level).
    z_bottom:
        Elevation of the base of the borehole (metres).
    dz:
        Vertical sample spacing (metres).
    """
    z = np.arange(config.z_top, config.z_bottom, -config.dz, dtype=np.float32)
    return GridSchema.new(
        x=('k', np.full_like(z, x)),
        y=('k', np.full_like(z, y)),
        z=('k', z)
    )
```

Now these grids can be included in code like so:

``` toml
[grid]
type = "borehole"
x = 0.0
y = 0.0
z_top = 0.0
z_bottom = 100.0
```

