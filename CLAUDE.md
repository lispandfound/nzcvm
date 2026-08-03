# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NZCVM (New Zealand Community Velocity Model) builds and queries 3-D seismic
velocity models. A velocity model is a collection of tetrahedral meshes
(VTKHDF) carrying Vp, Vs, density, and Qp/Qs, plus a priority value that
controls blending where meshes overlap. The core spatial query engine (BVH
tree lookup, alpha-compositing, mesh geometry) is written in Rust and exposed
to Python via a PyO3 extension module; everything above that (config parsing,
grid construction, the layer pipeline, output formats) is Python.

## Build

Python ≥ 3.13 and a stable Rust toolchain are required. `uv` is the
preferred workflow:

```sh
uv sync   # creates a venv and builds the Rust extension via maturin
```

The extension can also be built manually with `maturin build --release
--interpreter python3.13` and installed with pip. Any change under `src/`
(Rust) requires reinstalling/rebuilding the extension before Python picks it
up — `uv run` does this automatically via maturin's build isolation, but a
manual `pip install` workflow needs an explicit rebuild.

## Common commands

```sh
just test              # pytest + doctests + cargo test
just pytest            # python tests + doctests only
just cargo             # rust tests only (cargo test)
just lint              # ty + ruff + clippy
just ty                # uv run ty check nzcvm          (type checking)
just ruff              # uv run ruff format + ruff check --select I --fix
just clippy            # cargo clippy -- -D warnings

# equivalent raw invocations:
uv run --dev pytest -s tests
uv run --dev pytest tests/test_layers.py::test_clamp_vs_min_lower_bound  # single test
uv run --dev pytest --doctest-modules nzcvm/ -v
cargo test
cargo bench                      # criterion benchmarks, see benches/model_bench.rs
uv run ty check nzcvm
uv run ruff check nzcvm/ tests/
cargo clippy -- -D warnings

uv run nzcvm generate config.toml output.zarr   # run the CLI
uv run nzcvm --help                             # list all subcommands
```

`lefthook.yml` runs `ty`, `ruff`, `yamllint`, and `clippy` as pre-push hooks
on staged files — keep changes passing these before pushing.

`tests/conftest.py` builds raw Rust-backed fixtures (`unit_tetrahedron_mesh`,
`unit_tetrahedron_tree`) directly through the compiled `nzcvm.nzcvm` module,
and a `make_grid()` helper for constructing minimal non-dask `Grid`s. `test_cli.py`
is only a `--help` smoke test for every subcommand (no data files required).
`nzcvm/scripts/nzcvm.py` is excluded from doctest collection (see root
`conftest.py`).

The `basins` recipe in `Justfile` regenerates every regional basin model from
source data under `${NZCVM_DATA_ROOT}` (a large private data directory, path
set in `.env`); it's the closest thing to an integration test for the mesh
construction pipeline but is not part of `just test` since it needs
unversioned data.

## Architecture

### Rust core (`src/`, module `nzcvm`)

- `mesh.rs` — `MeshModel`: a single tetrahedral mesh with per-cell qualities.
- `model_tree.rs` — `ModelTree`: many meshes combined into a priority-ordered
  BVH (via the `bvh` crate) for spatial queries.
- `tree_query.rs` / `query.rs` — vectorised nearest/containing-tetrahedron
  queries and alpha-compositing across overlapping meshes.
- `surface.rs` — `SurfaceModel`: interpolation over a 2-D triangular surface
  mesh (used for DEMs and basin outlines).
- `model.rs` — `ConstantModel` / `InterpolateModel`: simple 1-D value models
  (e.g. constant-property basins, layered 1-D velocity models).
- `quality.rs`, `blend.rs`, `simplex.rs`, `triangle.rs`, `real.rs` — supporting
  geometry/numeric primitives. `Real` is the shared float type (`real.rs`)
  used across the extension, so Rust↔NumPy boundaries are typed consistently.
- PyO3 bindings live inline in the `#[pymodule] mod nzcvm` block in `lib.rs`
  rather than a separate bindings file — search there first for the Python
  entry points into Rust (`mesh_model`, `model_tree`, etc., as seen from
  `tests/conftest.py`).

### Python package (`nzcvm/`)

Four subpackages, each with a narrow responsibility (also documented in
`README.md`, which has fuller worked examples for extension points):

- **`nzcvm/models/`** — thin Python wrappers around the Rust types
  (`MeshModel`, `ModelTree`, `Surface`).
- **`nzcvm/layers/`** — the query pipeline. A `Layer` (`layers/core.py`)
  takes a `Grid`, a `shapely.Geometry` (the layer's spatial domain, used to
  prune/mask queries), and a `next_layer`, and returns `Qualities`. Layers
  chain via constructor injection — `build_pipeline()` in `layers/pipeline.py`
  builds the chain from a `list[LayerConfig]` in reverse, so the *first*
  config entry ends up as the *outermost* wrapper and the last config entry
  (conventionally `query`) is called innermost/first. `_SentinelLayer`
  terminates the chain and raises if a grid point falls outside every layer's
  geometry. Built-ins: `QueryLayer` (queries a `ModelTree`), `ElyLayer` (Ely
  et al. 2010 near-surface Vs30 taper), `OffshoreBasinLayer`, `CoastlineLayer`,
  `ClampLayer`. New layers self-register via `Layer.__init_subclass__` (or
  the `@functional_layer` decorator for stateless function-style layers) —
  simply importing the module that defines them is enough to make them
  available to `layer_from_config`.
- **`nzcvm/config/`** — dataclass configs deserialised from TOML/JSON/YAML via
  `mashumaro` (`ConfigObject` in `config/core.py`). Every layer has a paired
  `LayerConfig` subclass (`config/layers/`), every grid a paired `GridConfig`
  subclass (`config/grids/`). `LayerConfig` carries `provides`/`requires`
  lists of coordinate names; `VelocityModelConfig.__post_init__` walks the
  configured layer list and errors if a layer requires a coordinate no
  earlier layer provides. `ConfigObject.__post_init__` also runs any
  validators attached via `Annotated[T, validator]` type hints on fields.
  Discriminated union dispatch (which dataclass a `type = "..."` string in
  TOML maps to) uses mashumaro's `Discriminator`.
- **`nzcvm/grids/`** — builds 3-D curvilinear meshgrids (SW4 or EMOD3D layout)
  as `xr.DataTree`s, chunked lazily with Dask, from a `GridConfig`.
  `nzcvm.qualities.Qualities` is the `xr.Dataset` subclass every layer
  produces (typed `vp`/`vs`/`rho`/`qp`/`qs`/`alpha` arrays).

Execution flow (`layers/pipeline.py:execute_model_pipeline`): each grid's
Dask chunks are dispatched through the layer chain via a single
`map_blocks` call per grid, so individual layers always see a fully-computed
concrete `Grid` chunk and can use plain NumPy — no layer needs to know about
`map_blocks` or `apply_ufunc(dask="parallelized")` itself.

`nzcvm/registry.py` exists because the Rust-backed objects (`ModelTree`,
`Surface`, etc.) aren't picklable. When running the Dask `distributed`
scheduler with `processes=False`, `register_dask_type` stashes such objects
in a global `REGISTRY` keyed by UUID instead of serialising them, since
thread-worker processes share memory with the scheduler. Always load these
resources *outside* a `map_blocks`/`apply_ufunc` boundary and pass them in as
explicit kwargs so the registered `dask_serialize`/`dask_deserialize` hooks
apply. Wrap pipeline execution in `registry.pipeline_context()` so the
registry is cleared afterwards.

`nzcvm/formats/` writes the final `VelocityModel` out as zarr, netcdf, sfile
(NZVM binary), or emod3d.

`nzcvm/scripts/` holds the Typer CLI: `nzcvm.py` is the root app (`generate`
is the main entry point — reads a `VelocityModelConfig`, builds the pipeline
via `layers.pipeline.build_pipeline`, runs it, writes output); `construct_mesh.py`
(subcommand `basin`) builds the tetrahedral basin meshes referenced heavily by
`Justfile`; `convert_tomography.py`, `convert_tiff.py`, `surface_cli.py`,
`tree_stats.py`, `view.py`/`view_basin.py` are the other subcommands.

### Extending with custom layers/grids

See the "Extending" section of `README.md` for full worked examples
(`@functional_layer` for simple stateless transforms, `Layer` subclassing for
stateful/config-driven layers, `@build_grid_from_config.register` for new
grid types). The short version: registration is automatic on
import/subclassing, so a new layer or grid type just needs its module
imported somewhere reachable from config loading.

## Notes

- Config files under `examples/*.toml` are runnable references for the
  `[grid]`/`[[layers]]` schema.
- Model data (`models/`, `resources/`, `*.zarr`, `*.h5`, `*.vtkhdf`, etc.) is
  gitignored; don't assume files matching those patterns in the working tree
  are tracked or reproducible from the repo alone.
- `NZCVM_DATA_ROOT` (source data for basin construction) and `VS30_TIFF` are
  read from `.env` (not committed) by the `Justfile`'s basin-building recipes.
