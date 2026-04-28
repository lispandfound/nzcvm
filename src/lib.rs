pub mod blend;
pub mod mesh;
pub mod model;
pub mod model_tree;
pub mod quality;
pub mod query;
pub mod real;
pub mod simplex;
mod tree_query;
use pyo3::prelude::*;

#[pymodule]
mod nzcvm {
    use crate::blend::{Blend, BlendDispatch, Erase, Over};
    use crate::mesh::MeshModel;
    use crate::model::{ConstantModel, InterpolateModel, Model};
    use crate::model_tree::ModelTree;
    use crate::quality::Quality;
    use crate::query::Query;
    use crate::real::Real;
    use nalgebra::{Affine3, Matrix4, Point3, Point4};
    use ndarray::{array, azip, Axis};
    use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pythonize::pythonize;

    /// How the vectorised query result is composited into the output buffer.
    ///
    /// Passed to [`PyModelTree::query_many`] to select between overwriting
    /// and Porter-Duff alpha blending.
    ///
    /// Internally converted to a [`BlendDispatch`] before entering the query
    /// loop so that no `match` is needed inside the hot path.
    #[pyclass(from_py_object)]
    #[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum BlendMode {
        /// Overwrite each buffer row with the query result.
        ///
        /// Rows for points outside all matching models are left at their
        /// current value (typically zeros if the caller zeroed the buffer).
        Erase,
        /// Blend the query result *over* the existing buffer row using the
        /// Porter-Duff "over" operator.
        Over,
    }

    impl From<BlendMode> for BlendDispatch {
        fn from(mode: BlendMode) -> Self {
            match mode {
                BlendMode::Erase => BlendDispatch::Erase(Erase),
                BlendMode::Over => BlendDispatch::Over(Over),
            }
        }
    }

    /// Parameters controlling a vectorised query (priority range + blend mode).
    ///
    /// Bundles `priority_lo`, `priority_hi`, and `blend_mode` into a single
    /// object so that [`PyModelTree::query_many`] has a compact signature.
    #[pyclass(from_py_object)]
    #[derive(Debug, Clone, Copy)]
    pub struct QueryParams {
        pub priority_lo: u8,
        pub priority_hi: u8,
        pub blend_mode: BlendMode,
    }

    #[pymethods]
    impl QueryParams {
        /// Create a new ``QueryParams``.
        ///
        /// Parameters
        /// ----------
        /// priority_lo, priority_hi :
        ///     Inclusive priority bounds; only models whose priority falls in
        ///     ``[priority_lo, priority_hi]`` are queried.  Pass ``0, 255`` to
        ///     query all models.
        /// blend_mode :
        ///     How the query result is composited into the output buffer.
        #[new]
        pub fn new(priority_lo: u8, priority_hi: u8, blend_mode: BlendMode) -> Self {
            QueryParams {
                priority_lo,
                priority_hi,
                blend_mode,
            }
        }
    }

    /// Python-facing single tetrahedral mesh model.
    ///
    /// Wraps a [`MeshModel`] and exposes query, AABB, name and priority.
    /// Consumed (moved into a [`ModelTree`]) when passed to [`model_tree`].
    #[pyclass]
    pub struct PyMeshModel {
        inner: Option<MeshModel>,
    }

    /// Python-facing velocity model (a compiled [`ModelTree`]).
    #[pyclass]
    pub struct PyModelTree {
        pub inner: ModelTree,
    }

    /// Create a [`PyMeshModel`] from raw NumPy arrays.
    ///
    /// # Arguments
    ///
    /// * `vertices_py`   – `(N, 3)` float array of vertex coordinates.
    /// * `faces_py`      – `(M, 4)` integer array of tetrahedral cell indices.
    /// * `types_py`      – `(M,)` u8 array: `0` = constant, `1` = interpolate.
    /// * `models_py`     – flat index array for the model lookup table.
    /// * `qualities_py`  – `(Q, 6)` array of quality values indexed by `models_py`.
    /// * `priority`      – model priority (lower number = higher priority).
    /// * `transform_py`  – optional 4×4 world-to-local affine transform.
    /// * `name`          – optional human-readable name for the model.
    #[pyfunction]
    #[pyo3(signature = (vertices_py, faces_py, types_py, models_py, qualities_py, priority, transform_py, name=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn mesh_model(
        vertices_py: PyReadonlyArray2<Real>,
        faces_py: PyReadonlyArray2<usize>,
        types_py: PyReadonlyArray1<u8>,
        models_py: PyReadonlyArray1<usize>,
        qualities_py: PyReadonlyArray2<Real>,
        priority: u8,
        transform_py: Option<PyReadonlyArray2<Real>>,
        name: Option<String>,
    ) -> PyResult<PyMeshModel> {
        let vertices = vertices_py
            .as_array()
            .axis_iter(Axis(0))
            .map(|a| Point3::new(a[0], a[1], a[2]))
            .collect();
        let faces = faces_py
            .as_array()
            .axis_iter(Axis(0))
            .map(|a| Point4::new(a[0], a[1], a[2], a[3]))
            .collect();

        // Build Vec<Quality> from the (N, 6) numpy array.
        let qualities: Vec<Quality> = qualities_py
            .as_array()
            .axis_iter(Axis(0))
            .map(Quality::from)
            .collect();

        let types = types_py.as_array();
        let mut models_vec = Vec::with_capacity(types.len());
        let model_idx = models_py.as_array();
        let mut idx = 0;
        for model_type in types.iter() {
            match model_type {
                0 => {
                    models_vec.push(Model::from(ConstantModel {
                        quality: model_idx[idx],
                    }));
                    idx += 1;
                }
                1 => {
                    models_vec.push(Model::from(InterpolateModel {
                        qualities: Point4::new(
                            model_idx[idx],
                            model_idx[idx + 1],
                            model_idx[idx + 2],
                            model_idx[idx + 3],
                        ),
                    }));
                    idx += 4;
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Invalid model type {}",
                        model_type
                    )))
                }
            }
        }

        let transform = transform_py.map(|arr| {
            Affine3::from_matrix_unchecked(Matrix4::from_iterator(arr.as_array().iter().cloned()))
        });

        Ok(PyMeshModel {
            inner: Some(MeshModel::new(
                vertices,
                faces,
                models_vec,
                qualities,
                priority,
                transform,
                name.unwrap_or_default(),
            )),
        })
    }

    /// Combine one or more [`PyMeshModel`]s into a queryable [`PyModelTree`].
    ///
    /// Each `PyMeshModel` is consumed (its inner data is moved out) so it
    /// cannot be reused after this call.
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if a `PyMeshModel` has already been consumed.
    #[pyfunction]
    pub fn model_tree(py: Python<'_>, mesh_models: Vec<Py<PyMeshModel>>) -> PyResult<PyModelTree> {
        let mut models = Vec::with_capacity(mesh_models.len());
        for m_py in mesh_models {
            let mut m = m_py.borrow_mut(py);
            let model = m.inner.take().ok_or_else(|| {
                PyValueError::new_err(
                    "MeshModel has already been consumed by a previous model_tree() call",
                )
            })?;
            models.push(model);
        }
        Ok(PyModelTree {
            inner: ModelTree::new(models),
        })
    }

    #[pymethods]
    impl PyMeshModel {
        /// Query the mesh model at a single point.
        ///
        /// Returns a dict with keys `rho`, `vp`, `vs`, `qp`, `qs`, `alpha`,
        /// or `None` if the point lies outside this model's region.
        ///
        /// # Errors
        ///
        /// Returns `ValueError` if the model has been consumed by `model_tree()`.
        pub fn query<'py>(
            &self,
            py: Python<'py>,
            x: Real,
            y: Real,
            z: Real,
        ) -> PyResult<Option<Bound<'py, PyAny>>> {
            let inner = self.inner.as_ref().ok_or_else(|| {
                PyValueError::new_err("MeshModel has been consumed by model_tree()")
            })?;
            let pt = Point3::new(x, y, z);
            inner
                .query(pt)
                .map(|q| pythonize(py, &q).map_err(|e| e.into()))
                .transpose()
        }

        /// Return the 3-D axis-aligned bounding box of this mesh model.
        ///
        /// Returns a pair ``(min_xyz, max_xyz)`` of shape-``(3,)`` float32 arrays.
        ///
        /// # Errors
        ///
        /// Returns `ValueError` if the model has been consumed by `model_tree()`.
        #[allow(clippy::type_complexity)]
        pub fn aabb<'py>(
            &self,
            py: Python<'py>,
        ) -> PyResult<(Bound<'py, PyArray1<Real>>, Bound<'py, PyArray1<Real>>)> {
            let inner = self.inner.as_ref().ok_or_else(|| {
                PyValueError::new_err("MeshModel has been consumed by model_tree()")
            })?;
            let b = inner.aabb3();
            let min = array![b.min.x, b.min.y, b.min.z];
            let max = array![b.max.x, b.max.y, b.max.z];
            Ok((min.into_pyarray(py), max.into_pyarray(py)))
        }

        /// Human-readable name assigned at construction time.
        ///
        /// # Errors
        ///
        /// Returns `ValueError` if the model has been consumed by `model_tree()`.
        #[getter]
        pub fn name(&self) -> PyResult<String> {
            let inner = self.inner.as_ref().ok_or_else(|| {
                PyValueError::new_err("MeshModel has been consumed by model_tree()")
            })?;
            Ok(inner.name.clone())
        }

        /// Model priority (lower number = higher priority in the tree).
        ///
        /// # Errors
        ///
        /// Returns `ValueError` if the model has been consumed by `model_tree()`.
        #[getter]
        pub fn priority(&self) -> PyResult<u8> {
            let inner = self.inner.as_ref().ok_or_else(|| {
                PyValueError::new_err("MeshModel has been consumed by model_tree()")
            })?;
            Ok(inner.priority)
        }

        /// Return a serialisable Python dict describing this mesh model.
        ///
        /// # Errors
        ///
        /// Returns `ValueError` if the model has been consumed by `model_tree()`.
        pub fn view<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner.as_ref().ok_or_else(|| {
                PyValueError::new_err("MeshModel has been consumed by model_tree()")
            })?;
            pythonize(py, &inner.view()).map_err(|e| e.into())
        }
    }

    #[pymethods]
    impl PyModelTree {
        /// Query the velocity model at a single point.
        ///
        /// Only models whose priority falls in `[priority_lo, priority_hi]` contribute.
        /// Use `0, 255` to query all models.
        /// Returns a dict `{rho, vp, vs, qp, qs, alpha}` or `None` if outside all models.
        pub fn query<'py>(
            &self,
            py: Python<'py>,
            x: Real,
            y: Real,
            z: Real,
            priority_lo: u8,
            priority_hi: u8,
        ) -> PyResult<Option<Bound<'py, PyAny>>> {
            let pt = Point3::new(x, y, z);
            self.inner
                .query(pt, None, priority_lo, priority_hi)
                .map(|q| pythonize(py, &q).map_err(|e| e.into()))
                .transpose()
        }

        /// Return the combined 3-D axis-aligned bounding box of all mesh models.
        ///
        /// Returns a pair ``(min_xyz, max_xyz)`` of shape-``(3,)`` float32 arrays.
        pub fn aabb<'py>(
            &self,
            py: Python<'py>,
        ) -> (Bound<'py, PyArray1<Real>>, Bound<'py, PyArray1<Real>>) {
            let b = self.inner.aabb();
            let min = array![b.min.x, b.min.y, b.min.z];
            let max = array![b.max.x, b.max.y, b.max.z];
            (min.into_pyarray(py), max.into_pyarray(py))
        }

        /// Query with BVH traversal statistics (AABB tests, simplex tests, etc.).
        ///
        /// Returns a dict with keys `aabb_tests`, `simplex_tests`, `hit_count`,
        /// `output` (quality dict or `None`), and `elapsed` (nanoseconds).
        pub fn query_stats<'py>(
            &self,
            py: Python<'py>,
            x: Real,
            y: Real,
            z: Real,
        ) -> PyResult<Bound<'py, PyAny>> {
            let pt = Point3::new(x, y, z);
            pythonize(py, &self.inner.query_stats(pt)).map_err(|e| e.into())
        }

        /// Return a full diagnostic breakdown listing each model's contribution.
        ///
        /// Returns a dict with keys `contributions` (list of dicts), `output`
        /// (quality dict or `None`), and `termination` (int or `None`).
        pub fn explain<'py>(
            &self,
            py: Python<'py>,
            x: Real,
            y: Real,
            z: Real,
        ) -> PyResult<Bound<'py, PyAny>> {
            let pt = Point3::new(x, y, z);
            pythonize(py, &self.inner.query_explain(pt)).map_err(|e| e.into())
        }

        /// Return a serialisable Python dict describing the model tree structure.
        pub fn view<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let view = self.inner.view();
            pythonize(py, &view).map_err(|e| e.into())
        }

        /// Query the model for many points at once, writing results in-place.
        ///
        /// `buffer` is an `(N, 6)` float32 array with columns
        /// `[rho, vp, vs, qp, qs, alpha]`.  It is modified **in-place**.
        ///
        /// `xyz` is an `(N, 3)` float32 array with columns `[x, y, z]`.
        ///
        /// `params` bundles the priority bounds and blend mode.  Use
        /// `QueryParams(0, 255, BlendMode.Erase)` to overwrite all rows.
        pub fn query_many(
            &self,
            mut buffer: PyReadwriteArray2<Real>,
            xyz: PyReadonlyArray2<Real>,
            params: QueryParams,
        ) {
            let blend: BlendDispatch = params.blend_mode.into();
            let lo = params.priority_lo;
            let hi = params.priority_hi;
            let mut buf = buffer.as_array_mut();
            let coords = xyz.as_array();
            azip!((mut out_lane in buf.rows_mut(), xyz_row in coords.rows()) {
                let pt = Point3::new(xyz_row[0], xyz_row[1], xyz_row[2]);
                if let Some(new_q) = self.inner.query(pt, None, lo, hi) {
                    let existing = Some(Quality::from(out_lane.view()));
                    let q = blend.apply(existing, new_q);
                    out_lane[0] = q.rho;
                    out_lane[1] = q.vp;
                    out_lane[2] = q.vs;
                    out_lane[3] = q.qp;
                    out_lane[4] = q.qs;
                    out_lane[5] = q.alpha;
                }
            });
        }

        /// Print a human-readable summary of the model tree to stdout.
        pub fn print_structure(&self) {
            self.inner.pretty_print();
        }
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyMeshModel>()?;
        m.add_class::<PyModelTree>()?;
        m.add_class::<BlendMode>()?;
        m.add_class::<QueryParams>()?;
        m.add_function(wrap_pyfunction!(mesh_model, m)?)?;
        m.add_function(wrap_pyfunction!(model_tree, m)?)?;

        Ok(())
    }
}
