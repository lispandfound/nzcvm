mod geometry;
pub mod mesh;
pub mod model;
pub mod model_tree;
pub mod quality;
pub mod query;
mod real;
mod simplex;
mod size;
mod tree_query;
use pyo3::prelude::*;

#[pymodule]
mod nzcvm {
    use crate::mesh::MeshModel;
    use crate::model::{ConstantModel, InterpolateModel, Model, ModelExplanation};
    use crate::model_tree::ModelTree;
    use crate::quality::Quality;
    use crate::query::{Explanation, ModelContribution, Query, QueryStats};
    use crate::real::Real;
    use bvh::aabb::Aabb;
    use nalgebra::{Affine3, Matrix4, Point3, Point4};
    use ndarray::{azip, Array2, Axis};
    use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pythonize::pythonize;

    use std::sync::Arc;

    #[pyclass(get_all, from_py_object)]
    #[derive(Clone, Debug)]
    pub enum PySimplexModel {
        Constant {
            quality: PyQuality,
        },

        Interpolate {
            x: PyQuality,
            y: PyQuality,
            z: PyQuality,
            w: PyQuality,
        },
    }

    impl From<ModelExplanation> for PySimplexModel {
        fn from(item: ModelExplanation) -> Self {
            match item {
                ModelExplanation::Constant(ConstantModel { quality }) => PySimplexModel::Constant {
                    quality: quality.into(),
                },
                ModelExplanation::Interpolate(InterpolateModel { qualities }) => {
                    PySimplexModel::Interpolate {
                        x: qualities.x.into(),
                        y: qualities.y.into(),
                        z: qualities.z.into(),
                        w: qualities.w.into(),
                    }
                }
            }
        }
    }

    #[pyclass(get_all, from_py_object)]
    #[derive(Clone, Debug)]
    pub struct PyPoint {
        x: Real,
        y: Real,
        z: Real,
    }

    impl From<Point3<Real>> for PyPoint {
        fn from(item: Point3<Real>) -> Self {
            PyPoint {
                x: item.x,
                y: item.y,
                z: item.z,
            }
        }
    }

    #[pyclass(get_all, from_py_object)]
    #[derive(Clone, Debug)]
    pub struct PyQuality {
        pub rho: Real,
        pub vp: Real,
        pub vs: Real,
        pub qp: Real,
        pub qs: Real,
        pub alpha: Real,
    }

    impl From<Quality> for PyQuality {
        fn from(item: Quality) -> Self {
            PyQuality {
                rho: item.rho,
                vp: item.vp,
                vs: item.vs,
                qp: item.qp,
                qs: item.qs,
                alpha: item.alpha,
            }
        }
    }

    #[pyclass(get_all, from_py_object)]
    #[derive(Clone, Debug)]
    pub struct PyModelContribution {
        pub priority: u8,
        pub quality: PyQuality,
    }

    impl From<ModelContribution> for PyModelContribution {
        fn from(item: ModelContribution) -> Self {
            PyModelContribution {
                priority: item.priority,
                quality: item.quality.into(),
            }
        }
    }

    #[pyclass(get_all, from_py_object)]
    #[derive(Clone, Debug)]
    pub struct PyExplanation {
        pub contributions: Vec<PyModelContribution>,
        pub output: Option<PyQuality>,
        pub termination: Option<usize>,
    }

    impl From<Explanation> for PyExplanation {
        fn from(item: Explanation) -> Self {
            PyExplanation {
                contributions: item.contributions.into_iter().map(|c| c.into()).collect(),
                output: item.output.map(|q| q.into()),
                termination: item.termination,
            }
        }
    }

    #[pyclass(get_all, from_py_object)]
    #[derive(Clone, Debug)]
    pub struct PyQueryStats {
        pub aabb_tests: usize,
        pub simplex_tests: usize,
        pub hit_count: usize,
        pub output: Option<PyQuality>,
        pub elapsed: u128,
    }

    impl From<QueryStats> for PyQueryStats {
        fn from(item: QueryStats) -> Self {
            Self {
                aabb_tests: item.aabb_tests,
                simplex_tests: item.simplex_tests,
                hit_count: item.hit_count,
                output: item.output.map(|x| x.into()),
                elapsed: item.elapsed,
            }
        }
    }

    #[pyclass(get_all, from_py_object)]
    #[derive(Clone, Debug)]
    pub struct PyAabb {
        min: PyPoint,
        max: PyPoint,
    }
    impl From<Aabb<Real, 3>> for PyAabb {
        fn from(item: Aabb<Real, 3>) -> Self {
            Self {
                min: item.min.into(),
                max: item.max.into(),
            }
        }
    }

    #[pyclass]
    pub struct PyMeshModel {
        inner: Option<MeshModel>,
    }

    #[pyclass]
    pub struct PyModel {
        pub inner: Arc<ModelTree>,
    }

    #[pyfunction]
    pub fn mesh_model(
        vertices_py: PyReadonlyArray2<Real>,
        faces_py: PyReadonlyArray2<usize>,
        types_py: PyReadonlyArray1<u8>,
        models_py: PyReadonlyArray1<usize>,
        qualities_py: PyReadonlyArray2<Real>,
        priority: u8,
        transform_py: Option<PyReadonlyArray2<Real>>,
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
        let qualities = qualities_py
            .as_array()
            .axis_iter(Axis(0))
            .map(|a| a.into())
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
                vertices, faces, models_vec, qualities, priority, transform,
            )),
        })
    }

    #[pyfunction]
    pub fn model_tree(py: Python<'_>, mesh_models: Vec<Py<PyMeshModel>>) -> PyResult<PyModel> {
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
        Ok(PyModel {
            inner: Arc::new(ModelTree::new(models)),
        })
    }

    #[pymethods]
    impl PyModel {
        pub fn query(&self, x: Real, y: Real, z: Real) -> PyResult<Option<PyQuality>> {
            let pt = Point3::new(x, y, z);
            Ok(self.inner.query(pt).map(|q| q.into()))
        }

        pub fn aabb(&self) -> PyResult<PyAabb> {
            Ok(self.inner.aabb().into())
        }

        pub fn query_stats(&self, x: Real, y: Real, z: Real) -> PyResult<PyQueryStats> {
            let pt = Point3::new(x, y, z);
            Ok(self.inner.query_stats(pt).into())
        }

        pub fn explain(&self, x: Real, y: Real, z: Real) -> PyResult<PyExplanation> {
            let pt = Point3::new(x, y, z);
            Ok(self.inner.query_explain(pt).into())
        }

        pub fn view<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let view = self.inner.view();
            pythonize(py, &view).map_err(|e| e.into())
        }

        pub fn query_many<'py>(
            &self,
            py: Python<'py>,
            x_py: PyReadonlyArray1<Real>,
            y_py: PyReadonlyArray1<Real>,
            z_py: PyReadonlyArray1<Real>,
        ) -> Bound<'py, PyArray2<Real>> {
            let x = x_py.as_array();
            let y = y_py.as_array();
            let z = z_py.as_array();

            let results = py.detach(|| {
                let num_points = x.len();

                let mut out_array = Array2::zeros((num_points, 6));
                azip!((mut out_lane in out_array.rows_mut(), &xi in x, &yi in y, &zi in z) {
                    let query_point = Point3::new(xi, yi, zi);

                    let quality = self.inner.query(query_point).unwrap_or_else(|| {
                        panic!("Point {} outside of defined model layers", query_point);
                    });

                    out_lane[0] = quality.rho;
                    out_lane[1] = quality.vp;
                    out_lane[2] = quality.vs;
                    out_lane[3] = quality.qp;
                    out_lane[4] = quality.qs;
                    out_lane[5] = quality.alpha;
                });

                out_array
            });
            results.into_pyarray(py)
        }

        pub fn print_structure(&self) {
            self.inner.pretty_print();
        }
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyQuality>()?;
        m.add_class::<PyMeshModel>()?;
        m.add_class::<PyModel>()?;
        m.add_function(wrap_pyfunction!(mesh_model, m)?)?;
        m.add_function(wrap_pyfunction!(model_tree, m)?)?;

        Ok(())
    }
}
