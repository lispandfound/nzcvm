mod geometry;
pub mod mesh;
pub mod model;
pub mod quality;
mod real;
mod simplex;
mod surface;
mod tree_query;
use pyo3::prelude::*;

#[pymodule]
mod nzcvm {
    use crate::mesh::{Explanation, MeshModel};

    use crate::model::{ConstantModel, InterpolateModel, Model, ModelExplanation};
    use crate::quality::Quality;
    use crate::real::Real;
    use crate::simplex::Simplex;
    use nalgebra::{Point3, Point4};
    use ndarray::{azip, Array2, Axis};
    use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

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
    pub struct PySimplex {
        c0: PyPoint,
        c1: PyPoint,
        c2: PyPoint,
        c3: PyPoint,
        priority: u8,
    }

    impl From<Simplex> for PySimplex {
        fn from(item: Simplex) -> Self {
            // REFACTOR REQUIRED: Simplex now only stores c3 and inv_matrix.
            // You can either reconstruct c0, c1, c2 here or change the PySimplex struct.
            PySimplex {
                // c0: item.c0.into(), // ERROR: No field c0
                // c1: item.c1.into(), // ERROR: No field c1
                // c2: item.c2.into(), // ERROR: No field c2
                c0: item.c3.into(), // Placeholder
                c1: item.c3.into(), // Placeholder
                c2: item.c3.into(), // Placeholder
                c3: item.c3.into(),
                priority: item.priority,
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
    pub struct PyExplanation {
        pub simplices: Vec<PySimplex>,
        pub qualities: Vec<PyQuality>,
        pub models: Vec<PySimplexModel>,
        pub output: Option<PyQuality>,
        pub termination: Option<usize>,
    }

    impl From<Explanation> for PyExplanation {
        fn from(item: Explanation) -> Self {
            PyExplanation {
                simplices: item.simplices.into_iter().map(|x| x.into()).collect(),
                qualities: item.qualities.into_iter().map(|x| x.into()).collect(),
                models: item.models.into_iter().map(|x| x.into()).collect(),
                output: item.output.map(|x| x.into()),
                termination: item.termination,
            }
        }
    }

    #[pyclass]
    pub struct PyModel {
        pub inner: Arc<MeshModel>,
    }

    #[pyfunction]
    pub fn mesh_model(
        vertices_py: PyReadonlyArray2<Real>,
        faces_py: PyReadonlyArray2<usize>,
        types_py: PyReadonlyArray1<u8>,
        models_py: PyReadonlyArray1<usize>,
        qualities_py: PyReadonlyArray2<Real>,
        priorities_py: PyReadonlyArray1<u8>,
    ) -> PyResult<PyModel> {
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
        let priority = priorities_py.as_array().to_vec();
        Ok(PyModel {
            inner: Arc::new(MeshModel::new(
                vertices, faces, models_vec, qualities, priority,
            )),
        })
    }

    #[pymethods]
    impl PyModel {
        pub fn query(&self, x: Real, y: Real, z: Real) -> PyResult<Option<PyQuality>> {
            let pt = Point3::new(x, y, z);
            Ok(self.inner.query(pt).map(|q| q.into()))
        }

        pub fn explain(&self, x: Real, y: Real, z: Real) -> PyResult<PyExplanation> {
            let pt = Point3::new(x, y, z);
            Ok(self.inner.explain(pt).into())
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
        m.add_class::<PyModel>()?;
        m.add_function(wrap_pyfunction!(mesh_model, m)?)?;

        Ok(())
    }
}
