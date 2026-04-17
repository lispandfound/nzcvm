pub mod geometry;
pub mod mesh;
pub mod quality;
pub mod real;
pub mod surface;
pub mod tree_query;
use pyo3::prelude::*;

#[pymodule]
mod nzcvm {
    use crate::mesh::{MeshModel, Model, ModelType};
    use crate::quality::Quality;
    use crate::real::Real;
    use nalgebra::{Point3, Point4};
    use ndarray::{Array2, Axis, Zip};
    use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use std::sync::Arc;

    #[pyclass]
    pub struct PyQuality {
        #[pyo3(get)]
        pub rho: Real,
        #[pyo3(get)]
        pub vp: Real,
        #[pyo3(get)]
        pub vs: Real,
        #[pyo3(get)]
        pub qp: Real,
        #[pyo3(get)]
        pub qs: Real,
        #[pyo3(get)]
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
            match (*model_type).try_into() {
                Ok(ModelType::Constant) => {
                    models_vec.push(Model::Constant {
                        quality: model_idx[idx],
                    });
                    idx += 1;
                }
                Ok(ModelType::Interpolate) => {
                    models_vec.push(Model::Interpolate {
                        qualities: Point4::new(
                            model_idx[idx],
                            model_idx[idx + 1],
                            model_idx[idx + 2],
                            model_idx[idx + 3],
                        ),
                    });
                    idx += 4;
                }
                Err(_) => {
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

            let num_points = x.len();

            let mut out_array = Array2::zeros((num_points, 5));

            Zip::from(out_array.rows_mut())
                .and(&x)
                .and(&y)
                .and(&z)
                .par_for_each(|mut out_lane, &xi, &yi, &zi| {
                    let query_point = Point3::new(xi, yi, zi);

                    let quality = self
                        .inner
                        .query(query_point)
                        .expect("Point outside of defined model layers");

                    out_lane[0] = quality.rho;
                    out_lane[1] = quality.vp;
                    out_lane[2] = quality.vs;
                    out_lane[3] = quality.qp;
                    out_lane[4] = quality.qs;
                });

            out_array.into_pyarray(py)
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
