mod geometry;
mod geomodelgrid;
mod layers;
mod mesh;
mod model;
mod quality;
mod rfile;
mod tree_query;
use pyo3::prelude::*;
#[pymodule]
mod nzcvm {
    use crate::layers::{read_model_data, LayerTree};
    use crate::mesh::{load_mesh_from_hdf5, MeshModel};
    use crate::model::ModelTree;
    use crate::quality::Quality;
    use nalgebra::Point3;
    use ndarray::Array2;
    use ndarray::Zip;
    use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use std::sync::Arc;

    #[pyclass]
    pub struct PyQuality {
        #[pyo3(get)]
        pub rho: f32,
        #[pyo3(get)]
        pub vp: f32,
        #[pyo3(get)]
        pub vs: f32,
        #[pyo3(get)]
        pub qp: f32,
        #[pyo3(get)]
        pub qs: f32,
    }

    impl From<Quality> for PyQuality {
        fn from(item: Quality) -> Self {
            PyQuality {
                rho: item.rho,
                vp: item.vp,
                vs: item.vs,
                qp: item.qp,
                qs: item.qs,
            }
        }
    }

    #[pyclass]
    pub struct PyModel<'py> {
        pub inner: ModelTree<'py>,
    }

    #[pymethods]
    impl PyModel {
        pub fn query(&self, x: f32, y: f32, z: f32) -> PyResult<Option<(PyQuality, f32)>> {
            let pt = Point3::new(x, y, z);
            Ok(self.inner.query(pt).map(|(q, d)| (q.into(), d)))
        }
        pub fn query_many<'py>(
            &self,
            py: Python<'py>,
            x_py: PyReadonlyArray1<f32>,
            y_py: PyReadonlyArray1<f32>,
            z_py: PyReadonlyArray1<f32>,
        ) -> Bound<'py, PyArray2<f32>> {
            let x = x_py.as_array();
            let y = y_py.as_array();
            let z = z_py.as_array();

            let num_points = x.len();

            let mut out_array = Array2::zeros((num_points, 6));

            Zip::from(out_array.rows_mut())
                .and(&x)
                .and(&y)
                .and(&z)
                .par_for_each(|mut out_lane, &xi, &yi, &zi| {
                    let query_point = Point3::new(xi, yi, zi);

                    let (quality, dist) = self
                        .inner
                        .query(query_point)
                        .expect("Point outside of defined model layers");

                    out_lane[0] = quality.rho;
                    out_lane[1] = quality.vp;
                    out_lane[2] = quality.vs;
                    out_lane[3] = quality.qp;
                    out_lane[4] = quality.qs;
                    out_lane[5] = dist;
                });

            out_array.into_pyarray(py)
        }

        pub fn print_structure(&self) {
            self.inner.pretty_print();
        }
    }

    #[pyfunction]
    pub fn layer_model<'py>(
        py: Python<'py>,
        vertices_py: PyReadonlyArray2<f32>,
        qualities_py: PyReadonlyArray2<f32>,
        dimensions: (usize, usize, usize),
    ) -> PyResult<PyModel<'py>> {
        let (_, nj, nk) = dimensions;
        let chart = |i, j, k| k + j * nk + i * nj * nk;
        let vertices: Vec<Point3<f32>> = vertices_py
            .as_array()
            .rows()
            .into_iter()
            .map(|row| Point3::from_slice(row.as_slice().unwrap()))
            .collect();
        let qualities: Vec<Quality> = qualities_py
            .as_array()
            .rows()
            .into_iter()
            .map(|row| Quality {
                rho: row[0],
                vp: row[1],
                vs: row[2],
                qp: row[3],
                qs: row[4],
            })
            .collect();
        let mesh = MeshModel::curvilinear_mesh(&vertices, &qualities, dimensions, chart);
        Ok(PyModel {
            inner: ModelTree::mesh_model(mesh),
        })
    }

    #[pymethods]
    impl ModelBuilder {
        #[new]
        pub fn new() -> Self {
            ModelBuilder {}
        }

        pub fn mesh(
            &self,
            vertices_py: PyReadonlyArray2<f32>,
            qualities_py: PyReadonlyArray2<f32>,
            dimensions: (usize, usize, usize),
        ) -> PyResult<PyModel> {
            let (_, nj, nk) = dimensions;
            let chart = |i, j, k| k + j * nk + i * nj * nk;
            let vertices: Vec<Point3<f32>> = vertices_py
                .as_array()
                .rows()
                .into_iter()
                .map(|row| Point3::from_slice(row.as_slice().unwrap()))
                .collect();
            let qualities: Vec<Quality> = qualities_py
                .as_array()
                .rows()
                .into_iter()
                .map(|row| Quality {
                    rho: row[0],
                    vp: row[1],
                    vs: row[2],
                    qp: row[3],
                    qs: row[4],
                })
                .collect();
            let mesh = MeshModel::curvilinear_mesh(&vertices, &qualities, dimensions, chart);
            Ok(PyModel {
                inner: Arc::new(ModelTree::mesh_model(unsafe { std::mem::transmute(mesh) })),
                _vertices: Some(vertices),
                _qualities: Some(qualities),
            })
        }

        /// Loads a mesh model from HDF5
        pub fn load_mesh(&self, path: &str) -> PyResult<PyModel> {
            let mut vertices = Vec::new();
            let mut qualities = Vec::new();

            // We use 'unsafe' or transmute lifetimes here only because we are
            // tying the lifetime of the MeshModel to the PyModel struct that owns the Vecs.
            let mesh = load_mesh_from_hdf5(path, &mut vertices, &mut qualities)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            Ok(PyModel {
                inner: Arc::new(ModelTree::mesh_model(mesh)),
            })
        }

        /// Loads all .h5 files from a directory as a LayerTree
        pub fn load_layers_from_dir(&self, directory: &str) -> PyResult<PyModel> {
            let mut basins = Vec::new();
            let mut models = Vec::new();

            let paths =
                std::fs::read_dir(directory).map_err(|e| PyValueError::new_err(e.to_string()))?;

            for entry in paths.flatten() {
                if entry.path().extension().map_or(false, |ext| ext == "h5") {
                    if let Ok((geo, mod_data)) = read_model_data(entry.path()) {
                        basins.push(geo);
                        models.push(mod_data);
                    }
                }
            }

            let layer_tree = LayerTree::new(&mut basins, &models);

            Ok(PyModel {
                inner: Arc::new(ModelTree::layered_model(unsafe {
                    std::mem::transmute(layer_tree)
                })),
                _vertices: None,
                _qualities: None,
            })
        }

        /// Stacks two models: upper on top of lower
        pub fn stack(&self, upper: &PyModel, lower: &PyModel) -> PyModel {
            PyModel {
                // This is where the recursive Enum logic happens
                inner: Arc::new(ModelTree::Stack(
                    unsafe { std::mem::transmute(&*upper.inner) },
                    unsafe { std::mem::transmute(&*lower.inner) },
                )),
                // Keep references to parent buffers to prevent dropping
                _vertices: None,
                _qualities: None,
            }
        }
    }
}

#[pyclass]
pub struct PyMeshModel {
    vertices: Vec<Point3<f32>>,
    qualities: Vec<Point3<f32>>,
}
