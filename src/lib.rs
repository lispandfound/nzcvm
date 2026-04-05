pub mod geometry;
pub mod geomodelgrid;
pub mod layers;
pub mod mesh;
pub mod model;
pub mod quality;
pub mod surface;
pub mod tree_query;
pub mod writer;
pub mod coordinates;
pub mod generate;
use pyo3::prelude::*;

#[pymodule]
mod nzcvm {
    use crate::layers::{read_model_data, LayerGeometry, LayerTree, Model};
    use crate::mesh::{load_mesh_from_hdf5, MeshModel};
    use crate::model::ModelTree;
    use crate::quality::Quality;
    use bvh::aabb::Bounded;
    use geo::{Coord, LineString, Polygon};
    use nalgebra::Point3;
    use ndarray::Array2;
    use ndarray::Zip;
    use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use ordered_float::OrderedFloat;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use std::collections::BTreeMap;

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
    pub struct PyModel {
        pub inner: Arc<ModelTree>,
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

        pub fn stack(&self, other: &Self) -> Self {
            Self {
                inner: Arc::new(ModelTree::Stack(self.inner.clone(), other.inner.clone())),
            }
        }

        fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            fn aux<'py>(py: Python<'py>, model_tree: &ModelTree) -> PyResult<Bound<'py, PyDict>> {
                let dict = PyDict::new(py);

                match model_tree {
                    ModelTree::Stack(left, right) => {
                        dict.set_item("type", "stack")?;
                        dict.set_item("left", aux(py, left)?)?;
                        dict.set_item("right", aux(py, right)?)?;
                        Ok(dict)
                    },
                    ModelTree::Blend{left, right, distance} => {
                        dict.set_item("type", "blended")?;
                        dict.set_item("left", aux(py, left)?)?;
                        dict.set_item("right", aux(py, right)?)?;
                        dict.set_item("distance", distance)?;
                        Ok(dict)
                    },

                    ModelTree::Layers {layer_tree} => {
                        dict.set_item("type", "layered_models")?;
                        let aabb_tuples: Vec<(f32, f32, f32, f32, f32, f32)> = layer_tree.bounds().iter().map(|aabb| (aabb.min.x, aabb.min.y, aabb.min.z, aabb.max.x, aabb.max.y, aabb.max.z)).collect();
                        let priorities = layer_tree.priorities();
                        dict.set_item("bounds", aabb_tuples)?;
                        dict.set_item("priorities", priorities)?;
                        Ok(dict)
                    },
                    ModelTree::Mesh {mesh_model} => {
                        let aabb = mesh_model.aabb();
                        let aabb_tuple = (aabb.min.x, aabb.min.y, aabb.min.z, aabb.max.x, aabb.max.y, aabb.max.z);
                        dict.set_item("type", "mesh")?;
                        dict.set_item("bounds", aabb_tuple)?;
                        dict.set_item("points", mesh_model.points())?;
                        Ok(dict)
                    }

                }
            }
            aux(py, &self.inner)
        }
    }

    #[pyfunction]
    pub fn mesh(
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
        let mesh = MeshModel::curvilinear_mesh(vertices, qualities, dimensions, chart);
        Ok(PyModel {
            inner: Arc::new(ModelTree::mesh_model(mesh)),
        })
    }

    /// Loads a mesh model from HDF5
    #[pyfunction]
    pub fn load_mesh(path: &str) -> PyResult<PyModel> {
        let mesh = load_mesh_from_hdf5(path).map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PyModel {
            inner: Arc::new(ModelTree::mesh_model(mesh)),
        })
    }

    /// Loads all .h5 files from a directory as a LayerTree
    #[pyfunction]
    pub fn load_layers_from_dir(directory: &str) -> PyResult<PyModel> {
        let mut basins = Vec::new();
        let mut models = Vec::new();

        let paths =
            std::fs::read_dir(directory).map_err(|e| PyValueError::new_err(e.to_string()))?;

        for entry in paths.flatten() {
            if entry.path().extension().is_some_and(|ext| ext == "h5")
                && let Ok((geo, mod_data)) = read_model_data(entry.path()) {
                    basins.push(geo);
                    models.push(mod_data);
                }
        }

        let layer_tree = LayerTree::new(basins, models);

        Ok(PyModel {
            inner: Arc::new(ModelTree::layered_model(layer_tree)),
        })
    }

    #[pyfunction]
    pub fn create_layer_model(
        bounds_py: PyReadonlyArray2<f32>,
        surface_x_py: PyReadonlyArray2<f32>,
        surface_y_py: PyReadonlyArray2<f32>,
        z_top_py: PyReadonlyArray2<f32>,
        z_bottom_py: PyReadonlyArray2<f32>,
        layer_params_py: PyReadonlyArray2<f32>,
        priority: usize,
    ) -> PyResult<PyModel> {
        let bounds_arr = bounds_py.as_array();
        let coords: Vec<Coord<f32>> = bounds_arr
            .rows()
            .into_iter()
            .map(|r| Coord { x: r[0], y: r[1] })
            .collect();

        let poly = Polygon::new(LineString::new(coords), vec![]);

        let mut geometry = LayerGeometry::build(
            &poly,
            surface_x_py.as_array(),
            surface_y_py.as_array(),
            z_top_py.as_array(),
            z_bottom_py.as_array(),
        );
        geometry.priority = priority;

        let params_arr = layer_params_py.as_array();
        let mut layers = BTreeMap::new();
        for row in params_arr.rows() {
            layers.insert(
                OrderedFloat(row[0]),
                Quality {
                    rho: row[1],
                    vp: row[2],
                    vs: row[3],
                    qp: row[4],
                    qs: row[5],
                },
            );
        }
        let model = Model::Layered { layers };

        let layer_tree = LayerTree::new(vec![geometry], vec![model]);

        Ok(PyModel {
            inner: Arc::new(ModelTree::layered_model(layer_tree)),
        })
    }
}
