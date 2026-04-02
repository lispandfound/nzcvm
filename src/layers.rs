use crate::geometry::polygon_distance_sq;
use crate::quality::Quality;
use crate::tree_query::nearest_to_point_iterator;
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::Bvh;
use bvh::point_query::PointDistance;
use geo::{Coord, Geometry, MapCoords, Polygon};
use geozero::wkb::Wkb;
use geozero::ToGeo;
use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{File, Group, Result};
use nalgebra::Point3;
use ndarray::{array, Array1, Array2};
use ordered_float::OrderedFloat;
use scirs2_interpolate::interpnd::{InterpolationMethod, RegularGridInterpolator};
use scirs2_interpolate::ExtrapolateMode;
use std::collections::BTreeMap;
use std::iter::once;
use std::path::Path;

#[derive(Debug)]
pub struct LayerGeometry {
    pub id: usize,
    pub top_surface: RegularGridInterpolator<f32>,
    pub bottom_surface: RegularGridInterpolator<f32>,
    pub priority: usize,

    /// Internal representation of the model boundary as an interleaved array of x-y points.
    bounds: Array1<f32>,
    /// Absolute top value of the surface
    z_abs_top: f32,
    /// Absolute bottom value of the surface
    z_abs_bottom: f32,
    node_index: usize,
}

impl LayerGeometry {
    pub fn new_with_flat_surface(bounds: &Polygon<f32>, z_top: f32, z_bottom: f32) -> Self {
        // Due to extrapolation, we can treat the "interpolation" onto a flat
        // surface at the top and bottom using nearest-neighbour interpolation
        // with just four points.
        let x = Array1::from(vec![0.0, 1.0]);
        let y = Array1::from(vec![0.0, 1.0]);
        let z_top_array = Array2::from_elem((2, 2), z_top);
        let z_bottom_array = Array2::from_elem((2, 2), z_bottom);
        LayerGeometry::new(
            bounds,
            x,
            y,
            z_top_array,
            z_bottom_array,
            InterpolationMethod::Nearest,
            ExtrapolateMode::Nearest,
        )
    }

    pub fn new(
        bounds: &Polygon<f32>, // Still take the geo Polygon as input
        surface_x: Array1<f32>,
        surface_y: Array1<f32>,
        surface_z_top: Array2<f32>,
        surface_z_bottom: Array2<f32>,
        surface_interpolation_method: InterpolationMethod,
        surface_extrapolation_mode: ExtrapolateMode,
    ) -> Self {
        let exterior = bounds.exterior();
        let mut flat_coords = Vec::with_capacity(exterior.0.len() * 2);
        for p in exterior.points() {
            flat_coords.push(p.x());
            flat_coords.push(p.y());
        }
        let bounds_array = Array1::from(flat_coords);

        let z_top = *surface_z_top.iter().min_by(|x, y| x.total_cmp(y)).unwrap();
        let z_bottom = *surface_z_bottom
            .iter()
            .max_by(|x, y| x.total_cmp(y))
            .unwrap();

        let rect_points = vec![surface_x, surface_y];
        let top_surface = RegularGridInterpolator::new(
            rect_points.clone(),
            surface_z_top.into_dyn(),
            surface_interpolation_method,
            surface_extrapolation_mode,
        )
        .unwrap();

        let bottom_surface = RegularGridInterpolator::new(
            rect_points,
            surface_z_bottom.into_dyn(),
            surface_interpolation_method,
            surface_extrapolation_mode,
        )
        .unwrap();

        LayerGeometry {
            bounds: bounds_array, // Stored as flat Array1<f32>
            top_surface,
            bottom_surface,
            z_abs_top: z_top,
            z_abs_bottom: z_bottom,
            id: 0,
            node_index: 0,
            priority: 0,
        }
    }
}

impl PointDistance<f32, 3> for LayerGeometry {
    fn distance_squared(&self, query_point: Point3<f32>) -> f32 {
        // 1. Surface Interpolation (as before)
        let query_array = array![[query_point.x, query_point.y]];
        let query_view = query_array.view();
        let z_top_res = self.top_surface.__call__(&query_view);
        let z_bottom_res = self.bottom_surface.__call__(&query_view);

        let z_projected = match (z_top_res, z_bottom_res) {
            (Ok(z_top), Ok(z_bottom)) => query_point.z.clamp(z_top[0], z_bottom[0]),
            _ => query_point.z,
        };

        let dz_sq = (query_point.z - z_projected).powi(2);

        let coords_slice = self.bounds.as_slice().unwrap();
        let dxdy_sq = polygon_distance_sq(query_point, coords_slice);

        dz_sq + dxdy_sq
    }
}

impl Bounded<f32, 3> for LayerGeometry {
    fn aabb(&self) -> Aabb<f32, 3> {
        let coords = self.bounds.as_slice().unwrap();

        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;

        for chunk in coords.chunks_exact(2) {
            let x = chunk[0];
            let y = chunk[1];

            if x < min_x {
                min_x = x;
            }
            if x > max_x {
                max_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if y > max_y {
                max_y = y;
            }
        }

        let min_point = Point3::new(min_x, min_y, self.z_abs_top);
        let max_point = Point3::new(max_x, max_y, self.z_abs_bottom);

        Aabb::with_bounds(min_point, max_point)
    }
}

pub fn deserialise_layer_geometry(group: &Group) -> Result<LayerGeometry> {
    let wkb_dataset = group.dataset("bounds")?;
    let wkb_bytes: Vec<u8> = wkb_dataset.read_raw()?;

    let priority = if group.link_exists("priority") {
        group.attr("priority")?.read_scalar()?
    } else {
        0
    };

    let geo_obj = Wkb(wkb_bytes).to_geo().map_err(|e| e.to_string())?;

    let bounds = match geo_obj {
        Geometry::Polygon(p) => p.map_coords(|c| Coord {
            x: c.x as f32,
            y: c.y as f32,
        }),
        _ => return Err("Bounds dataset is not a polygon".into()),
    };

    let x: Array1<f32> = group.dataset("surface_x")?.read_1d()?;
    let y: Array1<f32> = group.dataset("surface_y")?.read_1d()?;
    let z_top: Array2<f32> = group.dataset("surface_z_top")?.read_2d()?;
    let z_bottom: Array2<f32> = group.dataset("surface_z_bottom")?.read_2d()?;
    let mut geometry = LayerGeometry::new(
        &bounds,
        x,
        y,
        z_top,
        z_bottom,
        InterpolationMethod::Linear,
        ExtrapolateMode::Nearest,
    );
    geometry.priority = priority;
    Ok(geometry)
}

pub fn deserialise_model(group: &Group) -> Result<Model> {
    let model_type: VarLenUnicode = group.attr("model_type")?.read_scalar()?;

    match model_type.as_str() {
        "uniform" => {
            let q = Quality {
                rho: group.attr("rho")?.read_scalar()?,
                vp: group.attr("vp")?.read_scalar()?,
                vs: group.attr("vs")?.read_scalar()?,
                qp: group.attr("qp")?.read_scalar()?,
                qs: group.attr("qs")?.read_scalar()?,
            };
            Ok(Model::Uniform(q))
        }
        "layered" => {
            let data: Array2<f32> = group.dataset("layers")?.read_2d()?;
            let mut layers = BTreeMap::new();

            for row in data.axis_iter(ndarray::Axis(0)) {
                let z = row[0];
                let quality = Quality {
                    rho: row[1],
                    vp: row[2],
                    vs: row[3],
                    qp: row[4],
                    qs: row[5],
                };
                layers.insert(OrderedFloat(z), quality);
            }
            Ok(Model::Layered { layers })
        }
        _ => Err(format!("Unknown model type: {}", model_type.as_str()).into()),
    }
}

pub fn read_model_data<P: AsRef<Path>>(path: P) -> Result<(LayerGeometry, Model)> {
    let file = File::open(path)?;

    let geo_group = file.group("geometry")?;
    let model_group = file.group("model")?;

    let geometry = deserialise_layer_geometry(&geo_group)?;
    let model = deserialise_model(&model_group)?;

    Ok((geometry, model))
}

impl BHShape<f32, 3> for LayerGeometry {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

pub enum Model {
    /// Simple uniform qualities.
    Uniform(Quality),
    /// 1D velocity model layering (common in basins).
    Layered {
        layers: BTreeMap<OrderedFloat<f32>, Quality>,
    },
}

impl Model {
    pub fn query(&self, point: Point3<f32>) -> Option<Quality> {
        match self {
            Self::Uniform(quality) => Some(*quality),
            Self::Layered { layers } => layers
                .range(..=OrderedFloat(point.z))
                .next_back()
                .map(|(_, &q)| q),
        }
    }
}

pub struct LayerTree<'a> {
    bvh_tree: Bvh<f32, 3>,
    models: &'a [Model],
    shapes: &'a [LayerGeometry],
}

impl<'a> LayerTree<'a> {
    pub fn new(shapes: &'a mut [LayerGeometry], models: &'a [Model]) -> Self {
        // Align shape ids with model ids.
        for (i, shape) in shapes.iter_mut().enumerate() {
            shape.id = i;
        }
        let bvh_tree = Bvh::build(shapes);

        Self {
            shapes,
            bvh_tree: bvh_tree,
            models: models,
        }
    }

    pub fn query(&self, point: Point3<f32>) -> Option<(Quality, f32)> {
        let mut iter = nearest_to_point_iterator(&self.bvh_tree, &self.shapes, &point);

        iter.next().and_then(|(shape, dist)| {
            if dist < f32::EPSILON {
                // Shape contains point, check for other shapes containing this point to resolve overlaps.
                let other_shapes = iter
                    // short-cut: no more than two extra models considered
                    .take_while(|(_, dist)| *dist < f32::EPSILON)
                    .map(|(shape, _)| shape);
                // The preferred shape is the highest priority shape.
                once(shape)
                    .chain(other_shapes)
                    .max_by_key(|shape| shape.priority)
                    .and_then(|best_shape| {
                        self.model_query_for(best_shape, point).map(|q| (q, dist))
                    })
            } else {
                self.model_query_for(shape, point).map(|q| (q, dist))
            }
        })
    }

    fn model_query_for(&self, shape: &LayerGeometry, point: Point3<f32>) -> Option<Quality> {
        self.models[shape.id].query(point)
    }

    pub fn pretty_print(&self) -> () {
        println!(
            "Disjoint layer models, having {} layers with structure:",
            self.models.len()
        );
        self.bvh_tree.pretty_print();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use geo::polygon;
    use nalgebra::Point3;
    use ordered_float::OrderedFloat;
    // --- Helpers ---

    fn create_unit_prism(z_top: f32, z_bottom: f32) -> LayerGeometry {
        let poly = polygon![
            (x: 0.0, y: 0.0),
            (x: 1.0, y: 0.0),
            (x: 1.0, y: 1.0),
            (x: 0.0, y: 1.0),
        ];
        LayerGeometry::new_with_flat_surface(&poly, z_top, z_bottom)
    }

    fn mock_quality(val: f32) -> Quality {
        Quality {
            rho: val,
            vp: val,
            vs: val,
            qp: val,
            qs: val,
        }
    }

    // --- Prism Geometry Tests ---

    #[test]
    fn test_prism_distance_invariants() {
        let prism = create_unit_prism(0.0, 10.0);

        let inside = Point3::new(0.5, 0.5, 5.0);
        assert_eq!(prism.distance_squared(inside), 0.0);

        let outside = Point3::new(-1.0, -1.0, -1.0);
        assert!(prism.distance_squared(outside) >= 0.0);

        let above = Point3::new(0.5, 0.5, -2.0);
        let below = Point3::new(0.5, 0.5, 12.0);
        assert_abs_diff_eq!(
            prism.distance_squared(above),
            prism.distance_squared(below),
            epsilon = f32::EPSILON
        );
    }

    #[test]
    fn test_aabb_containment_guarantee() {
        let prism = create_unit_prism(5.0, 15.0);
        let aabb = prism.aabb();

        assert!(aabb.min.z <= 5.0);
        assert!(aabb.max.z >= 15.0);
        assert!(aabb.min.x <= 0.0 && aabb.max.x >= 1.0);
        assert!(aabb.min.y <= 0.0 && aabb.max.y >= 1.0);
    }

    // --- Model Query Tests ---

    #[test]
    fn test_layered_model_stepping() {
        let mut layers = BTreeMap::new();
        layers.insert(OrderedFloat(0.0), mock_quality(10.0));
        layers.insert(OrderedFloat(100.0), mock_quality(20.0));

        let model = Model::Layered { layers };

        assert_eq!(model.query(Point3::new(0.0, 0.0, 100.0)).unwrap().rho, 20.0);
        assert_eq!(model.query(Point3::new(0.0, 0.0, 50.0)).unwrap().rho, 10.0);
        assert!(model.query(Point3::new(0.0, 0.0, -10.0)).is_none());
    }

    // --- LayerTree Integration Tests ---

    #[test]
    fn test_model_tree_nearest_neighbor_behavior() {
        let mut prisms = vec![create_unit_prism(0.0, 10.0)];
        let models = vec![Model::Uniform(mock_quality(1.0))];

        let tree = LayerTree::new(&mut prisms, &models);

        // Property: A point far outside now returns the nearest result + the actual distance
        let far_point = Point3::new(10.0, 0.0, 5.0); // 9 units away from the X=1 face
        let (q, dist) = tree
            .query(far_point)
            .expect("Should return nearest neighbor");

        assert_eq!(q.rho, 1.0);
        assert_abs_diff_eq!(dist, 9.0, epsilon = 1e-5);
    }

    #[test]
    fn test_model_tree_mapping_consistency() {
        let mut prisms = vec![create_unit_prism(0.0, 1.0), create_unit_prism(10.0, 11.0)];
        let models = vec![
            Model::Uniform(mock_quality(1.0)),
            Model::Uniform(mock_quality(2.0)),
        ];

        let tree = LayerTree::new(&mut prisms, &models);

        let p1 = Point3::new(0.5, 0.5, 0.5);
        let p2 = Point3::new(0.5, 0.5, 10.5);

        assert_eq!(tree.query(p1).unwrap().0.rho, 1.0);
        assert_eq!(tree.query(p2).unwrap().0.rho, 2.0);
    }

    #[test]
    fn test_sloped_surface_distance() {
        let poly = polygon![(x: 0.0, y: 0.0), (x: 2.0, y: 0.0), (x: 2.0, y: 2.0), (x: 0.0, y: 2.0)];
        let x = ndarray::Array1::from(vec![0.0, 2.0]);
        let y = ndarray::Array1::from(vec![0.0, 2.0]);
        let z_top = ndarray::array![[0.0, 0.0], [10.0, 10.0]];
        let z_bottom = ndarray::Array2::from_elem((2, 2), 20.0);

        let prism = LayerGeometry::new(
            &poly,
            x,
            y,
            z_top,
            z_bottom,
            scirs2_interpolate::interpnd::InterpolationMethod::Linear,
            scirs2_interpolate::ExtrapolateMode::Nearest,
        );

        let p_start = Point3::new(0.0, 0.0, -5.0);
        assert_abs_diff_eq!(prism.distance_squared(p_start), 25.0, epsilon = 1e-5);

        let p_mid = Point3::new(1.0, 1.0, 5.0);
        assert_eq!(prism.distance_squared(p_mid), 0.0);
    }

    #[test]
    fn test_model_interpolation_failure_fallback() {
        let prism = create_unit_prism(0.0, 10.0);
        let p = Point3::new(2.0, 0.5, 5.0);
        let dist_sq = prism.distance_squared(p);
        assert_abs_diff_eq!(dist_sq, 1.0, epsilon = 1e-5);
    }
}
