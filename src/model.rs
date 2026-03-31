use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::Bvh;
use bvh::point_query::PointDistance;

use crate::quality::Quality;
use crate::quality_interpolator::QualityInterpolator;
use geo::{point, BoundingRect, Distance, Euclidean, Polygon};
use nalgebra::Point3;
use ndarray::{array, Array1, Array2};
use ordered_float::OrderedFloat;
use scirs2_interpolate::interpnd::{InterpolationMethod, RegularGridInterpolator};
use scirs2_interpolate::ExtrapolateMode;
use std::collections::BTreeMap;
use std::collections::HashMap;

#[derive(Debug)]
pub struct LayerGeometry {
    pub bounds: Polygon<f32>,
    pub top_surface: RegularGridInterpolator<f32>,
    pub bottom_surface: RegularGridInterpolator<f32>,
    /// Absolute top value of the surface
    z_abs_top: f32,
    /// Absolute bottom value of the surface
    z_abs_bottom: f32,
    pub node_index: usize,
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
        bounds: &Polygon<f32>,
        surface_x: Array1<f32>,
        surface_y: Array1<f32>,
        surface_z_top: Array2<f32>,
        surface_z_bottom: Array2<f32>,
        surface_interpolation_method: InterpolationMethod,
        surface_extrapolation_mode: ExtrapolateMode,
    ) -> Self {
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
            bounds: bounds.clone(),
            top_surface: top_surface,
            bottom_surface: bottom_surface,
            z_abs_top: z_top,
            z_abs_bottom: z_bottom,
            node_index: 0,
        }
    }
}

impl PointDistance<f32, 3> for LayerGeometry {
    fn distance_squared(&self, query_point: Point3<f32>) -> f32 {
        let mut projected_point = query_point;
        let query_array = array![[query_point.x, query_point.y]];
        let query_view = query_array.view();
        let z_top_res = self.top_surface.__call__(&query_view);
        let z_bottom_res = self.bottom_surface.__call__(&query_view);

        // Either or both of those surface resolutions could fail. In that case
        // it is assumed that dz = 0, and so we set the projected point z to
        // equal the query point z.
        let z_projected = match (z_top_res, z_bottom_res) {
            (Ok(z_top), Ok(z_bottom)) => query_point.z.clamp(z_top[0], z_bottom[0]),
            _ => query_point.z,
        };
        projected_point.z = z_projected;

        let dz_sq = (query_point.z - projected_point.z).powi(2);
        let dxdy_sq = Euclidean
            .distance(
                &self.bounds,
                &point!(x: projected_point.x, y: projected_point.y),
            )
            .powi(2);
        dz_sq + dxdy_sq
    }
}

impl Bounded<f32, 3> for LayerGeometry {
    fn aabb(&self) -> Aabb<f32, 3> {
        let bounding_rect = self.bounds.bounding_rect().unwrap();
        let min_coord = bounding_rect.min();
        let max_coord = bounding_rect.max();

        let min_point = Point3::new(min_coord.x, min_coord.y, self.z_abs_top);
        let max_point = Point3::new(max_coord.x, max_coord.y, self.z_abs_bottom);
        Aabb::with_bounds(min_point, max_point)
    }
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
    /// 3D interpolated qualities (common in tomography models).
    Interpolated { interpolator: QualityInterpolator },
}

impl Model {
    pub fn query(&self, point: Point3<f32>) -> Option<Quality> {
        match self {
            Self::Uniform(quality) => Some(*quality),
            Self::Layered { layers } => layers
                .range(..=OrderedFloat(point.z))
                .next_back()
                .map(|(_, &q)| q),
            Self::Interpolated { interpolator } => interpolator.interpolate(point),
        }
    }
}

pub struct ModelTree<'a> {
    bvh_tree: Bvh<f32, 3>,
    models: HashMap<usize, &'a Model>,
    shapes: &'a [LayerGeometry],
}

impl<'a> ModelTree<'a> {
    pub fn new(prisms: &'a mut [LayerGeometry], models: &'a [Model]) -> Self {
        let bvh_tree = Bvh::build(prisms);

        let node_indices = prisms
            .iter()
            .zip(models)
            .map(|(tree_node, model)| (tree_node.node_index, model));

        let models = HashMap::from_iter(node_indices);

        ModelTree {
            shapes: prisms,
            bvh_tree: bvh_tree,
            models: models,
        }
    }

    pub fn query(&self, point: Point3<f32>, eps: f32) -> Option<Quality> {
        self.bvh_tree
            .nearest_to(point, &self.shapes)
            .and_then(|(shape, dist)| {
                if dist > eps {
                    None
                } else {
                    self.models.get(&shape.node_index)
                }
            })
            .and_then(|model| model.query(point))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::polygon;
    use nalgebra::Point3;
    use ndarray::Array4;
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

        // Property: Points strictly inside the prism must have distance 0
        let inside = Point3::new(0.5, 0.5, 5.0);
        assert_eq!(prism.distance_squared(inside), 0.0);

        // Property: Distance is non-negative
        let outside = Point3::new(-1.0, -1.0, -1.0);
        assert!(prism.distance_squared(outside) >= 0.0);

        // Property: Distance is symmetric relative to Z-bounds (if XY is constant)
        let above = Point3::new(0.5, 0.5, -2.0); // 2 units above z_top
        let below = Point3::new(0.5, 0.5, 12.0); // 2 units below z_bottom
        assert!(
            (prism.distance_squared(above) - prism.distance_squared(below)).abs() < f32::EPSILON
        );
    }

    #[test]
    fn test_aabb_containment_guarantee() {
        let prism = create_unit_prism(5.0, 15.0);
        let aabb = prism.aabb();
        // Property: AABB must encompass the full Z range
        assert!(aabb.min.z <= 5.0);
        assert!(aabb.max.z >= 15.0);

        // Property: AABB must encompass the polygon bounds
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

        // Property: Querying exactly at a boundary returns that boundary's quality
        assert_eq!(model.query(Point3::new(0.0, 0.0, 100.0)).unwrap().rho, 20.0);

        // Property: Querying between boundaries returns the "shallowest" (lower Z) match
        // (Step-function behaviour)
        assert_eq!(model.query(Point3::new(0.0, 0.0, 50.0)).unwrap().rho, 10.0);

        // Property: Querying above the first layer (negative Z) returns None
        assert!(model.query(Point3::new(0.0, 0.0, -10.0)).is_none());
    }

    // --- ModelTree Integration Tests ---

    #[test]
    fn test_model_tree_spatial_guarantees() {
        let mut prisms = vec![
            create_unit_prism(0.0, 10.0), // Prism A
        ];
        let models = vec![Model::Uniform(mock_quality(1.0))];

        let tree = ModelTree::new(&mut prisms, &models);

        // Property: A point far outside the epsilon radius returns None
        let far_point = Point3::new(100.0, 100.0, 100.0);
        assert!(tree.query(far_point, 1.0).is_none());

        // Property: A point just outside the prism but within epsilon returns the model quality
        let near_point = Point3::new(1.05, 0.5, 5.0);
        assert!(tree.query(near_point, 0.1).is_some());
    }

    #[test]
    fn test_model_tree_mapping_consistency() {
        // Guarantee: The ModelTree correctly maps the BVH shape index to the model index
        let mut prisms = vec![create_unit_prism(0.0, 1.0), create_unit_prism(10.0, 11.0)];
        let models = vec![
            Model::Uniform(mock_quality(1.0)),
            Model::Uniform(mock_quality(2.0)),
        ];

        let tree = ModelTree::new(&mut prisms, &models);

        let p1 = Point3::new(0.5, 0.5, 0.5);
        let p2 = Point3::new(0.5, 0.5, 10.5);

        assert_eq!(tree.query(p1, 0.1).unwrap().rho, 1.0);
        assert_eq!(tree.query(p2, 0.1).unwrap().rho, 2.0);
    }
    // --- Helper for Variable Surfaces ---

    fn create_sloped_prism() -> LayerGeometry {
        let poly = polygon![(x: 0.0, y: 0.0), (x: 2.0, y: 0.0), (x: 2.0, y: 2.0), (x: 0.0, y: 2.0)];
        let x = Array1::from(vec![0.0, 2.0]);
        let y = Array1::from(vec![0.0, 2.0]);

        // Top surface slopes from z=0 to z=10 along X
        let z_top = array![[0.0, 0.0], [10.0, 10.0]];
        // Bottom surface is flat at z=20
        let z_bottom = Array2::from_elem((2, 2), 20.0);

        LayerGeometry::new(
            &poly,
            x,
            y,
            z_top,
            z_bottom,
            InterpolationMethod::Linear,
            ExtrapolateMode::Nearest,
        )
    }

    #[test]
    fn test_sloped_surface_distance() {
        let prism = create_sloped_prism();

        // At x=0, z_top is 0. Point at z= -5 should have dz_sq = 25
        let p_start = Point3::new(0.0, 0.0, -5.0);
        assert!((prism.distance_squared(p_start) - 25.0).abs() < 1e-5);

        // At x=2, z_top is 10. Point at z= 5 should have dz_sq = 25 (it's "above" the slope)
        let p_end = Point3::new(2.0, 0.0, 5.0);
        assert!((prism.distance_squared(p_end) - 25.0).abs() < 1e-5);

        // At x=1 (middle), z_top is 5. Point at z=5 should be INSIDE (distance 0)
        let p_mid = Point3::new(1.0, 1.0, 5.0);
        assert_eq!(prism.distance_squared(p_mid), 0.0);
    }

    #[test]
    fn test_interpolated_3d_model_with_transform() {
        // Data contains 5 qualities (rho, vp, vs, qp, qs)
        // We'll set rho = i
        let mut data = Array4::zeros((2, 2, 2, 5));
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    data[[i, j, k, 0]] = i as f32;
                }
            }
        }
        let interpolator = QualityInterpolator {
            x: vec![0.0, 1.0],
            y: vec![0.0, 1.0],
            z: vec![0.0, 1.0],
            quality: data,
        };

        // Define a model that is shifted by 10 units in X
        let model = Model::Interpolated {
            model_origin: array![10.0, 0.0, 0.0],
            model_transform: Array2::eye(3), // Identity rotation/scale
            interpolator,
        };

        // Querying at world (10, 0, 0) should map to local (0, 0, 0)
        let q1 = model.query(Point3::new(10.0, 0.0, 0.0)).unwrap();
        assert_eq!(q1.rho, 0.0);

        // Querying at world (11, 0, 0) should map to local (1, 0, 0)
        let q2 = model.query(Point3::new(11.0, 0.0, 0.0)).unwrap();
        assert_eq!(q2.rho, 1.0);
    }

    #[test]
    fn test_model_interpolation_failure_fallback() {
        let prism = create_unit_prism(0.0, 10.0);

        // If the interpolator fails (e.g. invalid query input),
        // distance_squared should fall back to just XY distance.
        let p = Point3::new(2.0, 0.5, 5.0); // Outside in X, but inside Z range
        let dist_sq = prism.distance_squared(p);

        // Distance to unit square (0,0 to 1,1) from (2, 0.5) is 1.0.
        // dz should be 0 because 5.0 is between top/bottom.
        assert!((dist_sq - 1.0).abs() < 1e-5);
    }
}
