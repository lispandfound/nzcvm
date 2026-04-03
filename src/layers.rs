use crate::geometry::polygon_distance_sq;
use crate::quality::Quality;
use crate::surface::{Inclusion, Simplex, Surface, SurfacePoint};
use crate::tree_query::nearest_to_point_iterator;
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::Bvh;
use bvh::point_query::PointDistance;
use geo::Contains;
use geo::Intersects;
use geo::{Coord, Geometry, MapCoords, Polygon};
use geozero::wkb::Wkb;
use geozero::ToGeo;
use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{File, Group, Result};
use nalgebra::{Point2, Point3};
use ndarray::{Array1, Array2};
use ordered_float::OrderedFloat;
use scirs2_interpolate::interpnd::InterpolationMethod;
use scirs2_interpolate::ExtrapolateMode;
use std::collections::BTreeMap;
use std::iter::once;
use std::path::Path;

#[derive(Debug)]
pub struct LayerGeometry {
    pub id: usize,
    pub priority: usize,
    /// The new unified surface containing Top, Bottom, and Inclusion metadata
    pub surface: Surface,

    /// Still kept for the final precise distance check for Boundary/Outside cases
    bounds: Vec<f32>,
    z_abs_top: f32,
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
        bounds: &Polygon<f32>,
        surface_x: Array1<f32>,
        surface_y: Array1<f32>,
        surface_z_top: Array2<f32>,
        surface_z_bottom: Array2<f32>,
        // Note: interpolation method ignored as we are now strictly Linear (Barycentric)
        _method: InterpolationMethod,
        _extrapolate: ExtrapolateMode,
    ) -> Self {
        let nx = surface_x.len();
        let ny = surface_y.len();

        // 1. Create SurfacePoints (Elevations)
        let mut elevations = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                elevations.push(SurfacePoint {
                    top: surface_z_top[[j, i]],
                    bottom: surface_z_bottom[[j, i]],
                });
            }
        }

        // 2. Generate Triangles from the rectilinear grid
        let mut simplices = Vec::new();
        let mut vertex_map = Vec::new();
        let mut simplex_id = 0;

        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                let i00 = j * nx + i;
                let i10 = j * nx + (i + 1);
                let i01 = (j + 1) * nx + i;
                let i11 = (j + 1) * nx + (i + 1);

                let coords = [
                    Point2::new(surface_x[i], surface_y[j]),         // 0,0
                    Point2::new(surface_x[i + 1], surface_y[j]),     // 1,0
                    Point2::new(surface_x[i + 1], surface_y[j + 1]), // 1,1
                    Point2::new(surface_x[i], surface_y[j + 1]),     // 0,1
                ];

                // Split each grid cell into 2 triangles
                let tri_indices = [[i00, i10, i11], [i00, i11, i01]];
                let tri_coords = [
                    [coords[0], coords[1], coords[2]],
                    [coords[0], coords[2], coords[3]],
                ];

                for (idx, pts) in tri_indices.iter().zip(tri_coords.iter()) {
                    // Create a geo-polygon for this triangle to check against the bounds
                    let tri_poly = Polygon::new(
                        geo::LineString::from(vec![
                            (pts[0].x, pts[0].y),
                            (pts[1].x, pts[1].y),
                            (pts[2].x, pts[2].y),
                            (pts[0].x, pts[0].y),
                        ]),
                        vec![],
                    );

                    let mask = if bounds.contains(&tri_poly) {
                        Inclusion::Inside
                    } else if bounds.intersects(&tri_poly) {
                        Inclusion::Boundary
                    } else {
                        Inclusion::Outside
                    };

                    simplices.push(Simplex::new(pts[0], pts[1], pts[2], mask, simplex_id));
                    vertex_map.push(Point3::new(idx[0], idx[1], idx[2]));
                    simplex_id += 1;
                }
            }
        }

        let bvh_tree = Bvh::build(&mut simplices);
        let surface = Surface {
            bvh_tree,
            simplices,
            vertex_map,
            elevations,
        };

        // Flat bounds for distance calculation fallback
        let exterior = bounds.exterior();
        let mut flat_coords = Vec::with_capacity(exterior.0.len() * 2);
        for p in exterior.points() {
            flat_coords.push(p.x());
            flat_coords.push(p.y());
        }

        LayerGeometry {
            id: 0,
            priority: 0,
            surface,
            bounds: flat_coords,
            z_abs_top: *surface_z_top.iter().min_by(|a, b| a.total_cmp(b)).unwrap(),
            z_abs_bottom: *surface_z_bottom
                .iter()
                .max_by(|a, b| a.total_cmp(b))
                .unwrap(),
            node_index: 0,
        }
    }
}

impl PointDistance<f32, 3> for LayerGeometry {
    fn distance_squared(&self, query_point: Point3<f32>) -> f32 {
        let (z_top, z_bottom, inclusion) = match self.surface.query(query_point.xy()) {
            Some(res) => res,
            None => return f32::INFINITY,
        };

        let z_clamped = query_point.z.clamp(z_top, z_bottom);
        let dz_sq = (query_point.z - z_clamped).powi(2);

        let dxdy_sq = match inclusion {
            Inclusion::Inside => 0.0, // Guaranteed inside, skip O(N) check
            Inclusion::Boundary | Inclusion::Outside => {
                polygon_distance_sq(query_point, &self.bounds)
            }
        };
        dz_sq + dxdy_sq
    }
}

impl Bounded<f32, 3> for LayerGeometry {
    fn aabb(&self) -> Aabb<f32, 3> {
        let coords = self.bounds.as_slice();

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

pub struct LayerTree {
    bvh_tree: Bvh<f32, 3>,
    models: Vec<Model>,
    shapes: Vec<LayerGeometry>,
}

impl LayerTree {
    pub fn new(mut shapes: Vec<LayerGeometry>, models: Vec<Model>) -> Self {
        // Align shape ids with model ids.
        for (i, shape) in shapes.iter_mut().enumerate() {
            shape.id = i;
        }
        let bvh_tree = Bvh::build(&mut shapes);

        Self {
            shapes: shapes,
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

        // Symmetry check for distance squared
        let above = Point3::new(0.5, 0.5, -2.0); // 2 units above top
        let below = Point3::new(0.5, 0.5, 12.0); // 2 units below bottom
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

        // Note: z_abs_top (min) and z_abs_bottom (max)
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
        let prisms = vec![create_unit_prism(0.0, 10.0)];
        let models = vec![Model::Uniform(mock_quality(1.0))];

        // API Change: LayerTree::new now consumes the vectors
        let tree = LayerTree::new(prisms, models);

        // Point is 9 units away from the X=1.0 face.
        let far_point = Point3::new(10.0, 0.5, 5.0);
        let (q, dist) = tree
            .query(far_point)
            .expect("Should return nearest neighbour");

        assert_eq!(q.rho, 1.0);
        assert_abs_diff_eq!(dist, 9.0, epsilon = 1e-5);
    }

    #[test]
    fn test_model_tree_mapping_consistency() {
        let prisms = vec![create_unit_prism(0.0, 1.0), create_unit_prism(10.0, 11.0)];
        let models = vec![
            Model::Uniform(mock_quality(1.0)),
            Model::Uniform(mock_quality(2.0)),
        ];

        let tree = LayerTree::new(prisms, models);

        let p1 = Point3::new(0.5, 0.5, 0.5);
        let p2 = Point3::new(0.5, 0.5, 10.5);

        // API Change: query returns (Quality, f32)
        assert_eq!(tree.query(p1).unwrap().0.rho, 1.0);
        assert_eq!(tree.query(p2).unwrap().0.rho, 2.0);
    }

    #[test]
    fn test_overlap_priority_resolution() {
        let mut p1 = create_unit_prism(0.0, 10.0);
        p1.priority = 10;
        let mut p2 = create_unit_prism(0.0, 10.0);
        p2.priority = 20; // Higher priority should win

        let prisms = vec![p1, p2];
        let models = vec![
            Model::Uniform(mock_quality(1.0)),
            Model::Uniform(mock_quality(2.0)),
        ];

        let tree = LayerTree::new(prisms, models);
        let query_point = Point3::new(0.5, 0.5, 5.0);

        let (q, _) = tree.query(query_point).unwrap();
        assert_eq!(
            q.rho, 2.0,
            "Higher priority layer should be returned in case of overlap"
        );
    }

    #[test]
    fn test_sloped_surface_distance() {
        let poly = polygon![(x: 0.0, y: 0.0), (x: 2.0, y: 0.0), (x: 2.0, y: 2.0), (x: 0.0, y: 2.0)];
        let x = ndarray::Array1::from(vec![0.0, 2.0]);
        let y = ndarray::Array1::from(vec![0.0, 2.0]);

        // Top surface slopes from z=0 at x=0 to z=10 at x=2
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

        // Test vertical distance above the slope (at x=0, z_top=0)
        let p_start = Point3::new(0.0, 0.0, -5.0);
        assert_abs_diff_eq!(prism.distance_squared(p_start), 25.0, epsilon = 1e-5);

        // Test point inside the sloped volume (at x=1.0, interpolated z_top is 5.0)
        let p_mid = Point3::new(1.0, 1.0, 7.0);
        assert_eq!(prism.distance_squared(p_mid), 0.0);
    }
}
