use crate::geometry::{self, closest_point_to_line};
use crate::quality::Quality;
use crate::real::Real;
use crate::surface::{Inclusion, Simplex, Surface, SurfacePoint};
use crate::tree_query::nearest_to_point_iterator;
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::Bvh;
use bvh::point_query::PointDistance;
use geo::{coord, point, BoundingRect, Contains, PreparedGeometry, Relate, Triangle};
use geo::{Coord, Geometry, MapCoords, Polygon};
use geozero::wkb::Wkb;
use geozero::ToGeo;
use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{File, Group, Result};
use nalgebra::{Point2, Point3};
use ndarray::{array, Array2, ArrayView2};
use ordered_float::OrderedFloat;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct LineShape {
    pub start: Point2<Real>,
    pub end: Point2<Real>,

    pub node_index: usize,
}

impl From<geo::Line<Real>> for LineShape {
    fn from(line: geo::Line<Real>) -> Self {
        let a = Point2::new(line.start.x, line.start.y);
        let b = Point2::new(line.end.x, line.end.y);
        Self {
            start: a,
            end: b,
            node_index: 0,
        }
    }
}

impl Bounded<Real, 2> for LineShape {
    fn aabb(&self) -> Aabb<Real, 2> {
        let min_x = self.start.x.min(self.end.x);
        let min_y = self.start.y.min(self.end.y);
        let max_x = self.start.x.max(self.end.x);
        let max_y = self.start.y.max(self.end.y);
        Aabb::with_bounds(Point2::new(min_x, min_y), Point2::new(max_x, max_y))
    }
}

impl BHShape<Real, 2> for LineShape {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

impl PointDistance<Real, 2> for LineShape {
    fn distance_squared(&self, point: Point2<Real>) -> Real {
        geometry::line_to_point_dist_sq(point, self.start, self.end)
    }
}

#[derive(Debug)]
pub struct LayerGeometry {
    pub id: usize,
    pub priority: usize,
    pub surface: Surface,

    poly: Polygon<Real>,
    spatial_tree: Bvh<Real, 2>,
    spatial_shapes: Vec<LineShape>,

    z_abs_top: Real,
    z_abs_bottom: Real,
    node_index: usize,
}

impl LayerGeometry {
    pub fn new_with_flat_surface(bounds: &Polygon<Real>, z_top: Real, z_bottom: Real) -> Self {
        // Due to extrapolation, we can treat the "interpolation" onto a flat
        // surface at the top and bottom using interpolation with just four points.
        let x = array!([0.0, 1.0], [0.0, 1.0]);
        let y = array!([0.0, 0.0], [1.0, 1.0]);
        let z_top_array = Array2::from_elem((2, 2), z_top);
        let z_bottom_array = Array2::from_elem((2, 2), z_bottom);

        LayerGeometry::build(
            bounds,
            x.view(),
            y.view(),
            z_top_array.view(),
            z_bottom_array.view(),
        )
    }

    pub fn build(
        bounds: &Polygon<Real>,
        surface_x: ArrayView2<Real>,
        surface_y: ArrayView2<Real>,
        surface_z_top: ArrayView2<Real>,
        surface_z_bottom: ArrayView2<Real>,
    ) -> Self {
        let (nx, ny) = surface_x.dim();

        let mut elevations = Vec::with_capacity(surface_x.len());
        for i in 0..nx {
            for j in 0..ny {
                elevations.push(SurfacePoint {
                    top: surface_z_top[[i, j]],
                    bottom: surface_z_bottom[[i, j]],
                });
            }
        }

        // Generate Triangles from the rectilinear grid
        let mut simplices = Vec::new();
        let mut vertex_map = Vec::new();

        let mut simplex_id = 0;
        let prepared_geom = PreparedGeometry::from(bounds);

        for i in 0..nx - 1 {
            for j in 0..ny - 1 {
                let coords = [
                    coord!(x: surface_x[[i, j]], y: surface_y[[i, j]]), // 0,0
                    coord!(x: surface_x[[i + 1, j]], y: surface_y[[i + 1, j]]), // 1,0
                    coord!(x: surface_x[[i + 1, j + 1]], y: surface_y[[i + 1, j + 1]]), // 1,1
                    coord!(x: surface_x[[i, j + 1]], y: surface_y[[i, j + 1]]), // 0,1
                ];

                // Split each grid cell into 2 triangles
                let tri1 = Triangle::new(coords[0], coords[1], coords[2]);
                let tri2 = Triangle::new(coords[0], coords[2], coords[3]);
                let triangles = [tri1, tri2];

                // Map 2D indices to flat array indices for the quad corners
                let idx_v0 = i * ny + j;
                let idx_v1 = (i + 1) * ny + j;
                let idx_v2 = (i + 1) * ny + j + 1;
                let idx_v3 = i * ny + j + 1;

                let indices = [[idx_v0, idx_v1, idx_v2], [idx_v0, idx_v2, idx_v3]];

                for (idx, tri) in indices.iter().zip(triangles.iter()) {
                    let mask = if prepared_geom.relate(tri).is_contains() {
                        Inclusion::Inside
                    } else {
                        Inclusion::Boundary
                    };
                    let coords = tri.to_array();
                    let points: [Point2<Real>; 3] = [
                        Point2::new(coords[0].x, coords[0].y),
                        Point2::new(coords[1].x, coords[1].y),
                        Point2::new(coords[2].x, coords[2].y),
                    ];

                    simplices.push(Simplex::new(
                        points[0], points[1], points[2], mask, simplex_id,
                    ));
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

        // Calculate absolute Z bounds
        let z_abs_top = *surface_z_top.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
        let z_abs_bottom = *surface_z_bottom
            .iter()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();

        // Build the BVH from the polygon's line segments for fast distance queries
        let mut edges = Vec::new();
        for line in bounds.exterior().lines() {
            edges.push(line.into());
        }
        for interior in bounds.interiors() {
            for line in interior.lines() {
                edges.push(line.into());
            }
        }
        let spatial_tree = Bvh::build(&mut edges);

        LayerGeometry {
            id: 0,
            priority: 0,
            surface,
            poly: bounds.clone(),
            spatial_tree,
            spatial_shapes: edges,
            z_abs_top,
            z_abs_bottom,
            node_index: 0,
        }
    }
}

impl PointDistance<Real, 3> for LayerGeometry {
    fn distance_squared(&self, query_point: Point3<Real>) -> Real {
        let projected_point = query_point.xy();
        let (z_top, z_bottom, inclusion) = match self.surface.query(projected_point) {
            Some(res) => res,
            None => return Real::INFINITY,
        };

        let (dxdy_sq, dz_sq) = match inclusion {
            Inclusion::Inside => {
                let z_clamped = query_point.z.clamp(z_top, z_bottom);
                (0.0, (query_point.z - z_clamped).powi(2))
            } // Guaranteed inside, skip full check
            Inclusion::Outside | Inclusion::Boundary => {
                if inclusion == Inclusion::Boundary
                    && self
                        .poly
                        .contains(&point!(x: query_point.x, y: query_point.y))
                {
                    let z_clamped = query_point.z.clamp(z_top, z_bottom);
                    (0.0, (query_point.z - z_clamped).powi(2))
                } else {
                    let (l, dxdy) = self
                        .spatial_tree
                        .nearest_to(query_point.xy(), &self.spatial_shapes)
                        .unwrap();
                    let closest_point = closest_point_to_line(projected_point, l.start, l.end);
                    let (z_top, z_bottom, _) = self.surface.query(closest_point).unwrap();
                    let z_clamped = query_point.z.clamp(z_top, z_bottom);
                    (dxdy.powi(2), (query_point.z - z_clamped).powi(2))
                }
            }
        };
        dz_sq + dxdy_sq
    }
}

impl Bounded<Real, 3> for LayerGeometry {
    fn aabb(&self) -> Aabb<Real, 3> {
        let coords = self.poly.bounding_rect().unwrap();
        let min = coords.min();
        let max = coords.max();
        let min_point = Point3::new(min.x, min.y, self.z_abs_top);
        let max_point = Point3::new(max.x, max.y, self.z_abs_bottom);

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
            x: c.x as Real,
            y: c.y as Real,
        }),
        _ => return Err("Bounds dataset is not a polygon".into()),
    };

    let x: Array2<Real> = group.dataset("surface_x")?.read_2d()?;
    let y: Array2<Real> = group.dataset("surface_y")?.read_2d()?;
    let z_top: Array2<Real> = group.dataset("surface_z_top")?.read_2d()?;
    let z_bottom: Array2<Real> = group.dataset("surface_z_bottom")?.read_2d()?;
    let mut geometry =
        LayerGeometry::build(&bounds, x.view(), y.view(), z_top.view(), z_bottom.view());
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
            let data: Array2<Real> = group.dataset("layers")?.read_2d()?;
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

impl BHShape<Real, 3> for LayerGeometry {
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
        layers: BTreeMap<OrderedFloat<Real>, Quality>,
    },
}

impl Model {
    pub fn query(&self, point: Point3<Real>) -> Option<Quality> {
        match self {
            Self::Uniform(quality) => Some(*quality),
            Self::Layered { layers } => layers
                .range(..=OrderedFloat(point.z))
                .next_back()
                .map(|(_, &q)| q)
                .or(layers.first_key_value().map(|(_, &q)| q)),
        }
    }
}

pub struct LayerTree {
    bvh_tree: Bvh<Real, 3>,
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
            shapes,
            bvh_tree,
            models,
        }
    }

    pub fn query_within(&self, point: Point3<Real>, epsilon: Real) -> Option<(Quality, Real)> {
        nearest_to_point_iterator(&self.bvh_tree, &self.shapes, &point, epsilon)
            .max_by_key(|(shape, _)| shape.priority)
            .and_then(|(best_shape, dist)| {
                self.model_query_for(best_shape, point).map(|q| (q, dist))
            })
    }

    pub fn query(&self, point: Point3<Real>) -> Option<(Quality, Real)> {
        self.query_within(point, Real::EPSILON)
    }

    pub fn priorities(&self) -> Vec<usize> {
        self.shapes.iter().map(|shape| shape.priority).collect()
    }

    pub fn bounds(&self) -> Vec<Aabb<Real, 3>> {
        self.shapes.iter().map(|shape| shape.aabb()).collect()
    }

    fn model_query_for(&self, shape: &LayerGeometry, point: Point3<Real>) -> Option<Quality> {
        let z_top = shape
            .surface
            .query(point.xy())
            .map(|(z_top, _, _)| z_top)
            .unwrap_or(0.0);
        let mut projected_point = point;
        projected_point.z -= z_top;
        self.models[shape.id].query(projected_point)
    }

    pub fn pretty_print(&self) {
        println!(
            "Disjoint layer models, having {} layers with structure:",
            self.models.len()
        );
        self.bvh_tree.pretty_print();
    }
}

// TODO: More unit-tests here including:
// 1. Check that the correct distance in boundary type cells (0 on inside, positive on outside).
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use geo::polygon;
    use nalgebra::Point3;
    use ordered_float::OrderedFloat;

    fn create_unit_prism(z_top: Real, z_bottom: Real) -> LayerGeometry {
        let poly = polygon![
            (x: 0.0, y: 0.0),
            (x: 1.0, y: 0.0),
            (x: 1.0, y: 1.0),
            (x: 0.0, y: 1.0),
        ];
        LayerGeometry::new_with_flat_surface(&poly, z_top, z_bottom)
    }

    fn mock_quality(val: Real) -> Quality {
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
            epsilon = Real::EPSILON
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

    #[test]
    fn test_layered_model_stepping() {
        let mut layers = BTreeMap::new();
        layers.insert(OrderedFloat(0.0), mock_quality(10.0));
        layers.insert(OrderedFloat(100.0), mock_quality(20.0));

        let model = Model::Layered { layers };

        assert_eq!(model.query(Point3::new(0.0, 0.0, 100.0)).unwrap().rho, 20.0);
        assert_eq!(model.query(Point3::new(0.0, 0.0, 50.0)).unwrap().rho, 10.0);
        assert_eq!(model.query(Point3::new(0.0, 0.0, -10.0)).unwrap().rho, 10.0);
    }

    #[test]
    fn test_model_tree_nearest_neighbor_behavior() {
        let prisms = vec![create_unit_prism(0.0, 10.0)];
        let models = vec![Model::Uniform(mock_quality(1.0))];

        // API Change: LayerTree::new now consumes the vectors
        let tree = LayerTree::new(prisms, models);
        // Point is too far away from the X=1.0 face.
        let far_point = Point3::new(10.0, 0.5, 5.0);
        assert!(tree.query(far_point).is_none());
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

        // API Change: query returns (Quality, Real)
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
        let x = array!([0.0, 2.0], [2.0, 0.0]);
        let y = array!([0.0, 0.0], [2.0, 2.0]);

        // Top surface slopes from z=0 at x=0 to z=10 at x=2
        let z_top = array![[0.0, 0.0], [10.0, 10.0]];
        let z_bottom = Array2::from_elem((2, 2), 20.0);

        let prism = LayerGeometry::build(&poly, x.view(), y.view(), z_top.view(), z_bottom.view());

        // Test vertical distance above the slope (at x=0, z_top=0)
        let p_start = Point3::new(0.0, 0.0, -5.0);
        assert_abs_diff_eq!(prism.distance_squared(p_start), 25.0, epsilon = 1e-5);

        // Test point inside the sloped volume (at x=1.0, interpolated z_top is 5.0)
        let p_mid = Point3::new(1.0, 1.0, 7.0);
        assert_eq!(prism.distance_squared(p_mid), 0.0);
    }

    #[test]
    fn test_mesh_grid_to_simplex_indexing() {
        // Creates a 2x2 grid (nx=2, ny=2) consisting of 1 quad, divided into 2 triangles.
        let poly = polygon![(x: 0.0, y: 0.0), (x: 1.0, y: 0.0), (x: 1.0, y: 1.0), (x: 0.0, y: 1.0)];
        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let y = array![[0.0, 1.0], [0.0, 1.0]];
        let z_top = array![[0.0, 0.0], [0.0, 0.0]];
        let z_bottom = array![[1.0, 1.0], [1.0, 1.0]];

        let layer = LayerGeometry::build(&poly, x.view(), y.view(), z_top.view(), z_bottom.view());

        assert_eq!(layer.surface.simplices.len(), 2);

        // Assert the first triangle correctly targets the flat 1D coordinates of [0,0], [1,0], and [1,1]
        let t1_verts = layer.surface.vertex_map[0];
        assert_eq!(t1_verts, Point3::new(0, 2, 3));

        // Assert the second triangle correctly targets [0,0], [1,1], and [0,1]
        let t2_verts = layer.surface.vertex_map[1];
        assert_eq!(t2_verts, Point3::new(0, 3, 1));
    }
}
