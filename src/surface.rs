// TODO: Scream about invalid simplices instead of creating degenerate matices
use crate::real::Real;
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::Bvh;
use bvh::point_query::PointDistance;
use nalgebra::{Point2, Point3};

use crate::tree_query::nearest_to_point_within;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Inclusion {
    Inside,
    Boundary,
    Outside,
}

#[derive(Copy, Clone, Debug)]
pub struct SurfacePoint {
    pub top: Real,
    pub bottom: Real,
}

#[derive(Debug)]
pub struct Simplex {
    c0: Point2<Real>,
    c1: Point2<Real>,
    c2: Point2<Real>,
    // TODO: make this an actual matrix
    // Components of the inverse matrix
    // [ m00 m01 ]
    // [ m10 m11 ]
    inv_m: [Real; 4],
    pub mask: Inclusion,
    pub id: usize,
    node_index: usize,
}

impl Simplex {
    pub fn new(
        c0: Point2<Real>,
        c1: Point2<Real>,
        c2: Point2<Real>,
        mask: Inclusion,
        id: usize,
    ) -> Self {
        let v0 = c0 - c2;
        let v1 = c1 - c2;

        let det = v0.x * v1.y - v1.x * v0.y;

        let inv_det = if det.abs() > Real::EPSILON {
            1.0 / det
        } else {
            0.0
        };

        let inv_m = [
            v1.y * inv_det,
            -v1.x * inv_det,
            -v0.y * inv_det,
            v0.x * inv_det,
        ];

        Self {
            c0,
            c1,
            c2,
            inv_m,
            mask,
            id,
            node_index: 0,
        }
    }

    #[inline(always)]
    pub fn barycentric_coordinates(&self, p: Point2<Real>) -> Point3<Real> {
        let dx = p.x - self.c2.x;
        let dy = p.y - self.c2.y;

        // Multiply by the stored inverse matrix
        let w0 = self.inv_m[0] * dx + self.inv_m[1] * dy;
        let w1 = self.inv_m[2] * dx + self.inv_m[3] * dy;
        let w2 = 1.0 - w0 - w1;

        Point3::new(w0, w1, w2)
    }
}

impl Bounded<Real, 2> for Simplex {
    fn aabb(&self) -> Aabb<Real, 2> {
        let min_x = self.c0.x.min(self.c1.x).min(self.c2.x);
        let min_y = self.c0.y.min(self.c1.y).min(self.c2.y);
        let max_x = self.c0.x.max(self.c1.x).max(self.c2.x);
        let max_y = self.c0.y.max(self.c1.y).max(self.c2.y);
        Aabb::with_bounds(Point2::new(min_x, min_y), Point2::new(max_x, max_y))
    }
}

impl BHShape<Real, 2> for Simplex {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }
    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

impl PointDistance<Real, 2> for Simplex {
    fn distance_squared(&self, query_point: Point2<Real>) -> Real {
        let bary = self.barycentric_coordinates(query_point);
        if bary.x >= 0.0 && bary.y >= 0.0 && bary.z >= 0.0 {
            0.0
        } else {
            bary.x.min(0.0).powi(2) + bary.y.min(0.0).powi(2) + bary.z.min(0.0).powi(2)
        }
    }
}

#[derive(Debug)]
pub struct Surface {
    pub bvh_tree: Bvh<Real, 2>,
    pub simplices: Vec<Simplex>,
    /// Maps simplex to vertex indices (x, y, z) in the elevation buffers
    pub vertex_map: Vec<Point3<usize>>,
    /// Elevation data stored at vertices
    pub elevations: Vec<SurfacePoint>,
}

impl Surface {
    /// Queries the surface at a specific (x, y) coordinate.
    /// Returns (Top Elevation, Bottom Elevation, Inclusion Status)
    pub fn query(&self, point: Point2<Real>) -> Option<(Real, Real, Inclusion)> {
        nearest_to_point_within(&self.bvh_tree, &self.simplices, point, Real::EPSILON).map(
            |(simplex, _dist)| {
                let bary = simplex.barycentric_coordinates(point);

                let indices = self.vertex_map[simplex.id];
                let p0 = self.elevations[indices.x];
                let p1 = self.elevations[indices.y];
                let p2 = self.elevations[indices.z];

                let top = p0.top * bary.x + p1.top * bary.y + p2.top * bary.z;
                let bottom =
                    (p0.bottom * bary.x + p1.bottom * bary.y + p2.bottom * bary.z).max(top);
                // If barycentric coordinates indicate that we are
                // inside the simplex, we can use the simplex mask.
                // Otherwise assuming that the surface covers the
                // entire polygon we can simply say it is outside.
                // TODO: Handle this edge case more gracefully.
                let mask = if bary.x >= 0.0 && bary.y >= 0.0 && bary.z >= 0.0 {
                    simplex.mask
                } else {
                    Inclusion::Outside
                };
                (top, bottom, mask)
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn mock_unit_simplex(id: usize, mask: Inclusion) -> Simplex {
        // A right triangle: (0,0), (1,0), (0,1)
        Simplex::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
            mask,
            id,
        )
    }

    #[test]
    fn test_simplex_barycentric_identity() {
        let simplex = mock_unit_simplex(0, Inclusion::Inside);

        let b0 = simplex.barycentric_coordinates(Point2::new(0.0, 0.0));
        let b1 = simplex.barycentric_coordinates(Point2::new(1.0, 0.0));
        let b2 = simplex.barycentric_coordinates(Point2::new(0.0, 1.0));

        assert_relative_eq!(b0, Point3::new(1.0, 0.0, 0.0));
        assert_relative_eq!(b1, Point3::new(0.0, 1.0, 0.0));
        assert_relative_eq!(b2, Point3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_simplex_centroid() {
        let simplex = mock_unit_simplex(0, Inclusion::Inside);

        let p = Point2::new(1.0 / 3.0, 1.0 / 3.0);
        let bary = simplex.barycentric_coordinates(p);

        assert_relative_eq!(bary.x, 1.0 / 3.0);
        assert_relative_eq!(bary.y, 1.0 / 3.0);
        assert_relative_eq!(bary.z, 1.0 / 3.0);
        assert_relative_eq!(bary.x + bary.y + bary.z, 1.0);
    }

    #[test]
    fn test_simplex_distance_squared() {
        let simplex = mock_unit_simplex(0, Inclusion::Inside);

        assert_eq!(simplex.distance_squared(Point2::new(0.2, 0.2)), 0.0);

        let dist = simplex.distance_squared(Point2::new(-1.0, 0.2));
        assert!(dist > 0.0);
    }

    #[test]
    fn test_simplex_aabb() {
        let simplex = Simplex::new(
            Point2::new(-5.0, 10.0),
            Point2::new(2.0, -3.0),
            Point2::new(4.0, 4.0),
            Inclusion::Outside,
            0,
        );
        let aabb = simplex.aabb();

        assert_eq!(aabb.min, Point2::new(-5.0, -3.0));
        assert_eq!(aabb.max, Point2::new(4.0, 10.0));
    }

    #[test]
    fn test_surface_query_interpolation() {
        // Create two triangles forming a square from (0,0) to (1,1)
        // T0: (0,0), (1,0), (1,1)
        // T1: (0,0), (1,1), (0,1)
        let s0 = Simplex::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Inclusion::Inside,
            0,
        );
        let s1 = Simplex::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.1, 1.0),
            Inclusion::Boundary,
            1,
        );

        let mut simplices = vec![s0, s1];
        let bvh_tree = Bvh::build(&mut simplices);

        // Define elevations at vertices
        // Let's say top is always 0.0 and bottom is 100.0, except at (1,1) where bottom is 200.0 and top is 50.0
        let elevations = vec![
            SurfacePoint {
                top: 0.0,
                bottom: 100.0,
            }, // (0,0) - Index 0
            SurfacePoint {
                top: 0.0,
                bottom: 100.0,
            }, // (1,0) - Index 1
            SurfacePoint {
                top: 50.0,
                bottom: 200.0,
            }, // (1,1) - Index 2
            SurfacePoint {
                top: 0.0,
                bottom: 100.0,
            }, // (0,1) - Index 3
        ];

        let vertex_map = vec![
            Point3::new(0, 1, 2), // T0
            Point3::new(0, 2, 3), // T1
        ];

        let surface = Surface {
            bvh_tree,
            simplices,
            vertex_map,
            elevations,
        };

        // Query the middle of T0 (0.5, 0.25)
        // This point is 25% of the way toward the (1,1) vertex from the (0,0)-(1,0) base
        if let Some((top, bottom, mask)) = surface.query(Point2::new(0.75, 0.5)) {
            println!("top = {}, bottom = {}", top, bottom);
            assert!(bottom > 100.0 && bottom < 200.0);
            assert!(top > 0.0 && top < 50.0);
            match mask {
                Inclusion::Inside => {}
                _ => panic!("Expected Inside mask for T0"),
            }
        } else {
            panic!("Query failed to find simplex");
        }

        // Query T1 boundary
        if let Some((_, _, mask)) = surface.query(Point2::new(0.2, 0.8)) {
            match mask {
                Inclusion::Boundary => {}
                _ => panic!("Expected Boundary mask for T1"),
            }
        }
    }

    #[test]
    fn test_degenerate_simplex_handling() {
        // Triangle with zero area
        let s = Simplex::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(2.0, 0.0),
            Inclusion::Outside,
            0,
        );

        let bary = s.barycentric_coordinates(Point2::new(0.5, 0.0));
        assert_relative_eq!(bary.x + bary.y + bary.z, 1.0);
    }
}
