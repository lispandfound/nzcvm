use crate::geometry::point_triangle_distance_sq;
use crate::real::Real;
use crate::tree_query::Contains;
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use bvh::point_query::PointDistance;
use nalgebra::{Matrix3, Point3, Point4};

#[derive(Debug, Clone, Copy)]
pub struct Simplex {
    pub c0: Point3<Real>,
    pub c1: Point3<Real>,
    pub c2: Point3<Real>,
    pub c3: Point3<Real>,

    inv_matrix: Matrix3<Real>,

    pub id: usize,
    node_index: usize,
    pub priority: u8,
}

impl Simplex {
    pub fn new(
        c0: Point3<Real>,
        c1: Point3<Real>,
        c2: Point3<Real>,
        c3: Point3<Real>,
        id: usize,
    ) -> Self {
        let m = Matrix3::from_columns(&[c0 - c3, c1 - c3, c2 - c3]);
        let inv_matrix = m.try_inverse().unwrap();

        Self {
            c0,
            c1,
            c2,
            c3,
            id,
            inv_matrix,
            node_index: 0,
            priority: 0,
        }
    }

    pub fn barycentric_coordinates(&self, p: Point3<Real>) -> Point4<Real> {
        let diff = p - self.c3;
        let l = self.inv_matrix * diff;

        let l0 = l[0];
        let l1 = l[1];
        let l2 = l[2];
        let l3 = 1.0 - l0 - l1 - l2;

        Point4::new(l0, l1, l2, l3)
    }
}

impl Contains<Real, 3> for Simplex {
    fn contains(&self, query_point: &Point3<Real>) -> bool {
        let bary = self.barycentric_coordinates(*query_point);
        bary.x >= 0.0 && bary.y >= 0.0 && bary.z >= 0.0 && bary.w >= 0.0
    }
}

impl PointDistance<Real, 3> for Simplex {
    fn distance_squared(&self, query_point: Point3<Real>) -> Real {
        let bary = self.barycentric_coordinates(query_point);

        if bary.x >= 0.0 && bary.y >= 0.0 && bary.z >= 0.0 && bary.w >= 0.0 {
            return 0.0;
        }

        let mut min_dist_sq = Real::INFINITY;

        if bary.x < 0.0 {
            min_dist_sq = min_dist_sq.min(point_triangle_distance_sq(
                query_point,
                self.c1,
                self.c2,
                self.c3,
            ));
        }
        if bary.y < 0.0 {
            min_dist_sq = min_dist_sq.min(point_triangle_distance_sq(
                query_point,
                self.c0,
                self.c2,
                self.c3,
            ));
        }
        if bary.z < 0.0 {
            min_dist_sq = min_dist_sq.min(point_triangle_distance_sq(
                query_point,
                self.c0,
                self.c1,
                self.c3,
            ));
        }
        if bary.w < 0.0 {
            min_dist_sq = min_dist_sq.min(point_triangle_distance_sq(
                query_point,
                self.c0,
                self.c1,
                self.c2,
            ));
        }

        min_dist_sq
    }
}

impl Bounded<Real, 3> for Simplex {
    fn aabb(&self) -> Aabb<Real, 3> {
        let pts = [self.c0, self.c1, self.c2, self.c3];
        let min_point = pts
            .iter()
            .copied()
            .reduce(|acc, p| Point3::new(acc.x.min(p.x), acc.y.min(p.y), acc.z.min(p.z)))
            .unwrap();
        let max_point = pts
            .iter()
            .copied()
            .reduce(|acc, p| Point3::new(acc.x.max(p.x), acc.y.max(p.y), acc.z.max(p.z)))
            .unwrap();
        Aabb::with_bounds(min_point, max_point)
    }
}

impl BHShape<Real, 3> for Simplex {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}
