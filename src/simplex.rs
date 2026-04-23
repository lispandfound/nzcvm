use crate::real::Real;
use crate::tree_query::Contains;
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use nalgebra::{Matrix3, Point3, Point4};

#[derive(Debug, Copy, Clone)]
pub struct Simplex {
    pub c3: Point3<Real>,
    inv_matrix: Matrix3<Real>,

    aabb: Aabb<Real, 3>,

    pub id: usize,
    node_index: usize,
}

impl Simplex {
    pub fn new(
        c0: Point3<Real>,
        c1: Point3<Real>,
        c2: Point3<Real>,
        c3: Point3<Real>,
        id: usize,
    ) -> Self {
        let pts = [c0, c1, c2, c3];
        let min_p = pts
            .iter()
            .copied()
            .reduce(|acc, p| Point3::new(acc.x.min(p.x), acc.y.min(p.y), acc.z.min(p.z)))
            .unwrap();
        let max_p = pts
            .iter()
            .copied()
            .reduce(|acc, p| Point3::new(acc.x.max(p.x), acc.y.max(p.y), acc.z.max(p.z)))
            .unwrap();

        let aabb = Aabb::with_bounds(min_p, max_p);

        let m = Matrix3::from_columns(&[c0 - c3, c1 - c3, c2 - c3]);
        let inv_matrix = m
            .try_inverse()
            .expect("Degenerate simplex encountered during build");

        Self {
            c3,
            inv_matrix,
            aabb,
            id,
            node_index: 0,
        }
    }

    pub fn barycentric_coordinates(&self, p: Point3<Real>) -> Point4<Real> {
        let diff = p - self.c3;
        let l = self.inv_matrix * diff;

        let l0 = l.x;
        let l1 = l.y;
        let l2 = l.z;
        let l3 = 1.0 - l0 - l1 - l2;

        Point4::new(l0, l1, l2, l3)
    }
}

impl Contains<Real, 3> for Simplex {
    // This one inline statement speeds up calculations by 6%!
    #[inline(always)]
    fn contains(&self, query_point: &Point3<Real>) -> bool {
        let diff = query_point - self.c3;
        let l = self.inv_matrix * diff;

        let eps = 1e-4;

        let sum = l.x + l.y + l.z;

        (l.x >= -eps) & (l.y >= -eps) & (l.z >= -eps) & (sum <= 1.0 + eps)
    }
}

impl Bounded<Real, 3> for Simplex {
    fn aabb(&self) -> Aabb<Real, 3> {
        self.aabb
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
