use crate::real::Real;
use crate::tree_query::Contains;
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use deepsize::{Context, DeepSizeOf};

use nalgebra::{Matrix2, Point2, Point3, Vector2};

const CONTAINMENT_EPS: Real = 1e-4;

/// A 2D triangle with pre-computed 2D inverse matrix for fast $x,y$ containment queries.
///
/// Vertex `c2` is the "anchor"; $c_0$ and $c_1$ are stored implicitly through the
/// inverse of the $2 \times 2$ matrix $[(c_0-c_2)_{xy}, (c_1-c_2)_{xy}]$.
#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub c2: Point2<Real>,
    /// Precomputed inverse of the 2D basis mapping $l_0, l_1$ to $x, y$.
    inv_matrix: Matrix2<Real>,

    /// 2D AABB for internal surface BVH queries.
    aabb: Aabb<Real, 2>,

    pub id: usize,
    node_index: usize,
}

impl DeepSizeOf for Triangle {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

impl Triangle {
    /// Construct a new triangle from three vertices.
    ///
    /// Note: The BVH and containment logic use the XY projection.
    /// The Z-coordinates are preserved for interpolation.
    pub fn new(c0: Point2<Real>, c1: Point2<Real>, c2: Point2<Real>, id: usize) -> Self {
        let pts = [c0, c1, c2];

        // 2D AABB construction (XY plane)
        let min_p = Point2::new(
            pts.iter().map(|p| p.x).fold(Real::MAX, Real::min),
            pts.iter().map(|p| p.y).fold(Real::MAX, Real::min),
        );
        let max_p = Point2::new(
            pts.iter().map(|p| p.x).fold(Real::MIN, Real::max),
            pts.iter().map(|p| p.y).fold(Real::MIN, Real::max),
        );
        let aabb = Aabb::with_bounds(min_p, max_p);

        // Basis vectors in 2D (XY plane)
        let v0 = Vector2::new(c0.x - c2.x, c0.y - c2.y);
        let v1 = Vector2::new(c1.x - c2.x, c1.y - c2.y);

        let m = Matrix2::from_columns(&[v0, v1]);
        let inv_matrix = m
            .try_inverse()
            .expect("Degenerate triangle (zero area in XY plane) encountered");

        Self {
            c2,
            inv_matrix,
            aabb,
            id,
            node_index: 0,
        }
    }

    /// Return the 3-element barycentric coordinates $(l_0, l_1, l_2)$ for point `p`.
    ///
    /// Used for interpolating elevation ($z$) or other qualities.
    #[inline(always)]
    pub fn barycentric_coordinates(&self, p: &Point2<Real>) -> Point3<Real> {
        let diff = Vector2::new(p.x - self.c2.x, p.y - self.c2.y);
        let l = self.inv_matrix * diff;

        let l0 = l.x;
        let l1 = l.y;
        let l2 = 1.0 - l0 - l1;

        Point3::new(l0, l1, l2)
    }
}

/// Contains implementation for 2D space.
impl Contains<Real, 2, Triangle> for Triangle {
    #[inline(always)]
    fn contains(&self, query_point: &Point2<Real>) -> Option<Triangle> {
        let diff = Vector2::new(query_point.x - self.c2.x, query_point.y - self.c2.y);

        let l0 = self.inv_matrix[(0, 0)] * diff.x + self.inv_matrix[(0, 1)] * diff.y;
        let l1 = self.inv_matrix[(1, 0)] * diff.x + self.inv_matrix[(1, 1)] * diff.y;

        if (l0 >= -CONTAINMENT_EPS)
            && (l1 >= -CONTAINMENT_EPS)
            && (l0 + l1 <= 1.0 + CONTAINMENT_EPS)
        {
            Some(*self)
        } else {
            None
        }
    }
}

impl Bounded<Real, 2> for Triangle {
    fn aabb(&self) -> Aabb<Real, 2> {
        self.aabb
    }
}

impl BHShape<Real, 2> for Triangle {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }
    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}
