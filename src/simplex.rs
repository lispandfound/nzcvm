use crate::real::Real;
use crate::tree_query::Contains;
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use deepsize::{Context, DeepSizeOf};

use nalgebra::{Matrix3, Point3, Point4};

/// Tolerance applied to each barycentric coordinate in the point-in-simplex
/// test (`contains`).
///
/// Barycentric coordinates are dimensionless and lie in `[0, 1]` for interior
/// points; `CONTAINMENT_EPS` admits points that lie just outside a simplex
/// face by up to this fraction of the simplex extent, preventing cracks along
/// shared faces caused by floating-point rounding.
///
/// In world-space terms a point at barycentric distance `CONTAINMENT_EPS` from
/// a face is roughly `CONTAINMENT_EPS × L` metres away from the face, where
/// `L` is the characteristic edge length.  For meshes with edge lengths ≥ 10 m
/// (typical for NZCVM) this corresponds to ≤ 1 mm of penetration — well within
/// measurement uncertainty.
///
/// # TODO (Scientific Review)
///
/// Verify that this tolerance does not cause adjacent simplices to be double-
/// counted (overlap) at their shared faces in the coarsest meshes used by the
/// project.  A value of `1e-4` was benchmarked to speed up calculations by
/// approximately 6 % compared to the strict `0.0` threshold.
const CONTAINMENT_EPS: Real = 1e-4;

/// A tetrahedron (3-simplex) with pre-computed inverse matrix for fast
/// barycentric coordinate queries.
///
/// Vertex `c3` is the "anchor" vertex; the other three vertices are stored
/// implicitly through the inverse of the matrix `[c0-c3, c1-c3, c2-c3]`.
#[derive(Clone, Copy, Debug)]
pub struct Simplex {
    pub c3: Point3<Real>,
    inv_matrix: Matrix3<Real>,

    aabb: Aabb<Real, 3>,

    pub id: usize,
    node_index: usize,
}

impl DeepSizeOf for Simplex {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

impl Simplex {
    /// Construct a new simplex from four vertices.
    ///
    /// The inverse of `[c0-c3, c1-c3, c2-c3]` is precomputed here and reused
    /// in every subsequent [`barycentric_coordinates`](Self::barycentric_coordinates)
    /// and [`Contains::contains`] call, which is the hot path for BVH queries.
    ///
    /// # Panics
    ///
    /// Panics if the four vertices are coplanar (degenerate simplex).
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

    /// Return the barycentric coordinates of `p` with respect to this simplex.
    ///
    /// The four coordinates `(l0, l1, l2, l3)` always sum to `1.0`.  A point
    /// is inside the simplex when all four coordinates are non-negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use nalgebra::Point3;
    /// use nzcvm::simplex::Simplex;
    /// let s = Simplex::new(
    ///     Point3::new(0.0, 0.0, 0.0),
    ///     Point3::new(1.0, 0.0, 0.0),
    ///     Point3::new(0.0, 1.0, 0.0),
    ///     Point3::new(0.0, 0.0, 1.0),
    ///     0,
    /// );
    /// let bary = s.barycentric_coordinates(Point3::new(0.25, 0.25, 0.25));
    /// let sum = bary.x + bary.y + bary.z + bary.w;
    /// assert!((sum - 1.0).abs() < 1e-5);
    /// ```
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

impl Contains<Real, 3, Simplex> for Simplex {
    // This one inline statement speeds up calculations by 6%!
    #[inline(always)]
    fn contains(&self, query_point: &Point3<Real>) -> Option<Simplex> {
        let diff = query_point - self.c3;
        // This duplication of the matrix multiply from
        // `barycentric_coordinates` is deliberate. `l3` is not needed here, so
        // the subtraction is skipped.

        let l = self.inv_matrix * diff;

        let sum = l.x + l.y + l.z;

        if (l.x >= -CONTAINMENT_EPS)
            && (l.y >= -CONTAINMENT_EPS)
            && (l.z >= -CONTAINMENT_EPS)
            && (sum <= 1.0 + CONTAINMENT_EPS)
        {
            Some(*self)
        } else {
            None
        }
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
