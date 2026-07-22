use crate::real::Real;
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
///
/// This struct is deliberately minimal (48 bytes with `Real = f32` — one
/// cache line) because meshes hold tens of millions of them.  Everything
/// needed only while *building* the BVH (AABB, node bookkeeping) lives in
/// [`BuildSimplex`] and is discarded after construction.  A simplex is
/// identified by its position in the mesh's simplex array.
#[derive(Clone, Copy, Debug)]
pub struct Simplex {
    pub c3: Point3<Real>,
    inv_matrix: Matrix3<Real>,
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
    /// and [`contains`](Self::contains) call, which is the hot path for BVH
    /// queries.
    ///
    /// Returns `None` if the four vertices are coplanar (degenerate simplex).
    pub fn new(c0: Point3<Real>, c1: Point3<Real>, c2: Point3<Real>, c3: Point3<Real>) -> Option<Self> {
        let m = Matrix3::from_columns(&[c0 - c3, c1 - c3, c2 - c3]);
        let inv_matrix = m.try_inverse()?;

        Some(Self { c3, inv_matrix })
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
    /// ).unwrap();
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

    /// Test whether `query_point` lies inside this simplex (within
    /// [`CONTAINMENT_EPS`] of each face).
    // This one inline statement speeds up calculations by 6%!
    #[inline(always)]
    pub fn contains(&self, query_point: &Point3<Real>) -> bool {
        let diff = query_point - self.c3;
        // This duplication of the matrix multiply from
        // `barycentric_coordinates` is deliberate. `l3` is not needed here, so
        // the subtraction is skipped.

        let l = self.inv_matrix * diff;

        let sum = l.x + l.y + l.z;

        (l.x >= -CONTAINMENT_EPS)
            && (l.y >= -CONTAINMENT_EPS)
            && (l.z >= -CONTAINMENT_EPS)
            && (sum <= 1.0 + CONTAINMENT_EPS)
    }
}

/// Build-time wrapper around [`Simplex`] carrying the fields the BVH build
/// needs (AABB and node bookkeeping).  Discarded once the mesh's
/// [`CompactBvh`](crate::compact_bvh::CompactBvh) has been constructed.
pub struct BuildSimplex {
    pub simplex: Simplex,
    aabb: Aabb<Real, 3>,
    node_index: usize,
}

impl BuildSimplex {
    /// Construct a build-time simplex from four vertices.
    ///
    /// Returns `None` if the four vertices are coplanar (degenerate simplex).
    pub fn new(
        c0: Point3<Real>,
        c1: Point3<Real>,
        c2: Point3<Real>,
        c3: Point3<Real>,
    ) -> Option<Self> {
        let simplex = Simplex::new(c0, c1, c2, c3)?;

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

        Some(Self {
            simplex,
            aabb: Aabb::with_bounds(min_p, max_p),
            node_index: 0,
        })
    }
}

impl Bounded<Real, 3> for BuildSimplex {
    fn aabb(&self) -> Aabb<Real, 3> {
        self.aabb
    }
}

impl BHShape<Real, 3> for BuildSimplex {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }
    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}
