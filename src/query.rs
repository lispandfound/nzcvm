use crate::quality::Quality;
use crate::real::Real;
use nalgebra::Point3;
use serde::Serialize;

/// Performance counters collected during a single model query.
#[derive(Debug, Serialize)]
pub struct QueryStats {
    /// Number of axis-aligned bounding-box intersection tests performed.
    pub aabb_tests: usize,
    /// Number of simplex containment tests performed.
    pub simplex_tests: usize,
    /// Number of simplices that contained the query point.
    pub hit_count: usize,
    /// Final blended quality, or `None` if the point is outside all models.
    pub output: Option<Quality>,
    /// Wall-clock time for the full query in nanoseconds.
    pub elapsed: u64,
}

/// One model's contribution to a blended query result.
#[derive(Serialize)]
pub struct ModelContribution {
    /// Priority of the contributing model (lower = higher priority).
    pub priority: u8,
    /// Quality returned by this model at the query point.
    pub quality: Quality,
}

/// Diagnostic breakdown of a query, listing every model that contributed.
#[derive(Serialize)]
pub struct Explanation {
    /// Ordered list of contributions, from highest priority to lowest.
    pub contributions: Vec<ModelContribution>,
    /// Final blended output quality, or `None` if the point is outside all models.
    pub output: Option<Quality>,
    /// Index into `contributions` at which the blend became fully opaque.
    /// Contributions at or after this index do not affect the final result.
    pub termination: Option<usize>,
}

/// Spatial query interface for velocity models.
pub trait Query {
    type Explanation;

    /// Return the blended quality at `point`, or `None` if outside all models.
    fn query(&self, point: Point3<Real>) -> Option<Quality>;
    /// Return the quality at `point` together with BVH traversal statistics.
    fn query_stats(&self, point: Point3<Real>) -> QueryStats;
    /// Return a full diagnostic breakdown of the query at `point`.
    fn query_explain(&self, point: Point3<Real>) -> Self::Explanation;

    /// Return the blended quality at `point` considering only models whose
    /// priority falls in `[priority_lo, priority_hi]` (both inclusive).
    ///
    /// Priority convention: 0 is the highest priority and 255 is the lowest.
    fn query_bounded(
        &self,
        point: Point3<Real>,
        priority_lo: u8,
        priority_hi: u8,
    ) -> Option<Quality>;

    /// Alpha-blend the query result at `point` into `existing`.
    ///
    /// Equivalent to computing the query result and compositing it over
    /// `existing` using the Porter-Duff "over" operator, except that the
    /// traversal starts from `existing` rather than `None`.  If the existing
    /// quality is already fully opaque the call returns early without
    /// traversing the BVH.
    fn query_into(
        &self,
        point: Point3<Real>,
        existing: Option<Quality>,
    ) -> Option<Quality>;

    /// Combination of [`query_bounded`](Self::query_bounded) and
    /// [`query_into`](Self::query_into): blends only models in
    /// `[priority_lo, priority_hi]` into `existing`.
    fn query_bounded_into(
        &self,
        point: Point3<Real>,
        existing: Option<Quality>,
        priority_lo: u8,
        priority_hi: u8,
    ) -> Option<Quality>;
}
