use crate::quality::Quality;
use crate::real::Real;
use nalgebra::Point3;

/// Performance counters collected during a single model query.
#[derive(Debug)]
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
    pub elapsed: u128,
}

/// One model's contribution to a blended query result.
pub struct ModelContribution {
    /// Priority of the contributing model (lower = higher priority).
    pub priority: u8,
    /// Quality returned by this model at the query point.
    pub quality: Quality,
}

/// Diagnostic breakdown of a query, listing every model that contributed.
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
}
