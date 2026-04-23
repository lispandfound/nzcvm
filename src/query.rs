use crate::quality::Quality;
use crate::real::Real;
use nalgebra::Point3;

#[derive(Debug)]
pub struct QueryStats {
    pub aabb_tests: usize,
    pub simplex_tests: usize,
    pub hit_count: usize,
    pub output: Option<Quality>,
    pub elapsed: u128,
}

pub struct ModelContribution {
    pub priority: u8,
    pub quality: Quality,
}

pub struct Explanation {
    pub contributions: Vec<ModelContribution>,
    pub output: Option<Quality>,
    pub termination: Option<usize>,
}

pub trait Query {
    type Explanation;

    fn query(&self, point: Point3<Real>) -> Option<Quality>;
    fn query_stats(&self, point: Point3<Real>) -> QueryStats;
    fn query_explain(&self, point: Point3<Real>) -> Self::Explanation;
}
