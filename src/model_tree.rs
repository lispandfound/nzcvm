use std::time::Instant;

use approx::abs_diff_eq;
use bvh::bvh::{Bvh, BvhNode};
use deepsize::{Context, DeepSizeOf};
use serde::Serialize;

use crate::mesh::{MeshModel, MeshModelView};
use crate::quality::Quality;
use crate::query::{Explanation, ModelContribution, Query, QueryStats};
use crate::real::Real;
use crate::tree_query::{priority_ray_iterator, priority_ray_stats_iterator};
use nalgebra::Point3;

/// Alpha value at which a blended quality is considered fully opaque. When
/// the running blend reaches this threshold no further models need to
/// contribute to the result.
const ALPHA_SATURATED: Real = 1.0;
const ALPHA_EPS: Real = 1e-4;

#[derive(Serialize)]
pub struct ModelTreeView {
    pub models: Vec<MeshModelView>,
    pub size: usize,
}

pub struct ModelTree {
    bvh_tree: Bvh<Real, 4>,
    models: Vec<MeshModel>,
}

impl DeepSizeOf for ModelTree {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.bvh_tree.nodes.capacity() * size_of::<BvhNode<Real, 3>>()
            + self.models.deep_size_of_children(context)
    }
}

impl ModelTree {
    pub fn new(mut models: Vec<MeshModel>) -> Self {
        for (i, model) in models.iter_mut().enumerate() {
            model.id = i;
        }
        let bvh_tree = Bvh::build(&mut models);
        Self { bvh_tree, models }
    }

    pub fn aabb(&self) -> bvh::aabb::Aabb<Real, 3> {
        self.models
            .iter()
            .map(|m| m.aabb3())
            .reduce(|a, b| a.join(&b))
            .expect("ModelTree must contain at least one model")
    }

    pub fn pretty_print(&self) {
        println!("ModelTree with {} mesh models:", self.models.len());
        for model in &self.models {
            model.pretty_print();
        }
    }

    pub fn view(&self) -> ModelTreeView {
        ModelTreeView {
            models: self.models.iter().map(|model| model.view()).collect(),
            size: self.deep_size_of(),
        }
    }
}

impl Query for ModelTree {
    type Explanation = Explanation;

    fn query(&self, point: Point3<Real>) -> Option<Quality> {
        let mut quality: Option<Quality> = None;

        for (_, q) in priority_ray_iterator(&self.bvh_tree, &self.models, point) {
            quality = Some(match quality {
                None => q,
                Some(current) => {
                    if abs_diff_eq!(current.alpha, ALPHA_SATURATED, epsilon = ALPHA_EPS) {
                        return Some(current);
                    }
                    current.blend(&q)
                }
            });
        }

        quality
    }

    fn query_stats(&self, point: Point3<Real>) -> QueryStats {
        let now = Instant::now();

        let mut iter = priority_ray_stats_iterator(&self.bvh_tree, &self.models, point);
        let mut hits: Vec<(u8, Quality)> = Vec::new();
        while let Some(hit) = iter.next() {
            hits.push(hit);
        }
        let outer_stats = iter.stats;
        let mut hit_count = 0;
        let mut quality: Option<Quality> = None;

        for (_, q) in priority_ray_iterator(&self.bvh_tree, &self.models, point) {
            hit_count += 1;
            quality = Some(match quality {
                None => q,
                Some(current) => {
                    if abs_diff_eq!(current.alpha, ALPHA_SATURATED, epsilon = ALPHA_EPS) {
                        break;
                    }
                    current.blend(&q)
                }
            });
        }

        QueryStats {
            aabb_tests: outer_stats.aabb_tests,
            simplex_tests: outer_stats.simplex_tests,
            hit_count,
            output: quality,
            elapsed: now.elapsed().as_nanos(),
        }
    }

    fn query_explain(&self, point: Point3<Real>) -> Explanation {
        let mut contributions = Vec::new();
        let mut blended: Option<Quality> = None;
        let mut termination = None;

        for (i, (priority, q)) in
            priority_ray_iterator(&self.bvh_tree, &self.models, point).enumerate()
        {
            contributions.push(ModelContribution {
                priority,
                quality: q,
            });

            blended = Some(match blended {
                None => q,
                Some(ref current) => {
                    if abs_diff_eq!(current.alpha, ALPHA_SATURATED, epsilon = ALPHA_EPS)
                        && termination.is_none()
                    {
                        termination = Some(i);
                    }
                    current.blend(&q)
                }
            });
        }

        Explanation {
            contributions,
            output: blended,
            termination,
        }
    }
}
