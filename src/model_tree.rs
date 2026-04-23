use std::time::Instant;

use approx::abs_diff_eq;
use bvh::aabb::Bounded;
use bvh::bvh::Bvh;

use crate::mesh::MeshModel;
use crate::quality::Quality;
use crate::query::{Explanation, ModelContribution, Query, QueryStats};
use crate::real::Real;
use crate::tree_query::{contains_point_iterator, contains_point_stats_iterator};
use nalgebra::Point3;

pub struct ModelTree {
    bvh_tree: Bvh<Real, 3>,
    models: Vec<MeshModel>,
}

impl ModelTree {
    pub fn new(mut models: Vec<MeshModel>) -> Self {
        // Assign stable ids before building the BVH (which sets node_index).
        for (i, model) in models.iter_mut().enumerate() {
            model.id = i;
        }
        let bvh_tree = Bvh::build(&mut models);
        Self { bvh_tree, models }
    }

    pub fn aabb(&self) -> bvh::aabb::Aabb<Real, 3> {
        self.models
            .iter()
            .map(|m| m.aabb())
            .reduce(|a, b| a.join(&b))
            .expect("ModelTree must contain at least one model")
    }

    pub fn pretty_print(&self) {
        println!("ModelTree with {} mesh models:", self.models.len());
        for model in &self.models {
            model.pretty_print();
        }
    }
}

impl Query for ModelTree {
    type Explanation = Explanation;

    fn query(&self, point: Point3<Real>) -> Option<Quality> {
        // The iterator yields (priority, quality) pairs – no second traversal needed.
        let mut hits: Vec<(u8, Quality)> =
            contains_point_iterator(&self.bvh_tree, &self.models, &point).collect();

        if hits.is_empty() {
            return None;
        }

        hits.sort_by_key(|(priority, _)| *priority);

        let mut quality = hits[0].1;
        for (_, q) in &hits[1..] {
            if abs_diff_eq!(quality.alpha, 1.0, epsilon = 1e-4) {
                break;
            }
            quality = quality.blend(q);
        }
        Some(quality)
    }

    fn query_stats(&self, point: Point3<Real>) -> QueryStats {
        let now = Instant::now();

        let mut iter = contains_point_stats_iterator(&self.bvh_tree, &self.models, &point);
        let mut hits: Vec<(u8, Quality)> = Vec::new();
        while let Some(hit) = iter.next() {
            hits.push(hit);
        }
        let outer_stats = iter.stats;
        let hit_count = hits.len();

        hits.sort_by_key(|(priority, _)| *priority);

        let quality = if !hits.is_empty() {
            let mut q = hits[0].1;
            for (_, next_q) in &hits[1..] {
                if abs_diff_eq!(q.alpha, 1.0, epsilon = 1e-4) {
                    break;
                }
                q = q.blend(next_q);
            }
            Some(q)
        } else {
            None
        };

        QueryStats {
            aabb_tests: outer_stats.aabb_tests,
            simplex_tests: outer_stats.simplex_tests,
            hit_count,
            output: quality,
            elapsed: now.elapsed().as_nanos(),
        }
    }

    fn query_explain(&self, point: Point3<Real>) -> Explanation {
        // Collect (priority, quality) pairs in a single pass over the BVH.
        let mut hits: Vec<(u8, Quality)> =
            contains_point_iterator(&self.bvh_tree, &self.models, &point).collect();

        let mut contributions = Vec::new();
        let mut output = None;
        let mut termination = None;

        if !hits.is_empty() {
            hits.sort_by_key(|(priority, _)| *priority);

            let mut blended: Option<Quality> = None;

            for (i, (priority, q)) in hits.iter().enumerate() {
                contributions.push(ModelContribution {
                    priority: *priority,
                    quality: *q,
                });

                blended = Some(match blended {
                    None => *q,
                    Some(ref current) => {
                        // Record the index at which the blended quality was
                        // already saturated (alpha ≈ 1).  The contribution at
                        // this index is still included in the output for
                        // informational purposes but is marked inactive by
                        // callers via `termination` (i.e. index < termination
                        // ⇒ active).
                        if abs_diff_eq!(current.alpha, 1.0, epsilon = 1e-4)
                            && termination.is_none()
                        {
                            termination = Some(i);
                        }
                        current.blend(q)
                    }
                });
            }

            output = blended;
        }

        Explanation {
            contributions,
            output,
            termination,
        }
    }
}
