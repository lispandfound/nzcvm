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
        let mut models: Vec<&MeshModel> =
            contains_point_iterator(&self.bvh_tree, &self.models, &point).collect();

        if models.is_empty() {
            return None;
        }

        models.sort_by_key(|m| m.priority);

        let mut quality = models[0].query(point)?;
        for model in &models[1..] {
            if abs_diff_eq!(quality.alpha, 1.0, epsilon = 1e-4) {
                break;
            }
            if let Some(q) = model.query(point) {
                quality = quality.blend(&q);
            }
        }
        Some(quality)
    }

    fn query_stats(&self, point: Point3<Real>) -> QueryStats {
        let now = Instant::now();

        let mut iter = contains_point_stats_iterator(&self.bvh_tree, &self.models, &point);
        let mut models: Vec<&MeshModel> = Vec::new();
        while let Some(model) = iter.next() {
            models.push(model);
        }
        let outer_stats = iter.stats;
        let hit_count = models.len();

        models.sort_by_key(|m| m.priority);

        let quality = if hit_count > 0 {
            let mut q = models[0].query(point);
            for model in &models[1..] {
                let Some(ref current) = q else { break };
                if abs_diff_eq!(current.alpha, 1.0, epsilon = 1e-4) {
                    break;
                }
                if let Some(next_q) = model.query(point) {
                    q = Some(current.blend(&next_q));
                }
            }
            q
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
        let mut models: Vec<&MeshModel> =
            contains_point_iterator(&self.bvh_tree, &self.models, &point).collect();

        let mut contributions = Vec::new();
        let mut output = None;
        let mut termination = None;

        if !models.is_empty() {
            models.sort_by_key(|m| m.priority);

            let mut blended: Option<Quality> = None;

            for (i, model) in models.iter().enumerate() {
                if let Some(q) = model.query(point) {
                    contributions.push(ModelContribution {
                        priority: model.priority,
                        quality: q,
                    });

                    blended = Some(match blended {
                        None => q,
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
                            current.blend(&q)
                        }
                    });
                }
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
