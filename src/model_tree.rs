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

/// The fraction of the alpha channel considered "fully opaque" for early exit.
const ALPHA_SATURATED: Real = 1.0;
const ALPHA_EPS: Real = 1e-4;

/// Priority value that represents the lowest possible priority (include all models).
const PRIORITY_MIN: u8 = 0;
const PRIORITY_MAX: u8 = 255;

/// Serialisable top-level summary of a [`ModelTree`] for diagnostics.
#[derive(Serialize)]
pub struct ModelTreeView {
    pub models: Vec<MeshModelView>,
    /// Total in-memory size of the tree in bytes.
    pub size: usize,
}

/// A collection of [`MeshModel`]s indexed by a 4-D BVH.
///
/// The fourth BVH dimension encodes model priority, so a conceptual ray cast
/// along that axis visits models in ascending priority order (lowest priority
/// number first).  [`Query::query`] blends models using Porter-Duff compositing
/// and returns early once the accumulated alpha is saturated.
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
    /// Construct a `ModelTree` from a list of mesh models.
    ///
    /// Assigns consecutive IDs to models and builds the 4-D BVH.
    pub fn new(mut models: Vec<MeshModel>) -> Self {
        for (i, model) in models.iter_mut().enumerate() {
            model.id = i;
        }
        let bvh_tree = Bvh::build(&mut models);
        Self { bvh_tree, models }
    }

    /// Return the combined 3-D axis-aligned bounding box of all models.
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

/// Accumulate quality contributions from an iterator into an optional starting value.
///
/// This is the shared core used by [`Query::query`], [`Query::query_bounded`],
/// [`Query::query_into`], and [`Query::query_bounded_into`] to avoid duplicating
/// the Porter-Duff compositing loop.
fn blend_accumulate(
    mut quality: Option<Quality>,
    iter: impl Iterator<Item = (u8, Quality)>,
) -> Option<Quality> {
    for (_, q) in iter {
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

impl Query for ModelTree {
    type Explanation = Explanation;

    fn query(&self, point: Point3<Real>) -> Option<Quality> {
        blend_accumulate(
            None,
            priority_ray_iterator(
                &self.bvh_tree,
                &self.models,
                point,
                PRIORITY_MIN as Real,
                PRIORITY_MAX as Real,
            ),
        )
    }

    fn query_bounded(
        &self,
        point: Point3<Real>,
        priority_lo: u8,
        priority_hi: u8,
    ) -> Option<Quality> {
        blend_accumulate(
            None,
            priority_ray_iterator(
                &self.bvh_tree,
                &self.models,
                point,
                priority_lo as Real,
                priority_hi as Real,
            ),
        )
    }

    fn query_into(
        &self,
        point: Point3<Real>,
        existing: Option<Quality>,
    ) -> Option<Quality> {
        blend_accumulate(
            existing,
            priority_ray_iterator(
                &self.bvh_tree,
                &self.models,
                point,
                PRIORITY_MIN as Real,
                PRIORITY_MAX as Real,
            ),
        )
    }

    fn query_bounded_into(
        &self,
        point: Point3<Real>,
        existing: Option<Quality>,
        priority_lo: u8,
        priority_hi: u8,
    ) -> Option<Quality> {
        blend_accumulate(
            existing,
            priority_ray_iterator(
                &self.bvh_tree,
                &self.models,
                point,
                priority_lo as Real,
                priority_hi as Real,
            ),
        )
    }

    fn query_stats(&self, point: Point3<Real>) -> QueryStats {
        let now = Instant::now();

        let mut iter = priority_ray_stats_iterator(
            &self.bvh_tree,
            &self.models,
            point,
            PRIORITY_MIN as Real,
            PRIORITY_MAX as Real,
        );
        let mut hits: Vec<(u8, Quality)> = Vec::new();
        for hit in iter.by_ref() {
            hits.push(hit);
        }
        let outer_stats = iter.stats;
        let mut hit_count = 0;
        let quality = blend_accumulate(
            None,
            priority_ray_iterator(
                &self.bvh_tree,
                &self.models,
                point,
                PRIORITY_MIN as Real,
                PRIORITY_MAX as Real,
            )
            .inspect(|_| hit_count += 1),
        );

        QueryStats {
            aabb_tests: outer_stats.aabb_tests,
            simplex_tests: outer_stats.simplex_tests,
            hit_count,
            output: quality,
            elapsed: now.elapsed().as_nanos() as u64,
        }
    }

    fn query_explain(&self, point: Point3<Real>) -> Explanation {
        let mut contributions = Vec::new();
        let mut blended: Option<Quality> = None;
        let mut termination = None;

        for (i, (priority, q)) in priority_ray_iterator(
            &self.bvh_tree,
            &self.models,
            point,
            PRIORITY_MIN as Real,
            PRIORITY_MAX as Real,
        )
        .enumerate()
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{Point3, Point4};

    use crate::mesh::MeshModel;
    use crate::model::{ConstantModel, InterpolateModel, Model};
    use crate::quality::Quality;
    use crate::real::Real;

    fn unit_cube_mesh(priority: u8, quality_val: Real, alpha: Real) -> MeshModel {
        let vertices: Vec<Point3<Real>> = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
        ];
        let faces = vec![
            Point4::new(0usize, 1, 2, 4),
            Point4::new(3, 1, 2, 7),
            Point4::new(5, 1, 4, 7),
            Point4::new(6, 2, 4, 7),
            Point4::new(1, 2, 4, 7),
        ];
        let q = Quality { rho: quality_val, vp: quality_val, vs: quality_val, qp: quality_val, qs: quality_val, alpha };
        let qualities = vec![q; vertices.len()];
        let models = faces
            .iter()
            .map(|f| Model::from(InterpolateModel { qualities: *f }))
            .collect();
        MeshModel::new(vertices, faces, models, qualities, priority, None, String::new())
    }

    #[test]
    fn test_single_model_query_inside() {
        let mesh = unit_cube_mesh(0, 5.0, 1.0);
        let tree = ModelTree::new(vec![mesh]);
        let result = tree.query(Point3::new(0.5, 0.5, 0.5));
        assert!(result.is_some());
        let q = result.unwrap();
        assert_relative_eq!(q.rho, 5.0, epsilon = 1e-4);
    }

    #[test]
    fn test_query_outside_returns_none() {
        let mesh = unit_cube_mesh(0, 5.0, 1.0);
        let tree = ModelTree::new(vec![mesh]);
        let result = tree.query(Point3::new(10.0, 10.0, 10.0));
        assert!(result.is_none());
    }

    #[test]
    fn test_priority_ordering_higher_priority_wins() {
        let mesh0 = unit_cube_mesh(0, 5.0, 1.0);
        let mesh1 = unit_cube_mesh(1, 10.0, 1.0);
        let tree = ModelTree::new(vec![mesh0, mesh1]);
        let result = tree.query(Point3::new(0.5, 0.5, 0.5));
        assert!(result.is_some());
        let q = result.unwrap();
        assert_relative_eq!(q.rho, 5.0, epsilon = 1e-4);
    }

    #[test]
    fn test_priority_ordering_lower_number_first() {
        let mesh1 = unit_cube_mesh(1, 10.0, 1.0);
        let mesh0 = unit_cube_mesh(0, 5.0, 1.0);
        let tree = ModelTree::new(vec![mesh1, mesh0]);
        let result = tree.query(Point3::new(0.5, 0.5, 0.5));
        let q = result.unwrap();
        assert_relative_eq!(q.rho, 5.0, epsilon = 1e-4);
    }

    #[test]
    fn test_alpha_blending_partial_alpha() {
        let mesh0 = unit_cube_mesh(0, 0.0, 0.5);
        let mesh1 = unit_cube_mesh(1, 10.0, 1.0);
        let tree = ModelTree::new(vec![mesh0, mesh1]);
        let result = tree.query(Point3::new(0.5, 0.5, 0.5));
        assert!(result.is_some());
        let q = result.unwrap();
        assert_relative_eq!(q.rho, 5.0, epsilon = 1e-4);
        assert_relative_eq!(q.alpha, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_query_stats_counts() {
        let mesh = unit_cube_mesh(0, 5.0, 1.0);
        let tree = ModelTree::new(vec![mesh]);
        let stats = tree.query_stats(Point3::new(0.5, 0.5, 0.5));
        assert!(stats.hit_count >= 1);
        assert!(stats.output.is_some());
    }

    #[test]
    fn test_query_explain_contributions() {
        let mesh0 = unit_cube_mesh(0, 5.0, 0.5);
        let mesh1 = unit_cube_mesh(1, 10.0, 1.0);
        let tree = ModelTree::new(vec![mesh0, mesh1]);
        let explanation = tree.query_explain(Point3::new(0.5, 0.5, 0.5));
        assert!(explanation.contributions.len() >= 2);
        assert!(explanation.output.is_some());
        assert_eq!(explanation.contributions[0].priority, 0);
    }

    #[test]
    fn test_aabb_covers_all_models() {
        let mesh0 = unit_cube_mesh(0, 5.0, 1.0);
        let vertices: Vec<Point3<Real>> = vec![
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(3.0, 1.0, 0.0),
            Point3::new(2.0, 0.0, 1.0),
            Point3::new(3.0, 0.0, 1.0),
            Point3::new(2.0, 1.0, 1.0),
            Point3::new(3.0, 1.0, 1.0),
        ];
        let faces = vec![
            Point4::new(0usize, 1, 2, 4),
            Point4::new(3, 1, 2, 7),
            Point4::new(5, 1, 4, 7),
            Point4::new(6, 2, 4, 7),
            Point4::new(1, 2, 4, 7),
        ];
        let q = Quality { rho: 7.0, vp: 7.0, vs: 7.0, qp: 7.0, qs: 7.0, alpha: 1.0 };
        let qualities = vec![q; vertices.len()];
        let models = faces.iter().map(|f| Model::from(InterpolateModel { qualities: *f })).collect();
        let mesh1 = MeshModel::new(vertices, faces, models, qualities, 1, None, String::new());

        let tree = ModelTree::new(vec![mesh0, mesh1]);
        let aabb = tree.aabb();
        assert_relative_eq!(aabb.min.x, 0.0, epsilon = 1e-5);
        assert_relative_eq!(aabb.max.x, 3.0, epsilon = 1e-5);
        assert_relative_eq!(aabb.min.y, 0.0, epsilon = 1e-5);
        assert_relative_eq!(aabb.max.y, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_query_bounded_tomography_only() {
        // priority 10 = tomography range (0-127), priority 200 = basin range (128-255)
        let mesh_tomo = unit_cube_mesh(10, 5.0, 1.0);
        let mesh_basin = unit_cube_mesh(200, 99.0, 1.0);
        let tree = ModelTree::new(vec![mesh_tomo, mesh_basin]);
        let pt = Point3::new(0.5, 0.5, 0.5);

        // Query only tomography range: should return tomo value
        let result = tree.query_bounded(pt, 0, 127);
        assert!(result.is_some());
        assert_relative_eq!(result.unwrap().rho, 5.0, epsilon = 1e-4);

        // Query only basin range: should return basin value
        let result = tree.query_bounded(pt, 128, 255);
        assert!(result.is_some());
        assert_relative_eq!(result.unwrap().rho, 99.0, epsilon = 1e-4);
    }

    #[test]
    fn test_query_bounded_excludes_out_of_range() {
        let mesh = unit_cube_mesh(50, 7.0, 1.0);
        let tree = ModelTree::new(vec![mesh]);
        let pt = Point3::new(0.5, 0.5, 0.5);

        // Model has priority 50; query in range 128-255 should return None
        let result = tree.query_bounded(pt, 128, 255);
        assert!(result.is_none());

        // Query in range 0-127 should return the value
        let result = tree.query_bounded(pt, 0, 127);
        assert!(result.is_some());
        assert_relative_eq!(result.unwrap().rho, 7.0, epsilon = 1e-4);
    }

    #[test]
    fn test_query_bounded_exact_boundary() {
        let mesh_lo = unit_cube_mesh(127, 1.0, 1.0);
        let mesh_hi = unit_cube_mesh(128, 2.0, 1.0);
        let tree = ModelTree::new(vec![mesh_lo, mesh_hi]);
        let pt = Point3::new(0.5, 0.5, 0.5);

        // Priority 127 is the last value in 0-127 range
        let result = tree.query_bounded(pt, 0, 127);
        assert!(result.is_some());
        assert_relative_eq!(result.unwrap().rho, 1.0, epsilon = 1e-4);

        // Priority 128 is the first value in 128-255 range
        let result = tree.query_bounded(pt, 128, 255);
        assert!(result.is_some());
        assert_relative_eq!(result.unwrap().rho, 2.0, epsilon = 1e-4);
    }

    #[test]
    fn test_query_into_blends_with_existing() {
        // Priority-0 model: rho=4.0, alpha=0.5
        let mesh0 = unit_cube_mesh(0, 4.0, 0.5);
        let tree = ModelTree::new(vec![mesh0]);
        let pt = Point3::new(0.5, 0.5, 0.5);

        // Start with an existing quality (rho=10.0, alpha=0.5)
        let existing = Some(Quality {
            rho: 10.0,
            vp: 10.0,
            vs: 10.0,
            qp: 10.0,
            qs: 10.0,
            alpha: 0.5,
        });
        let result = tree.query_into(pt, existing);
        assert!(result.is_some());
        let q = result.unwrap();
        // existing.blend(&new_quality): existing takes precedence
        // alpha = 0.5 + 0.5*(1-0.5) = 0.75
        // rho = (0.5/0.75)*10.0 + (0.5*0.5/0.75)*4.0
        let expected_rho = (0.5 / 0.75) * 10.0 + (0.5 * 0.5 / 0.75) * 4.0;
        assert_relative_eq!(q.rho, expected_rho as Real, epsilon = 1e-4);
        assert_relative_eq!(q.alpha, 0.75, epsilon = 1e-4);
    }

    #[test]
    fn test_query_into_no_existing_matches_query() {
        let mesh = unit_cube_mesh(0, 6.0, 1.0);
        let tree = ModelTree::new(vec![mesh]);
        let pt = Point3::new(0.5, 0.5, 0.5);

        let from_query = tree.query(pt);
        let from_query_into = tree.query_into(pt, None);
        assert_eq!(from_query, from_query_into);
    }

    #[test]
    fn test_query_bounded_into_combines_two_queries() {
        // Simulate the Ely taper pattern:
        // 1. Query tomography (priority 0-127) → use as "existing"
        // 2. query_bounded_into basin (128-255) to blend basin over tomo
        let mesh_tomo = unit_cube_mesh(10, 5.0, 0.5);
        let mesh_basin = unit_cube_mesh(200, 9.0, 0.5);
        let tree = ModelTree::new(vec![mesh_tomo, mesh_basin]);
        let pt = Point3::new(0.5, 0.5, 0.5);

        let tomo = tree.query_bounded(pt, 0, 127);
        let result = tree.query_bounded_into(pt, tomo, 128, 255);
        assert!(result.is_some());
        // Both contribute; combined alpha should be 0.75
    }

    #[test]
    fn test_constant_model_query() {
        let vertices: Vec<Point3<Real>> = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];
        let faces = vec![Point4::new(0usize, 1, 2, 3)];
        let q = Quality { rho: 42.0, vp: 1.0, vs: 1.0, qp: 1.0, qs: 1.0, alpha: 1.0 };
        let qualities = vec![q];
        let models = vec![Model::from(ConstantModel { quality: 0usize })];
        let mesh = MeshModel::new(vertices, faces, models, qualities, 0, None, String::new());
        let tree = ModelTree::new(vec![mesh]);
        let pt = Point3::new(0.1, 0.1, 0.1);
        let result = tree.query(pt);
        assert!(result.is_some());
        assert_relative_eq!(result.unwrap().rho, 42.0, epsilon = 1e-4);
    }
}
