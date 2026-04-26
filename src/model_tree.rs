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
        for hit in iter.by_ref() {
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
            elapsed: now.elapsed().as_nanos() as u64,
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
        let qualities = Quality::from_slice(&vec![q; vertices.len()]);
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
        let qualities = Quality::from_slice(&vec![q; vertices.len()]);
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
    fn test_constant_model_query() {
        let vertices: Vec<Point3<Real>> = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];
        let faces = vec![Point4::new(0usize, 1, 2, 3)];
        let q = Quality { rho: 42.0, vp: 1.0, vs: 1.0, qp: 1.0, qs: 1.0, alpha: 1.0 };
        let qualities = Quality::from_slice(&[q]);
        let models = vec![Model::from(ConstantModel { quality: 0usize })];
        let mesh = MeshModel::new(vertices, faces, models, qualities, 0, None, String::new());
        let tree = ModelTree::new(vec![mesh]);
        let pt = Point3::new(0.1, 0.1, 0.1);
        let result = tree.query(pt);
        assert!(result.is_some());
        assert_relative_eq!(result.unwrap().rho, 42.0, epsilon = 1e-4);
    }
}
