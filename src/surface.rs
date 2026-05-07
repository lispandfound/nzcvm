use crate::real::Real;
use crate::tree_query::contains_point_iterator; // Assumed 2D iterator
use crate::triangle::Triangle; // Assumed 2D equivalent of Simplex
use deepsize::{Context, DeepSizeOf};

use bvh::bvh::{Bvh, BvhNode};
use nalgebra::{Point2, Point3};
use serde::Serialize;

/// Serialisable summary of a [`SurfaceModel`] for diagnostics.
#[derive(Serialize)]
pub struct SurfaceModelView {
    pub size: usize,
}

/// A 2D triangular surface model with an internal 2D BVH for fast elevation queries.
pub struct SurfaceModel {
    /// Internal BVH built in 2D space (X, Y)
    bvh_tree: Bvh<Real, 2>,
    triangles: Vec<Triangle>,
    model_map: Vec<Point3<usize>>,
    z: Vec<Real>,
}

impl DeepSizeOf for SurfaceModel {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.triangles.deep_size_of_children(context)
            + self.z.deep_size_of_children(context)
            + self.bvh_tree.nodes.capacity() * size_of::<BvhNode<Real, 2>>()
    }
}

impl SurfaceModel {
    /// Create a surface model from raw geometry.
    ///
    /// Note: `vertices` are 3D, but the internal BVH is built on their 2D projections.
    pub fn new(
        vertices: Vec<Point2<Real>>,
        faces: Vec<Point3<usize>>, // Triangle indices
        z: Vec<Real>,
    ) -> Self {
        // Build 2D Triangles for the internal BVH
        let mut triangles: Vec<Triangle> = faces
            .iter()
            .enumerate()
            .map(|(i, f)| Triangle::new(vertices[f.x], vertices[f.y], vertices[f.z], i))
            .collect();

        // BVH is built in 2D space (Triangles must implement Bounded<Real, 2>)
        let bvh_tree = Bvh::build(&mut triangles);

        Self {
            bvh_tree,
            triangles,
            z,
            model_map: faces,
        }
    }

    /// Interpolate quality at (x, y) by finding the triangle and using barycentric coordinates.
    pub fn query(&self, point: Point2<Real>) -> Option<Real> {
        contains_point_iterator(&self.bvh_tree, &self.triangles, &point)
            .next()
            .map(|tri| {
                let bary = tri.barycentric_coordinates(&point);
                let model = self.model_map[tri.id];
                let z0 = self.z[model.x];
                let z1 = self.z[model.y];
                let z2 = self.z[model.z];
                bary.x * z0 + bary.y * z1 + bary.z * z2
            })
    }

    pub fn view(&self) -> SurfaceModelView {
        SurfaceModelView {
            size: self.deep_size_of(),
        }
    }
}
