use crate::model::*;
use crate::quality::Quality;
use crate::real::Real;
use crate::simplex::Simplex;
use crate::tree_query::{Contains, contains_point_iterator};
use deepsize::{Context, DeepSizeOf};

use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::{BHShape, BoundingHierarchy};
use bvh::bvh::{Bvh, BvhNode};
use nalgebra::{Affine3, Point, Point3, Point4};
use serde::Serialize;

/// Default priority for models that do not specify one explicitly.
pub const DEFAULT_PRIORITY: u8 = 0;

/// Half-unit extent added to the priority dimension of the 4D AABB to keep
/// the AABB non-degenerate. The value 0.5 sits between any two consecutive
/// integer priorities, so BVH node AABBs correctly reflect the minimum and
/// maximum priority reachable through each subtree.
const PRIORITY_AABB_EXTENT: Real = 0.5;

/// Serialisable summary of a [`MeshModel`] for diagnostics.
#[derive(Serialize)]
pub struct MeshModelView {
    pub id: usize,
    pub name: String,
    pub bounds: [f32; 6],
    pub transform: Option<Affine3<Real>>,
    pub priority: u8,
    /// In-memory size of the model in bytes.
    pub size: usize,
}

/// A single tetrahedral mesh model with an internal BVH for fast point queries.
///
/// Each simplex in the mesh carries a [`Model`] that maps a query point to a
/// [`Quality`].  The outer [`ModelTree`](crate::model_tree::ModelTree) holds
/// a collection of `MeshModel`s and dispatches queries via a 4-D BVH (the
/// fourth dimension encodes priority).
pub struct MeshModel {
    bvh_tree: Bvh<Real, 3>,
    simplices: Vec<Simplex>,
    model_map: Vec<Model>,
    qualities: Vec<Quality>,
    aabb: Aabb<Real, 3>,
    transform: Option<Affine3<Real>>,
    pub priority: u8,
    /// Human-readable name for this mesh model.
    pub name: String,

    // BVH bookkeeping for the model tree
    pub id: usize,
    node_index: usize,
}

impl DeepSizeOf for MeshModel {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.simplices.deep_size_of_children(context)
            + self.model_map.deep_size_of_children(context)
            + self.qualities.deep_size_of_children(context)
            + self.bvh_tree.nodes.capacity() * size_of::<BvhNode<Real, 3>>()
            + self.name.deep_size_of_children(context)
    }
}

impl MeshModel {
    /// Build a curvilinear mesh by decomposing a structured grid into tetrahedra.
    ///
    /// Each voxel is split into 5 tetrahedra using the alternating-parity
    /// 5-simplex decomposition (Knuth, *TAOCP* Vol. 4 Fasc. 6).  All tetrahedra
    /// use [`InterpolateModel`] for barycentric quality interpolation.
    ///
    /// # Parameters
    ///
    /// * `vertices`   – `ni × nj × nk` grid points in some ordering defined by `chart`.
    /// * `qualities`  – one `Quality` per vertex.
    /// * `dimensions` – `(ni, nj, nk)` grid dimensions.
    /// * `chart`      – maps `(i, j, k)` grid indices to a vertex index.
    pub fn curvilinear_mesh<F>(
        vertices: Vec<Point3<Real>>,
        qualities: Vec<Quality>,
        dimensions: (usize, usize, usize),
        chart: F,
    ) -> Self
    where
        F: Fn(usize, usize, usize) -> usize,
    {
        let (ni, nj, nk) = dimensions;
        let mut faces = Vec::with_capacity(5 * (ni - 1) * (nj - 1) * (nk - 1));

        // See: The Art of Computer Programming, Volume 4, Fascicle 6 for the
        // 5-simplex decomposition of a cube.
        for i in 0..ni - 1 {
            for j in 0..nj - 1 {
                for k in 0..nk - 1 {
                    let v000 = chart(i, j, k);
                    let v100 = chart(i + 1, j, k);
                    let v010 = chart(i, j + 1, k);
                    let v110 = chart(i + 1, j + 1, k);
                    let v001 = chart(i, j, k + 1);
                    let v101 = chart(i + 1, j, k + 1);
                    let v011 = chart(i, j + 1, k + 1);
                    let v111 = chart(i + 1, j + 1, k + 1);

                    if (i + j + k) % 2 == 0 {
                        faces.push(Point4::new(v000, v100, v010, v001)); // Corner 0
                        faces.push(Point4::new(v110, v100, v010, v111)); // Corner 1
                        faces.push(Point4::new(v101, v100, v001, v111)); // Corner 2
                        faces.push(Point4::new(v011, v010, v001, v111)); // Corner 3
                        faces.push(Point4::new(v100, v010, v001, v111)); // Central Core
                    } else {
                        faces.push(Point4::new(v100, v000, v110, v101)); // Corner 0
                        faces.push(Point4::new(v010, v000, v110, v011)); // Corner 1
                        faces.push(Point4::new(v001, v000, v101, v011)); // Corner 2
                        faces.push(Point4::new(v111, v110, v101, v011)); // Corner 3
                        faces.push(Point4::new(v000, v110, v101, v011)); // Central Core
                    }
                }
            }
        }

        let models = faces
            .iter()
            .map(|q| Model::from(InterpolateModel { qualities: *q }))
            .collect();

        Self::new(
            vertices,
            faces,
            models,
            qualities,
            DEFAULT_PRIORITY,
            None,
            String::new(),
        )
    }

    /// Create a mesh model from raw geometry data.
    ///
    /// Builds an internal BVH over the simplices.  If `transform` is
    /// provided it is treated as a world-to-local affine map: query points
    /// are transformed into local space before the BVH is consulted, and
    /// the AABB is computed in world space by transforming vertices with the
    /// inverse.
    pub fn new(
        vertices: Vec<Point3<Real>>,
        faces: Vec<Point4<usize>>,
        models: Vec<Model>,
        qualities: Vec<Quality>,
        priority: u8,
        transform: Option<Affine3<Real>>,
        name: String,
    ) -> Self {
        let local_to_global_map = |p| transform.map_or(p, |aff| aff.inverse_transform_point(&p));
        let min_point =
            vertices
                .iter()
                .fold(Point3::new(Real::MAX, Real::MAX, Real::MAX), |acc, p| {
                    let transformed = local_to_global_map(*p);
                    Point3::new(
                        acc.x.min(transformed.x),
                        acc.y.min(transformed.y),
                        acc.z.min(transformed.z),
                    )
                });

        let max_point =
            vertices
                .iter()
                .fold(Point3::new(Real::MIN, Real::MIN, Real::MIN), |acc, p| {
                    let transformed = local_to_global_map(*p);
                    Point3::new(
                        acc.x.max(transformed.x),
                        acc.y.max(transformed.y),
                        acc.z.max(transformed.z),
                    )
                });
        let aabb = Aabb::with_bounds(min_point, max_point);

        let mut simplices: Vec<Simplex> = faces
            .iter()
            .enumerate()
            .map(|(i, f)| {
                Simplex::new(
                    vertices[f.x],
                    vertices[f.y],
                    vertices[f.z],
                    vertices[f.w],
                    i,
                )
            })
            .collect();
        let bvh_tree = Bvh::build_par(&mut simplices);

        Self {
            bvh_tree,
            simplices,
            qualities,
            aabb,
            model_map: models,
            priority,
            name,
            id: 0,
            node_index: 0,
            transform,
        }
    }

    /// Number of vertex-quality entries in this mesh.
    pub fn points(&self) -> usize {
        self.qualities.len()
    }

    fn quality_for(&self, simplex: &Simplex, point: &Point3<Real>) -> Quality {
        self.model_map[simplex.id].quality_at(&self.qualities, simplex, point)
    }

    /// Transform a point from world space into the mesh's local space.
    ///
    /// If no transform was supplied at construction the point is returned unchanged.
    pub fn global_to_local(&self, point: Point3<Real>) -> Point3<Real> {
        self.transform
            .map_or(point, |aff| aff.transform_point(&point))
    }

    /// Returns the quality at the given point using the first simplex that contains it.
    /// Returns `None` if no simplex contains the point.
    pub fn query(&self, point: Point3<Real>) -> Option<Quality> {
        let transformed = self.global_to_local(point);
        contains_point_iterator(&self.bvh_tree, &self.simplices, &transformed)
            .next()
            .map(|simplex| self.quality_for(&simplex, &transformed))
    }

    pub fn pretty_print(&self) {
        use bvh::bvh::BvhNode;
        fn max_depth(nodes: &[BvhNode<Real, 3>], node_index: usize) -> usize {
            match nodes[node_index] {
                BvhNode::Node {
                    child_l_index,
                    child_r_index,
                    ..
                } => (max_depth(nodes, child_l_index) + 1).max(max_depth(nodes, child_r_index) + 1),
                _ => 0,
            }
        }

        let depth = max_depth(&self.bvh_tree.nodes, 0);
        let name_display = if self.name.is_empty() {
            format!("(id={})", self.id)
        } else {
            format!("{:?}", self.name)
        };
        println!(
            "Mesh model {} with {} vertices and {} simplices, tree depth = {}, priority = {}.",
            name_display,
            self.qualities.len(),
            self.simplices.len(),
            depth,
            self.priority,
        )
    }

    pub fn view(&self) -> MeshModelView {
        let bounds = [
            self.aabb.min.x,
            self.aabb.min.y,
            self.aabb.min.z,
            self.aabb.max.x,
            self.aabb.max.y,
            self.aabb.max.z,
        ];
        MeshModelView {
            id: self.id,
            name: self.name.clone(),
            bounds,
            transform: self.transform,
            priority: self.priority,
            size: self.deep_size_of(),
        }
    }
}

/// `Contains` for `MeshModel` yields `(priority, quality)` when a simplex inside
/// this model contains the query point. This avoids a second BVH traversal in
/// the outer `ModelTree` – the traversal result is returned directly.
impl Contains<Real, 3, (u8, Quality)> for MeshModel {
    fn contains(&self, query_point: &Point3<Real>) -> Option<(u8, Quality)> {
        self.query(*query_point).map(|q| (self.priority, q))
    }
}

/// Returns the 3-D bounding box of this model's geometry.
/// Used when the caller needs the geometry AABB without the priority dimension.
impl MeshModel {
    pub fn aabb3(&self) -> Aabb<Real, 3> {
        self.aabb
    }
}

/// 4-D AABB used by the outer `ModelTree` BVH.
///
/// The first three dimensions are the model's geometry AABB. The fourth
/// dimension is the model's priority: both `min[3]` and `max[3]` are set to
/// `priority` (with a half-unit extent so the AABB is non-degenerate), which
/// lets the priority-ray traversal use `aabb.min[3]` as the `t_min` hit
/// distance and thereby visit models in priority order.
impl Bounded<Real, 4> for MeshModel {
    fn aabb(&self) -> Aabb<Real, 4> {
        let p = self.priority as Real;
        let min4 = Point::<Real, 4>::from(nalgebra::vector![
            self.aabb.min.x,
            self.aabb.min.y,
            self.aabb.min.z,
            p
        ]);
        let max4 = Point::<Real, 4>::from(nalgebra::vector![
            self.aabb.max.x,
            self.aabb.max.y,
            self.aabb.max.z,
            p + PRIORITY_AABB_EXTENT
        ]);
        Aabb::with_bounds(min4, max4)
    }
}

impl BHShape<Real, 4> for MeshModel {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    use nalgebra::{Point3, Point4};

    fn unit_tetrahedron_universe() -> Vec<Point3<Real>> {
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ]
    }

    fn mock_quality(val: Real) -> Quality {
        Quality {
            rho: val,
            vp: val,
            vs: val,
            qp: val,
            qs: val,
            alpha: 1.0,
        }
    }

    fn generate_grid(ni: usize, nj: usize, nk: usize) -> Vec<Point3<Real>> {
        let mut vertices = Vec::with_capacity(ni * nj * nk);
        for k in 0..nk {
            for j in 0..nj {
                for i in 0..ni {
                    vertices.push(Point3::new(i as Real, j as Real, k as Real));
                }
            }
        }
        vertices
    }

    #[test]
    fn test_simplex_barycentric_properties() {
        let v = unit_tetrahedron_universe();
        let simplex = Simplex::new(v[0], v[1], v[2], v[3], 0);

        let points_to_test = [Point3::new(0.25, 0.25, 0.25), Point3::new(10.0, -5.0, 2.0)];

        for p in points_to_test.iter() {
            let bary = simplex.barycentric_coordinates(*p);
            let sum = bary.x + bary.y + bary.z + bary.w;
            assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
        }

        // Test vertex identity
        assert_relative_eq!(
            simplex.barycentric_coordinates(v[0]),
            Point4::new(1.0, 0.0, 0.0, 0.0)
        );
        assert_relative_eq!(
            simplex.barycentric_coordinates(v[3]),
            Point4::new(0.0, 0.0, 0.0, 1.0)
        );
    }

    #[test]
    fn test_simplex_aabb_properties() {
        let v0 = Point3::new(-1.0, 2.0, 0.0);
        let v1 = Point3::new(3.0, -4.0, 1.0);
        let v2 = Point3::new(0.0, 0.0, 5.0);
        let v3 = Point3::new(1.0, 1.0, 1.0);

        let simplex = Simplex::new(v0, v1, v2, v3, 0);
        let aabb = simplex.aabb();

        assert_relative_eq!(aabb.min.x, -1.0);
        assert_relative_eq!(aabb.max.x, 3.0);
        assert_relative_eq!(aabb.min.z, 0.0);
        assert_relative_eq!(aabb.max.z, 5.0);
    }

    #[test]
    fn test_mesh_model_query_interpolation() {
        let ni = 2;
        let nj = 2;
        let nk = 2;
        let vertices = generate_grid(ni, nj, nk);
        let qualities: Vec<Quality> = (0..vertices.len())
            .map(|idx| mock_quality(idx as Real))
            .collect();

        let chart = |i, j, k| i + j * ni + k * ni * nj;
        let mesh = MeshModel::curvilinear_mesh(vertices, qualities, (ni, nj, nk), chart);

        // Vertex (1,1,1) is index 7 in a 2x2x2 grid
        let p_v7 = Point3::new(1.0, 1.0, 1.0);
        let q_v7 = mesh.query(p_v7).expect("Should find a simplex");

        assert_relative_eq!(q_v7.rho, 7.0, epsilon = 1e-5);
    }

    #[test]
    fn test_mesh_model_query_multi_cell_interpolation() {
        let ni = 5;
        let nj = 5;
        let nk = 5;
        let vertices = generate_grid(ni, nj, nk);
        let qualities: Vec<Quality> = vertices
            .iter()
            .map(|p| mock_quality(p.x + p.y + p.z))
            .collect();

        let chart = |i, j, k| i + j * ni + k * ni * nj;
        let mesh = MeshModel::curvilinear_mesh(vertices, qualities, (ni, nj, nk), chart);

        // Interior: x=2.5, y=1.2, z=3.7 -> rho = 7.4
        let p_in = Point3::new(2.5, 1.2, 3.7);
        let q_in = mesh.query(p_in).expect("Should find interior");

        assert_relative_eq!(q_in.rho, 7.4, epsilon = 1e-5);
    }

    #[test]
    fn test_mesh_model_query_outside_returns_none() {
        let v = unit_tetrahedron_universe();
        let quality = mock_quality(1.0);
        let faces = vec![Point4::new(0usize, 1, 2, 3)];
        let models = vec![Model::from(InterpolateModel {
            qualities: faces[0],
        })];
        let qualities = vec![quality; 4];
        let mesh = MeshModel::new(v, faces, models, qualities, 0, None, String::new());
        let q = mesh.query(Point3::new(5.0, 5.0, 5.0));
        assert!(q.is_none());
    }

    #[test]
    fn test_aabb_correct_for_simple_mesh() {
        let ni = 3;
        let nj = 3;
        let nk = 3;
        let vertices = generate_grid(ni, nj, nk);
        let qualities: Vec<Quality> = vertices.iter().map(|_| mock_quality(1.0)).collect();
        let chart = |i, j, k| i + j * ni + k * ni * nj;
        let mesh = MeshModel::curvilinear_mesh(vertices, qualities, (ni, nj, nk), chart);
        let aabb = mesh.aabb3();
        assert_relative_eq!(aabb.min.x, 0.0);
        assert_relative_eq!(aabb.max.x, 2.0);
        assert_relative_eq!(aabb.min.z, 0.0);
        assert_relative_eq!(aabb.max.z, 2.0);
    }

    #[test]
    fn test_mesh_model_interpolate_centroid() {
        let v = unit_tetrahedron_universe();
        let faces = vec![Point4::new(0usize, 1, 2, 3)];
        let qualities_vec: Vec<Quality> = (0..4).map(|i| mock_quality(i as Real)).collect();
        let models = vec![Model::from(InterpolateModel {
            qualities: faces[0],
        })];
        let mesh = MeshModel::new(v, faces, models, qualities_vec, 0, None, String::new());
        let q = mesh.query(Point3::new(0.25, 0.25, 0.25));
        assert!(q.is_some());
        // Centroid bary coords all equal 0.25; qualities are 0,1,2,3
        // InterpolateModel uses: q0=qualities[w=3]=3, q1=qualities[x=0]=0, q2=qualities[y=1]=1, q3=qualities[z=2]=2
        // result = 3*0.25 + 0*0.25 + 1*0.25 + 2*0.25 = 1.5
        assert_relative_eq!(q.unwrap().rho, 1.5, epsilon = 1e-4);
    }

    #[test]
    fn test_constant_model_returns_fixed_quality() {
        let v = unit_tetrahedron_universe();
        let faces = vec![Point4::new(0usize, 1, 2, 3)];
        let q_fixed = Quality {
            rho: 42.0,
            vp: 1.0,
            vs: 2.0,
            qp: 3.0,
            qs: 4.0,
            alpha: 1.0,
        };
        let qualities = vec![q_fixed];
        let models = vec![Model::from(ConstantModel { quality: 0usize })];
        let mesh = MeshModel::new(v, faces, models, qualities, 0, None, String::new());
        let result = mesh.query(Point3::new(0.2, 0.1, 0.1));
        assert!(result.is_some());
        let q_result = result.unwrap();
        assert_relative_eq!(q_result.rho, 42.0, epsilon = 1e-4);
        assert_relative_eq!(q_result.vp, 1.0, epsilon = 1e-4);
    }

    /// A world-to-local translation by (-5, 0, 0) places the unit tetrahedron at
    /// world position [5, 6] × [0, 1] × [0, 1].  Queries must use world coords.
    #[test]
    fn test_transform_translates_queries() {
        use nalgebra::{Affine3, Translation3};
        let v = unit_tetrahedron_universe();
        let faces = vec![Point4::new(0usize, 1, 2, 3)];
        let models = vec![Model::from(ConstantModel { quality: 0usize })];
        let qualities = vec![mock_quality(5.0)];

        // World-to-local: subtract 5 from x-coordinate.
        let aff: Affine3<Real> = Affine3::from_matrix_unchecked(
            Translation3::new(-5.0_f32, 0.0_f32, 0.0_f32).to_homogeneous(),
        );
        let mesh = MeshModel::new(v, faces, models, qualities, 0, Some(aff), String::new());

        // (5.1, 0.1, 0.1) in world → (0.1, 0.1, 0.1) in local → inside
        let q = mesh.query(Point3::new(5.1, 0.1, 0.1));
        assert!(q.is_some());
        assert_relative_eq!(q.unwrap().rho, 5.0, epsilon = 1e-4);

        // (0.1, 0.1, 0.1) in world → (-4.9, 0.1, 0.1) in local → outside
        let q_outside = mesh.query(Point3::new(0.1, 0.1, 0.1));
        assert!(q_outside.is_none());
    }

    /// After the translation, the AABB must reflect the tetrahedron's world position.
    #[test]
    fn test_transform_aabb_in_world_space() {
        use nalgebra::{Affine3, Translation3};
        let v = unit_tetrahedron_universe();
        let faces = vec![Point4::new(0usize, 1, 2, 3)];
        let models = vec![Model::from(ConstantModel { quality: 0usize })];
        let qualities = vec![mock_quality(1.0)];

        let aff: Affine3<Real> = Affine3::from_matrix_unchecked(
            Translation3::new(-5.0_f32, 0.0_f32, 0.0_f32).to_homogeneous(),
        );
        let mesh = MeshModel::new(v, faces, models, qualities, 0, Some(aff), String::new());
        let aabb = mesh.aabb3();

        // Tetrahedron spans [5,6] × [0,1] × [0,1] in world space.
        assert_relative_eq!(aabb.min.x, 5.0, epsilon = 1e-4);
        assert_relative_eq!(aabb.max.x, 6.0, epsilon = 1e-4);
        assert_relative_eq!(aabb.min.y, 0.0, epsilon = 1e-4);
        assert_relative_eq!(aabb.min.z, 0.0, epsilon = 1e-4);
    }
}
