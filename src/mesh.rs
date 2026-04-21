use crate::geometry::point_triangle_distance_sq;
use crate::quality::Quality;
use crate::real::Real;
use crate::tree_query::{contains_point_iterator, nearest_to_point_within, Contains};
use approx::abs_diff_eq;
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::Bvh;
use bvh::bvh::BvhNode;
use bvh::point_query::PointDistance;
use enum_dispatch::enum_dispatch;
use nalgebra::Scalar;
use nalgebra::{Matrix3, Point3, Point4};

#[derive(Debug, Clone, Copy)]
pub struct Simplex {
    pub c0: Point3<Real>,
    pub c1: Point3<Real>,
    pub c2: Point3<Real>,
    pub c3: Point3<Real>,

    inv_matrix: Matrix3<Real>,

    id: usize,
    node_index: usize,
    pub priority: u8,
}

impl Simplex {
    pub fn new(
        c0: Point3<Real>,
        c1: Point3<Real>,
        c2: Point3<Real>,
        c3: Point3<Real>,
        id: usize,
    ) -> Self {
        let m = Matrix3::from_columns(&[c0 - c3, c1 - c3, c2 - c3]);
        let inv_matrix = m.try_inverse().unwrap();

        Self {
            c0,
            c1,
            c2,
            c3,
            id,
            inv_matrix,
            node_index: 0,
            priority: 0,
        }
    }

    pub fn barycentric_coordinates(&self, p: Point3<Real>) -> Point4<Real> {
        let diff = p - self.c3;
        let l = self.inv_matrix * diff;

        let l0 = l[0];
        let l1 = l[1];
        let l2 = l[2];
        let l3 = 1.0 - l0 - l1 - l2;

        Point4::new(l0, l1, l2, l3)
    }
}

impl Contains<Real, 3> for Simplex {
    fn contains(&self, query_point: &Point3<Real>) -> bool {
        let bary = self.barycentric_coordinates(*query_point);
        bary.x >= 0.0 && bary.y >= 0.0 && bary.z >= 0.0 && bary.w >= 0.0
    }
}

impl PointDistance<Real, 3> for Simplex {
    fn distance_squared(&self, query_point: Point3<Real>) -> Real {
        let bary = self.barycentric_coordinates(query_point);

        if bary.x >= 0.0 && bary.y >= 0.0 && bary.z >= 0.0 && bary.w >= 0.0 {
            return 0.0;
        }

        let mut min_dist_sq = Real::INFINITY;

        if bary.x < 0.0 {
            min_dist_sq = min_dist_sq.min(point_triangle_distance_sq(
                query_point,
                self.c1,
                self.c2,
                self.c3,
            ));
        }
        if bary.y < 0.0 {
            min_dist_sq = min_dist_sq.min(point_triangle_distance_sq(
                query_point,
                self.c0,
                self.c2,
                self.c3,
            ));
        }
        if bary.z < 0.0 {
            min_dist_sq = min_dist_sq.min(point_triangle_distance_sq(
                query_point,
                self.c0,
                self.c1,
                self.c3,
            ));
        }
        if bary.w < 0.0 {
            min_dist_sq = min_dist_sq.min(point_triangle_distance_sq(
                query_point,
                self.c0,
                self.c1,
                self.c2,
            ));
        }

        min_dist_sq
    }
}

impl Bounded<Real, 3> for Simplex {
    fn aabb(&self) -> Aabb<Real, 3> {
        let pts = [self.c0, self.c1, self.c2, self.c3];
        let min_point = pts
            .iter()
            .copied()
            .reduce(|acc, p| Point3::new(acc.x.min(p.x), acc.y.min(p.y), acc.z.min(p.z)))
            .unwrap();
        let max_point = pts
            .iter()
            .copied()
            .reduce(|acc, p| Point3::new(acc.x.max(p.x), acc.y.max(p.y), acc.z.max(p.z)))
            .unwrap();
        Aabb::with_bounds(min_point, max_point)
    }
}

impl BHShape<Real, 3> for Simplex {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

#[enum_dispatch]
pub trait Queryable {
    fn quality_at(&self, qualities: &[Quality], simplex: &Simplex, point: &Point3<Real>)
        -> Quality;
    fn explanation(&self, qualities: &[Quality]) -> ModelExplanation;
}

#[enum_dispatch(Queryable)]
pub enum Model {
    Constant(ConstantModel<usize>),
    Interpolate(InterpolateModel<usize>),
}

pub enum ModelExplanation {
    Constant(ConstantModel<Quality>),
    Interpolate(InterpolateModel<Quality>),
}

pub struct ConstantModel<T> {
    pub quality: T,
}

impl Queryable for ConstantModel<usize> {
    fn quality_at(
        &self,
        qualities: &[Quality],
        _simplex: &Simplex,
        _point: &Point3<Real>,
    ) -> Quality {
        qualities[self.quality]
    }

    fn explanation(&self, qualities: &[Quality]) -> ModelExplanation {
        ModelExplanation::Constant(ConstantModel {
            quality: qualities[self.quality],
        })
    }
}

pub struct InterpolateModel<T: Scalar> {
    pub qualities: Point4<T>,
}

impl Queryable for InterpolateModel<usize> {
    fn quality_at(
        &self,
        qualities: &[Quality],
        simplex: &Simplex,
        point: &Point3<Real>,
    ) -> Quality {
        let bary = simplex.barycentric_coordinates(*point);
        let q0 = qualities[self.qualities.w];
        let q1 = qualities[self.qualities.x];
        let q2 = qualities[self.qualities.y];
        let q3 = qualities[self.qualities.z];
        q0 * bary.w + q1 * bary.x + q2 * bary.y + q3 * bary.z
    }

    fn explanation(&self, qualities: &[Quality]) -> ModelExplanation {
        ModelExplanation::Interpolate(InterpolateModel {
            qualities: self.qualities.map(|x| qualities[x]),
        })
    }
}

pub struct Explanation {
    pub simplices: Vec<Simplex>,
    pub qualities: Vec<Quality>,
    pub models: Vec<ModelExplanation>,
    pub output: Option<Quality>,
    pub termination: Option<usize>,
}

pub struct MeshModel {
    bvh_tree: Bvh<Real, 3>,
    simplices: Vec<Simplex>,
    model_map: Vec<Model>,
    qualities: Vec<Quality>,
    aabb: Aabb<Real, 3>,
}

impl MeshModel {
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
        let priority = vec![0; faces.len()];

        Self::new(vertices, faces, models, qualities, priority)
    }

    pub fn new(
        vertices: Vec<Point3<Real>>,
        faces: Vec<Point4<usize>>,
        models: Vec<Model>,
        qualities: Vec<Quality>,
        priority: Vec<u8>,
    ) -> Self {
        let min_point = vertices
            .iter()
            .fold(Point3::new(Real::MAX, Real::MAX, Real::MAX), |acc, p| {
                Point3::new(acc.x.min(p.x), acc.y.min(p.y), acc.z.min(p.z))
            });

        let max_point = vertices
            .iter()
            .fold(Point3::new(Real::MIN, Real::MIN, Real::MIN), |acc, p| {
                Point3::new(acc.x.max(p.x), acc.y.max(p.y), acc.z.max(p.z))
            });
        let aabb = Aabb::with_bounds(min_point, max_point);

        let mut simplices: Vec<Simplex> = faces
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let mut simplex = Simplex::new(
                    vertices[f.x],
                    vertices[f.y],
                    vertices[f.z],
                    vertices[f.w],
                    i,
                );

                simplex.priority = priority[i];
                simplex
            })
            .collect();
        let bvh_tree = Bvh::build(&mut simplices);

        Self {
            bvh_tree,
            simplices,
            qualities,
            aabb,
            model_map: models,
        }
    }

    pub fn points(&self) -> usize {
        self.qualities.len()
    }

    fn model_for(&self, simplex: &Simplex) -> ModelExplanation {
        self.model_map[simplex.id].explanation(&self.qualities)
    }

    fn quality_for(&self, simplex: &Simplex, point: &Point3<Real>) -> Quality {
        self.model_map[simplex.id].quality_at(&self.qualities, simplex, point)
    }

    pub fn query_within(&self, point: Point3<Real>, epsilon: Real) -> Option<(Quality, Real)> {
        // TODO: Accelerate this with the epsilon logic we use to prune the layer tree queries.
        nearest_to_point_within(&self.bvh_tree, &self.simplices, point, epsilon).map(
            |(simplex, dist)| {
                let q = self.quality_for(&simplex, &point);
                (q, dist)
            },
        )
    }

    pub fn explain(&self, point: Point3<Real>) -> Explanation {
        let mut simplices: Vec<&Simplex> =
            contains_point_iterator(&self.bvh_tree, &self.simplices, &point).collect();
        let mut query_simplices = Vec::new();
        let mut query_models = Vec::new();
        let mut query_qualities = Vec::new();
        let mut quality = None;
        let mut termination = None;

        if simplices.len() > 0 {
            simplices.sort_by_key(|simplex| simplex.priority);
            let mut computed_quality = self.quality_for(simplices[0], &point);
            query_simplices.push(*simplices[0]);
            query_models.push(self.model_for(simplices[0]));
            query_qualities.push(computed_quality);

            for i in 1..simplices.len() {
                if abs_diff_eq!(computed_quality.alpha, 1.0, epsilon = 1e-4)
                    && termination.is_none()
                {
                    termination = Some(i);
                }
                let new_quality = self.quality_for(simplices[i], &point);
                query_simplices.push(*simplices[i]);
                query_models.push(self.model_for(simplices[i]));
                query_qualities.push(new_quality);
                computed_quality = computed_quality.blend(&new_quality);
            }

            quality = Some(computed_quality);
        }

        Explanation {
            simplices: query_simplices,
            models: query_models,
            qualities: query_qualities,
            output: quality,
            termination: termination,
        }
    }

    pub fn query(&self, point: Point3<Real>) -> Option<Quality> {
        let mut simplices: Vec<&Simplex> =
            contains_point_iterator(&self.bvh_tree, &self.simplices, &point).collect();
        if simplices.len() == 1 {
            Some(self.quality_for(simplices[0], &point))
        } else if simplices.len() > 0 {
            simplices.sort_by_key(|simplex| simplex.priority);
            let mut quality = self.quality_for(simplices[0], &point);

            for i in 1..simplices.len() {
                if abs_diff_eq!(quality.alpha, 1.0, epsilon = 1e-4) {
                    break;
                }
                quality = quality.blend(&self.quality_for(simplices[i], &point));
            }
            Some(quality)
        } else {
            None
        }
    }

    pub fn pretty_print(&self) {
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
        println!(
            "Mesh model with {} vertices and {} simplices, tree depth = {}.",
            self.qualities.len(),
            self.simplices.len(),
            depth
        )
    }
}

impl Bounded<Real, 3> for MeshModel {
    fn aabb(&self) -> Aabb<Real, 3> {
        self.aabb
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
    fn test_simplex_distance_properties() {
        let v = unit_tetrahedron_universe();
        let simplex = Simplex::new(v[0], v[1], v[2], v[3], 0);

        assert_relative_eq!(
            simplex.distance_squared(Point3::new(0.1, 0.1, 0.1)),
            0.0,
            epsilon = 1e-5
        );

        let below_face = Point3::new(0.5, 0.5, -2.0);
        assert_relative_eq!(simplex.distance_squared(below_face), 4.0, epsilon = 1e-5);
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

        // Exterior: Query (6,4,4) while Max is (4,4,4). Dist = 2.0
        let p_out = Point3::new(6.0, 4.0, 4.0);
        let (q_out, dist_sq_out) = mesh
            .query_within(p_out, Real::MAX)
            .expect("Should extrapolate");

        assert_relative_eq!(dist_sq_out, 2.0, epsilon = 1e-5);
        assert_relative_eq!(q_out.rho, 14.0, epsilon = 1e-5);
    }
}
