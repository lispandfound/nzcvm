use crate::geometry::point_triangle_distance_sq;
use crate::quality::Quality;
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::Bvh;
use bvh::bvh::BvhNode;
use bvh::point_query::PointDistance;
use core::f32;
use hdf5_metno::File;
use nalgebra::{Matrix3, Point3, Point4};

#[derive(Debug)]
struct Simplex {
    c0: Point3<f32>,
    c1: Point3<f32>,
    c2: Point3<f32>,
    c3: Point3<f32>,

    // Pre-calculated inverse matrix for fast projection
    inv_matrix: Matrix3<f32>,
    id: usize,
    node_index: usize,
}

impl Simplex {
    pub fn new(
        c0: Point3<f32>,
        c1: Point3<f32>,
        c2: Point3<f32>,
        c3: Point3<f32>,
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
        }
    }

    pub fn barycentric_coordinates(&self, p: Point3<f32>) -> Point4<f32> {
        let diff = p - self.c3;
        let l = self.inv_matrix * diff;

        let l0 = l[0];
        let l1 = l[1];
        let l2 = l[2];
        let l3 = 1.0 - l0 - l1 - l2;

        Point4::new(l0, l1, l2, l3)
    }
}

impl PointDistance<f32, 3> for Simplex {
    fn distance_squared(&self, query_point: Point3<f32>) -> f32 {
        let bary = self.barycentric_coordinates(query_point);

        // 1. Inside check: All barycentric coords non-negative
        if bary.x >= 0.0 && bary.y >= 0.0 && bary.z >= 0.0 && bary.w >= 0.0 {
            return 0.0;
        }

        // 2. Outside: Min distance to faces with negative weights
        let mut min_dist_sq = f32::INFINITY;

        if bary.x < 0.0 {
            // Face opposite c0: (c1, c2, c3)
            min_dist_sq = min_dist_sq.min(point_triangle_distance_sq(
                query_point,
                self.c1,
                self.c2,
                self.c3,
            ));
        }
        if bary.y < 0.0 {
            // Face opposite c1: (c0, c2, c3)
            min_dist_sq = min_dist_sq.min(point_triangle_distance_sq(
                query_point,
                self.c0,
                self.c2,
                self.c3,
            ));
        }
        if bary.z < 0.0 {
            // Face opposite c2: (c0, c1, c3)
            min_dist_sq = min_dist_sq.min(point_triangle_distance_sq(
                query_point,
                self.c0,
                self.c1,
                self.c3,
            ));
        }
        if bary.w < 0.0 {
            // Face opposite c3: (c0, c1, c2)
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

impl Bounded<f32, 3> for Simplex {
    fn aabb(&self) -> Aabb<f32, 3> {
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

impl BHShape<f32, 3> for Simplex {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

pub struct MeshModel {
    bvh_tree: Bvh<f32, 3>,
    simplices: Vec<Simplex>,
    vertex_map: Vec<Point4<usize>>,
    qualities: Vec<Quality>,
}

impl MeshModel {
    pub fn curvilinear_mesh<F>(
        vertices: Vec<Point3<f32>>,
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
        Self::new(vertices, faces, qualities)
    }

    fn new(vertices: Vec<Point3<f32>>, faces: Vec<Point4<usize>>, qualities: Vec<Quality>) -> Self {
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
        let bvh_tree = Bvh::build(&mut simplices);

        Self {
            bvh_tree,
            simplices,
            qualities,
            vertex_map: faces,
        }
    }

    pub fn query(&self, point: Point3<f32>) -> Option<(Quality, f32)> {
        self.bvh_tree
            .nearest_to(point, &self.simplices)
            .and_then(|(simplex, dist)| {
                let bary = simplex.barycentric_coordinates(point);
                let vertex_indices = self.vertex_map[simplex.id];
                let q0 = self.qualities[vertex_indices.w];
                let q1 = self.qualities[vertex_indices.x];
                let q2 = self.qualities[vertex_indices.y];
                let q3 = self.qualities[vertex_indices.z];
                let q = q0 * bary.w + q1 * bary.x + q2 * bary.y + q3 * bary.z;
                Some((q, dist))
            })
    }
    pub fn pretty_print(&self) -> () {
        fn max_depth(nodes: &[BvhNode<f32, 3>], node_index: usize) -> usize {
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

pub fn load_mesh_from_hdf5(file_path: &str) -> Result<MeshModel, hdf5_metno::Error> {
    let file = File::open(file_path)?;
    // Read datasets as dynamic-dimensional arrays
    let ds_x = file.dataset("X_NZTM")?.read_dyn::<f64>()?;
    let ds_y = file.dataset("Y_NZTM")?.read_dyn::<f64>()?;
    let ds_z = file.dataset("Z_meters")?.read_dyn::<f64>()?;
    let ds_vp = file.dataset("vp")?.read_dyn::<f64>()?;
    let ds_vs = file.dataset("vs")?.read_dyn::<f64>()?;
    let ds_rho = file.dataset("rho")?.read_dyn::<f64>()?;

    let shape = ds_x.shape();
    if shape.len() != 3 {
        return Err("Dataset must be 3D".into());
    }

    let (nz, ny, nx) = (shape[0], shape[1], shape[2]);
    let mut vertices_buf = Vec::with_capacity(nx * ny * nz);
    let mut qualities_buf = Vec::with_capacity(nx * ny * nz);

    // Iterate through the 3D grid and flatten into our vectors
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = [z, y, x];

                vertices_buf.push(Point3::new(
                    ds_x[&idx[..]] as f32,
                    ds_y[&idx[..]] as f32,
                    ds_z[&idx[..]] as f32,
                ));

                qualities_buf.push(Quality {
                    vp: ds_vp[&idx[..]] as f32,
                    vs: ds_vs[&idx[..]] as f32,
                    rho: ds_rho[&idx[..]] as f32,
                    qp: 100.0,
                    qs: 50.0,
                });
            }
        }
    }

    // Chart function to map 3D grid coords to the flat index in vertices_buf
    let chart = move |z: usize, y: usize, x: usize| -> usize { (z * ny * nx) + (y * nx) + x };

    Ok(MeshModel::curvilinear_mesh(
        vertices_buf,
        qualities_buf,
        (nz, ny, nx),
        chart,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{Point3, Point4};

    // --- Helpers ---

    fn unit_tetrahedron_universe() -> Vec<Point3<f32>> {
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ]
    }

    fn mock_quality(val: f32) -> Quality {
        Quality {
            rho: val,
            vp: val,
            vs: val,
            qp: val,
            qs: val,
        }
    }

    fn generate_grid(ni: usize, nj: usize, nk: usize) -> Vec<Point3<f32>> {
        let mut vertices = Vec::with_capacity(ni * nj * nk);
        for k in 0..nk {
            for j in 0..nj {
                for i in 0..ni {
                    vertices.push(Point3::new(i as f32, j as f32, k as f32));
                }
            }
        }
        vertices
    }

    // --- Simplex Unit Tests ---

    #[test]
    fn test_simplex_barycentric_properties() {
        let v = unit_tetrahedron_universe();
        // API Change: Pass points directly, no more universe index + reference
        let simplex = Simplex::new(v[0], v[1], v[2], v[3], 0);

        let points_to_test = [
            Point3::new(0.25, 0.25, 0.25), // Centroid
            Point3::new(10.0, -5.0, 2.0),  // Far outside
        ];

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

        // Inside point
        assert_relative_eq!(
            simplex.distance_squared(Point3::new(0.1, 0.1, 0.1)),
            0.0,
            epsilon = 1e-5
        );

        // Outside point: 2.0 units below the z=0 face. DistSq = 4.0
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

    // --- MeshModel Integration Tests ---

    #[test]
    fn test_mesh_model_query_interpolation() {
        let ni = 2;
        let nj = 2;
        let nk = 2;
        let vertices = generate_grid(ni, nj, nk);
        let qualities: Vec<Quality> = (0..vertices.len())
            .map(|idx| mock_quality(idx as f32))
            .collect();

        // Assuming curvilinear_mesh now consumes the vectors
        let chart = |i, j, k| i + j * ni + k * ni * nj;
        let mesh = MeshModel::curvilinear_mesh(vertices, qualities, (ni, nj, nk), chart);

        // Vertex (1,1,1) is index 7 in a 2x2x2 grid
        let p_v7 = Point3::new(1.0, 1.0, 1.0);
        let (q_v7, dist_sq) = mesh.query(p_v7).expect("Should find a simplex");

        assert_relative_eq!(dist_sq, 0.0, epsilon = 1e-5);
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
        let (q_in, dist_sq) = mesh.query(p_in).expect("Should find interior");

        assert_relative_eq!(dist_sq, 0.0, epsilon = 1e-5);
        assert_relative_eq!(q_in.rho, 7.4, epsilon = 1e-5);

        // Exterior: Query (6,4,4) while Max is (4,4,4). Dist = 2.0, DistSq = 4.0
        let p_out = Point3::new(6.0, 4.0, 4.0);
        let (q_out, dist_sq_out) = mesh.query(p_out).expect("Should extrapolate");

        assert_relative_eq!(dist_sq_out, 2.0, epsilon = 1e-5);
        assert_relative_eq!(q_out.rho, 14.0, epsilon = 1e-5);
    }
}
