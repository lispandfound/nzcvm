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
pub struct Simplex<'a> {
    pub c0: usize,
    pub c1: usize,
    pub c2: usize,
    pub c3: usize,
    universe: &'a [Point3<f32>],
    node_index: usize,
    // Pre-calculated inverse matrix for fast projection
    inv_matrix: Matrix3<f32>,
}

impl<'a> Simplex<'a> {
    pub fn new(
        c0: usize,
        c1: usize,
        c2: usize,
        c3: usize,
        universe: &'a [Point3<f32>],
        node_index: usize,
    ) -> Self {
        let m = Matrix3::from_columns(&[
            universe[c0] - universe[c3],
            universe[c1] - universe[c3],
            universe[c2] - universe[c3],
        ]);
        let inv_matrix = m.try_inverse().unwrap();

        Self {
            c0,
            c1,
            c2,
            c3,
            universe,
            node_index,
            inv_matrix,
        }
    }

    fn points(&self) -> [Point3<f32>; 4] {
        [
            self.universe[self.c0],
            self.universe[self.c1],
            self.universe[self.c2],
            self.universe[self.c3],
        ]
    }

    pub fn barycentric_coordinates(&self, p: Point3<f32>) -> Point4<f32> {
        let diff = p - self.universe[self.c3];
        let l = self.inv_matrix * diff;

        let l0 = l[0];
        let l1 = l[1];
        let l2 = l[2];
        let l3 = 1.0 - l0 - l1 - l2;

        Point4::new(l0, l1, l2, l3)
    }
}

impl<'a> PointDistance<f32, 3> for Simplex<'a> {
    fn distance_squared(&self, query_point: Point3<f32>) -> f32 {
        let bary = self.barycentric_coordinates(query_point);
        let pts = self.points();

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
                pts[1],
                pts[2],
                pts[3],
            ));
        }
        if bary.y < 0.0 {
            // Face opposite c1: (c0, c2, c3)
            min_dist_sq = min_dist_sq.min(point_triangle_distance_sq(
                query_point,
                pts[0],
                pts[2],
                pts[3],
            ));
        }
        if bary.z < 0.0 {
            // Face opposite c2: (c0, c1, c3)
            min_dist_sq = min_dist_sq.min(point_triangle_distance_sq(
                query_point,
                pts[0],
                pts[1],
                pts[3],
            ));
        }
        if bary.w < 0.0 {
            // Face opposite c3: (c0, c1, c2)
            min_dist_sq = min_dist_sq.min(point_triangle_distance_sq(
                query_point,
                pts[0],
                pts[1],
                pts[2],
            ));
        }

        min_dist_sq
    }
}

impl<'a> Bounded<f32, 3> for Simplex<'a> {
    fn aabb(&self) -> Aabb<f32, 3> {
        let pts = self.points();
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

impl<'a> BHShape<f32, 3> for Simplex<'a> {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

pub struct MeshModel<'a> {
    bvh_tree: Bvh<f32, 3>,
    simplices: Vec<Simplex<'a>>,
    qualities: &'a [Quality],
}

impl<'a> MeshModel<'a> {
    pub fn curvilinear_mesh<F>(
        vertices: &'a [Point3<f32>],
        qualities: &'a [Quality],
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
        Self::new(vertices, &faces, qualities)
    }

    fn new(vertices: &'a [Point3<f32>], faces: &[Point4<usize>], qualities: &'a [Quality]) -> Self {
        let mut simplices: Vec<Simplex<'a>> = faces
            .iter()
            .map(|f| Simplex::new(f.x, f.y, f.z, f.w, vertices, 0))
            .collect();

        let bvh_tree = Bvh::build(&mut simplices);

        Self {
            bvh_tree,
            simplices,
            qualities,
        }
    }

    pub fn query(&self, point: Point3<f32>) -> Option<(Quality, f32)> {
        self.bvh_tree
            .nearest_to(point, &self.simplices)
            .and_then(|(simplex, dist)| {
                let bary = simplex.barycentric_coordinates(point);

                let points = simplex.points();
                let q0 = self.qualities[simplex.c0];
                let q1 = self.qualities[simplex.c1];
                let q2 = self.qualities[simplex.c2];
                let q3 = self.qualities[simplex.c3];
                let q = q0 * bary.x + q1 * bary.y + q2 * bary.z + q3 * bary.w;

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
        fn leaf_count(nodes: &[BvhNode<f32, 3>], node_index: usize) -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{Point3, Point4};

    // =========================================================================
    // Test Helpers (DRY)
    // =========================================================================

    /// Creates a standard unit right tetrahedron at the origin.
    fn unit_tetrahedron_universe() -> Vec<Point3<f32>> {
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ]
    }

    /// Helper to mock a quality struct. Assumes `rho` is a representative field.
    fn mock_quality(val: f32) -> Quality {
        Quality {
            rho: val,
            vp: val,
            vs: val,
            qp: val,
            qs: val,
        }
    }

    /// Helper to generate a basic 3D grid of points for mesh testing.
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

    // =========================================================================
    // Simplex Unit Tests
    // =========================================================================

    #[test]
    fn test_simplex_barycentric_properties() {
        let universe = unit_tetrahedron_universe();
        let simplex = Simplex::new(0, 1, 2, 3, &universe, 0);

        // Property 1: The sum of barycentric coordinates for ANY point must be 1.0
        let points_to_test = [
            Point3::new(0.25, 0.25, 0.25), // Centroid
            Point3::new(10.0, -5.0, 2.0),  // Far outside
            Point3::new(0.5, 0.5, 0.0),    // On an edge
        ];

        for p in points_to_test.iter() {
            let bary = simplex.barycentric_coordinates(*p);
            let sum = bary.x + bary.y + bary.z + bary.w;
            assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
        }

        // Property 2: A query exactly at a vertex should yield 1.0 for that vertex and 0.0 for others
        let bary_v0 = simplex.barycentric_coordinates(universe[0]);
        assert_relative_eq!(bary_v0, Point4::new(1.0, 0.0, 0.0, 0.0), epsilon = 1e-5);

        let bary_v3 = simplex.barycentric_coordinates(universe[3]);
        assert_relative_eq!(bary_v3, Point4::new(0.0, 0.0, 0.0, 1.0), epsilon = 1e-5);
    }

    #[test]
    fn test_simplex_distance_properties() {
        let universe = unit_tetrahedron_universe();
        let simplex = Simplex::new(0, 1, 2, 3, &universe, 0);

        // Property 1: Points inside or on the boundary of the simplex have a distance of 0.0
        let inside_pts = [
            Point3::new(0.1, 0.1, 0.1),
            Point3::new(0.0, 0.0, 0.0), // Origin vertex
            Point3::new(0.5, 0.5, 0.0), // Face center
        ];
        for p in inside_pts.iter() {
            assert_relative_eq!(simplex.distance_squared(*p), 0.0, epsilon = 1e-5);
        }

        // Property 2: Distance is strictly non-negative for points outside
        let outside_p = Point3::new(-1.0, -1.0, -1.0);
        let dist_sq = simplex.distance_squared(outside_p);
        assert!(dist_sq > 0.0);

        // Property 3: Distance squared to a plane matching a face should match theoretical Euclidean geometry
        // The point (0.5, 0.5, -2.0) is exactly 2.0 units directly "below" the z=0 face.
        // The distance squared should be 4.0.
        let below_face = Point3::new(0.5, 0.5, -2.0);
        assert_relative_eq!(simplex.distance_squared(below_face), 4.0, epsilon = 1e-5);
    }

    #[test]
    fn test_simplex_aabb_properties() {
        let universe = vec![
            Point3::new(-1.0, 2.0, 0.0),
            Point3::new(3.0, -4.0, 1.0),
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(1.0, 1.0, 1.0),
        ];
        let simplex = Simplex::new(0, 1, 2, 3, &universe, 0);
        let aabb = simplex.aabb();

        // Property: The AABB must correctly bound the geometric extrema of the vertices
        assert_relative_eq!(aabb.min.x, -1.0);
        assert_relative_eq!(aabb.max.x, 3.0);

        assert_relative_eq!(aabb.min.y, -4.0);
        assert_relative_eq!(aabb.max.y, 2.0);

        assert_relative_eq!(aabb.min.z, 0.0);
        assert_relative_eq!(aabb.max.z, 5.0);
    }

    // =========================================================================
    // MeshModel Integration Tests
    // =========================================================================

    #[test]
    fn test_curvilinear_mesh_generation_properties() {
        let ni = 3;
        let nj = 3;
        let nk = 3;
        let vertices = generate_grid(ni, nj, nk);
        let qualities = vec![mock_quality(1.0); vertices.len()];

        // Chart: Map 3D index to 1D flat array index
        let chart = |i, j, k| i + j * ni + k * ni * nj;

        let mesh = MeshModel::curvilinear_mesh(&vertices, &qualities, (ni, nj, nk), chart);

        // Property 1: The total number of tetrahedra created by the 5-tet decomposition
        // Formula: 5 tetrahedra per hexahedron cell.
        // Cells = (ni - 1) * (nj - 1) * (nk - 1)
        let expected_cells = (ni - 1) * (nj - 1) * (nk - 1);
        let expected_tets = 5 * expected_cells;

        assert_eq!(mesh.simplices.len(), expected_tets);
    }

    #[test]
    fn test_mesh_model_query_interpolation() {
        // Create a 2x2x2 grid (1 cell, 5 tetrahedra)
        let ni = 2;
        let nj = 2;
        let nk = 2;
        let vertices = generate_grid(ni, nj, nk);

        // Map vertex index to a quality where rho = its flat index (for traceable interpolation)
        let qualities: Vec<Quality> = (0..vertices.len())
            .map(|idx| mock_quality(idx as f32))
            .collect();

        let chart = |i, j, k| i + j * ni + k * ni * nj;
        let mesh = MeshModel::curvilinear_mesh(&vertices, &qualities, (ni, nj, nk), chart);

        // Property 1: Querying exactly at a vertex should yield exactly that vertex's quality.
        // Vertex (0,0,0) is at index 0.
        let p_v0 = Point3::new(0.0, 0.0, 0.0);
        let (q_v0, dist_v0) = mesh.query(p_v0).expect("Should find a simplex");

        assert_relative_eq!(dist_v0, 0.0, epsilon = 1e-5);
        assert_relative_eq!(q_v0.rho, 0.0, epsilon = 1e-5);

        // Vertex (1,1,1) is at index 7.
        let p_v7 = Point3::new(1.0, 1.0, 1.0);
        let (q_v7, dist_v7) = mesh.query(p_v7).expect("Should find a simplex");

        assert_relative_eq!(dist_v7, 0.0, epsilon = 1e-5);
        assert_relative_eq!(q_v7.rho, 7.0, epsilon = 1e-5);

        // Property 2: A point strictly inside the grid must have distance 0
        let p_mid = Point3::new(0.5, 0.5, 0.5);
        let (_, dist_mid) = mesh.query(p_mid).expect("Should find a simplex");
        assert_relative_eq!(dist_mid, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_mesh_model_query_exterior() {
        let ni = 2;
        let nj = 2;
        let nk = 2;
        let vertices = generate_grid(ni, nj, nk);
        let qualities = vec![mock_quality(42.0); vertices.len()];
        let chart = |i, j, k| i + j * ni + k * ni * nj;

        let mesh = MeshModel::curvilinear_mesh(&vertices, &qualities, (ni, nj, nk), chart);

        // Property: Querying outside the mesh bounds should still return a result
        // via nearest-neighbor projection, with a mathematically correct distance squared.
        // Max bounds of mesh are (1.0, 1.0, 1.0). Query at (3.0, 1.0, 1.0).
        // Distance strictly along X axis = 2.0. Distance squared = 4.0.
        let outside_p = Point3::new(3.0, 1.0, 1.0);
        let (q_ext, dist_ext) = mesh.query(outside_p).expect("Should find nearest neighbor");

        assert_relative_eq!(dist_ext, 2.0, epsilon = 1e-5);

        // Quality should extrapolate (in this case, uniform 42.0)
        assert_relative_eq!(q_ext.rho, 42.0, epsilon = 1e-5);
    }
    #[test]
    fn test_mesh_model_query_multi_cell_interpolation() {
        // 1. Setup a 5x5x5 grid (64 cells, 320 tetrahedra)
        let ni = 5;
        let nj = 5;
        let nk = 5;
        let vertices = generate_grid(ni, nj, nk);

        // 2. Assign qualities using a linear field: Quality.rho = x + y + z
        // This allows us to predict the exact interpolation result anywhere.
        let qualities: Vec<Quality> = vertices
            .iter()
            .map(|p| mock_quality(p.x + p.y + p.z))
            .collect();

        let chart = |i, j, k| i + j * ni + k * ni * nj;
        let mesh = MeshModel::curvilinear_mesh(&vertices, &qualities, (ni, nj, nk), chart);

        // --- TEST 1: Interior Query ---
        // Point is inside the cell at (i=2, j=1, k=3)
        let p_in = Point3::new(2.5, 1.2, 3.7);
        let (q_in, dist_in) = mesh.query(p_in).expect("Should find an interior simplex");

        // Distance should be 0.0 because it's inside
        assert_relative_eq!(dist_in, 0.0, epsilon = 1e-5);

        // Expected quality: 2.5 + 1.2 + 3.7 = 7.4
        // If the BVH picked a simplex from the wrong part of the grid,
        // this would fail significantly.
        assert_relative_eq!(q_in.rho, 7.4, epsilon = 1e-5);

        // --- TEST 2: Boundary Query ---
        // Testing exactly on a grid line (integer coordinates)
        let p_boundary = Point3::new(1.0, 4.0, 2.0);
        let (q_bound, dist_bound) = mesh.query(p_boundary).unwrap();

        assert_relative_eq!(dist_bound, 0.0, epsilon = 1e-5);
        assert_relative_eq!(q_bound.rho, 7.0, epsilon = 1e-5);

        // --- TEST 3: Exterior Query ---
        // Querying far outside the 0.0..4.0 bounds
        // Max vertex is (4,4,4). Let's query (6, 4, 4).
        // Expected distance: 2.0. Expected rho: 14.0.
        // Reason for this that we are extrapolating a linear function with linear interpolation
        // so there should not be any error.
        let p_out = Point3::new(6.0, 4.0, 4.0);
        let (q_out, dist_out) = mesh.query(p_out).expect("Should extrapolate from boundary");

        assert_relative_eq!(dist_out, 2.0, epsilon = 1e-5);
        assert_relative_eq!(q_out.rho, 14.0, epsilon = 1e-5);
    }
    #[test]

    fn test_mesh_face_continuity() {
        // Setup 2x2x2 grid (1x1x1 cells)
        let ni = 2;
        let nj = 2;
        let nk = 2;
        let vertices = generate_grid(ni, nj, nk);
        let qualities: Vec<Quality> = vertices
            .iter()
            .map(|p| mock_quality(p.x)) // Simple gradient along X
            .collect();

        let mesh = MeshModel::curvilinear_mesh(&vertices, &qualities, (ni, nj, nk), |i, j, k| {
            i + j * ni + k * ni * nj
        });

        // Query exactly on the face where x = 0.5 (shared by multiple tetrahedra)
        let p_face = Point3::new(0.5, 0.5, 0.5);
        let (q, dist) = mesh.query(p_face).unwrap();

        assert_relative_eq!(dist, 0.0, epsilon = 1e-5);
        assert_relative_eq!(q.rho, 0.5, epsilon = 1e-5);
    }
}

pub fn load_mesh_from_hdf5<'a>(
    file_path: &str,
    vertices_buf: &'a mut Vec<Point3<f32>>,
    qualities_buf: &'a mut Vec<Quality>,
) -> Result<MeshModel<'a>, hdf5_metno::Error> {
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
