use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
use geo::{Coord, LineString, Polygon};
use nalgebra::Point3;
use ndarray::{meshgrid, Array1, Array2, MeshIndex};
use nzcvm::layers::{LayerGeometry, LayerTree, Model};
use nzcvm::mesh::MeshModel;
use nzcvm::model::ModelTree;
use nzcvm::quality::Quality;
use ordered_float::OrderedFloat;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::sync::Arc;

fn mock_quality() -> Quality {
    Quality {
        rho: 2500.0,
        vp: 3000.0,
        vs: 1500.0,
        qp: 100.0,
        qs: 50.0,
    }
}

fn generate_ngon(num_vertices: usize, center_x: f32, center_y: f32, radius: f32) -> Polygon<f32> {
    let mut coords = Vec::with_capacity(num_vertices + 1);
    for i in 0..num_vertices {
        let angle = 2.0 * std::f32::consts::PI * (i as f32) / (num_vertices as f32);
        coords.push(Coord {
            x: center_x + radius * angle.cos(),
            y: center_y + radius * angle.sin(),
        });
    }
    coords.push(coords[0]);
    Polygon::new(LineString::new(coords), vec![])
}

fn bench_mesh_queries(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("Mesh_Point_Query");
    group.plot_config(plot_config);

    for size in [10, 20, 40, 80, 160].iter() {
        let n = *size;
        let total_vertices = n * n * n;
        let vertices = (0..total_vertices)
            .map(|idx| {
                let i = idx / (n * n);
                let j = (idx / n) % n;
                let k = idx % n;
                Point3::new(i as f32, j as f32, k as f32)
            })
            .collect();

        let mesh = MeshModel::curvilinear_mesh(
            vertices,
            vec![mock_quality(); total_vertices],
            (n, n, n),
            |i, j, k| k + j * n + i * n * n,
        );
        let model = ModelTree::mesh_model(mesh);

        // Multiple query points to isolate spatial bias
        let queries = [
            Point3::new(0.1, 0.1, 0.1),
            Point3::new(n as f32 / 2.0, n as f32 / 2.0, n as f32 / 2.0),
            Point3::new(n as f32 - 1.1, n as f32 - 1.1, n as f32 - 1.1),
        ];

        group.bench_with_input(
            BenchmarkId::from_parameter(total_vertices),
            &model,
            |b, m| {
                b.iter(|| {
                    for q in queries.iter() {
                        m.query(black_box(*q));
                    }
                })
            },
        );
    }
    group.finish();
}

fn bench_layer_poly_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("Layer_Query_Poly_Vertices");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let n = 20;
    let (x, y) = (
        Array1::linspace(0.0, 100.0, n),
        Array1::linspace(0.0, 100.0, n),
    );
    let (x_mesh, y_mesh) = meshgrid((&x, &y), MeshIndex::IJ);
    let (z_t, z_b) = (
        Array2::from_elem((n, n), 0.0),
        Array2::from_elem((n, n), 50.0),
    );

    for v_count in [4, 16, 64, 256, 1024, 4096].iter() {
        let poly = generate_ngon(*v_count, 50.0, 50.0, 40.0);
        let geom = LayerGeometry::build(&poly, x_mesh, y_mesh, z_t.view(), z_b.view());
        let mut layers = BTreeMap::new();
        layers.insert(OrderedFloat(0.0), mock_quality());
        let tree =
            ModelTree::layered_model(LayerTree::new(vec![geom], vec![Model::Layered { layers }]));

        let queries = [Point3::new(50.0, 50.0, 25.0), Point3::new(10.0, 10.0, 25.0)];

        group.bench_with_input(BenchmarkId::from_parameter(v_count), &tree, |b, t| {
            b.iter(|| {
                for q in queries.iter() {
                    t.query(black_box(*q));
                }
            })
        });
    }
    group.finish();
}

fn bench_layer_surface_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("Layer_Query_Surface_Size");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let poly = generate_ngon(4, 50.0, 50.0, 80.0);

    for n in [10, 32, 100, 316, 1000].iter() {
        let x = Array1::linspace(0.0, 100.0, *n);
        let y = Array1::linspace(0.0, 100.0, *n);
        let (x_mesh, y_mesh) = meshgrid((&x, &y), MeshIndex::IJ);
        let (z_t, z_b) = (
            Array2::from_elem((*n, *n), 0.0),
            Array2::from_elem((*n, *n), 50.0),
        );

        let geom = LayerGeometry::build(&poly, x_mesh, y_mesh, z_t.view(), z_b.view());
        let mut layers = BTreeMap::new();
        layers.insert(OrderedFloat(0.0), mock_quality());
        let tree =
            ModelTree::layered_model(LayerTree::new(vec![geom], vec![Model::Layered { layers }]));

        let queries = [Point3::new(50.0, 50.0, 25.0), Point3::new(1.0, 1.0, 25.0)];

        group.bench_with_input(BenchmarkId::from_parameter(n * n), &tree, |b, t| {
            b.iter(|| {
                for q in queries.iter() {
                    t.query(black_box(*q));
                }
            })
        });
    }
    group.finish();
}

fn bench_complex_stacked_model(c: &mut Criterion) {
    let mut group = c.benchmark_group("Complex_Stacked_Query");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    // 1. Setup a "Large" Mesh (e.g., 64x64x64 = 262,144 vertices)
    let m_size = 64;
    let total_m_vertices = m_size * m_size * m_size;
    let mesh_vertices = (0..total_m_vertices)
        .map(|idx| {
            let i = idx / (m_size * m_size);
            let j = (idx / m_size) % m_size;
            let k = idx % m_size;
            Point3::new(i as f32, j as f32, k as f32)
        })
        .collect();

    let mesh = MeshModel::curvilinear_mesh(
        mesh_vertices,
        vec![mock_quality(); total_m_vertices],
        (m_size, m_size, m_size),
        |i, j, k| k + j * m_size + i * m_size * m_size,
    );
    let mut current_tree = Arc::new(ModelTree::mesh_model(mesh));

    // 2. Setup "Several" Polygonal Layers (5 layers)
    // Each layer has high polygon counts (4096 vertices)
    let layer_count = 5;
    let poly_vertices = 4096;
    let surface_res = 50; // Resolution of the height arrays

    for i in 1..=layer_count {
        let x = Array1::linspace(0.0, 100.0, surface_res);
        let y = Array1::linspace(0.0, 100.0, surface_res);
        let (x_mesh, y_mesh) = meshgrid((&x, &y), MeshIndex::IJ);

        // Stack layers vertically: Layer 1 is 0-10, Layer 2 is 10-20, etc.
        let z_top = (i as f32 - 1.0) * 10.0;
        let z_bottom = (i as f32) * 10.0;

        let (z_t, z_b) = (
            Array2::from_elem((surface_res, surface_res), z_top),
            Array2::from_elem((surface_res, surface_res), z_bottom),
        );

        let poly = generate_ngon(poly_vertices, 50.0, 50.0, 40.0);
        let geom = LayerGeometry::build(&poly, x_mesh, y_mesh, z_t.view(), z_b.view());

        let mut layers_map = BTreeMap::new();
        layers_map.insert(OrderedFloat(0.0), mock_quality());

        let layer_tree = LayerTree::new(vec![geom], vec![Model::Layered { layers: layers_map }]);
        let layer_model = ModelTree::layered_model(layer_tree);

        // Stack the new layer on top of the previous tree
        // Note: query() prioritizes the "left" side of the stack.
        current_tree = Arc::new(ModelTree::Stack(Arc::new(layer_model), current_tree));
    }

    // 3. Define Query Points
    let queries = [
        Point3::new(50.0, 50.0, 5.0),  // Deep inside Layer 1 (Top)
        Point3::new(50.0, 50.0, 45.0), // Inside Layer 5 (Bottom of layers)
        Point3::new(32.0, 32.0, 32.0), // Inside the Mesh
        Point3::new(99.0, 99.0, 99.0), // Outside layers, likely in mesh or empty
    ];

    group.bench_with_input(
        BenchmarkId::new(
            "Stacked_Layers_And_Mesh",
            format!("L:{}_P:{}", layer_count, poly_vertices),
        ),
        &current_tree,
        |b, tree| {
            b.iter(|| {
                for q in queries.iter() {
                    black_box(tree.query(black_box(*q)));
                }
            })
        },
    );

    group.finish();
}

fn bench_realistic_blended_model(c: &mut Criterion) {
    let mut group = c.benchmark_group("Realistic_Blended_Query");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    // 1. Right side: Dense Background Tomography Mesh
    let m_size = 64;
    let total_m_vertices = m_size * m_size * m_size;
    let mesh_vertices = (0..total_m_vertices)
        .map(|idx| {
            let i = idx / (m_size * m_size);
            let j = (idx / m_size) % m_size;
            let k = idx % m_size;
            Point3::new(i as f32 * 1000.0, j as f32 * 1000.0, k as f32 * 1000.0)
        })
        .collect();

    let mesh = MeshModel::curvilinear_mesh(
        mesh_vertices,
        vec![mock_quality(); total_m_vertices],
        (m_size, m_size, m_size),
        |i, j, k| k + j * m_size + i * m_size * m_size,
    );
    let background_tomo = Arc::new(ModelTree::mesh_model(mesh));

    // 2. Left side: Disjoint Basin Models
    // We'll create 3 separate high-poly basins far apart
    let poly_vertices = 4096;
    let surface_res = 50;
    let mut basin_geometries = Vec::new();

    // Basin centers
    let centers = [
        (10000.0, 10000.0),
        (30000.0, 10000.0),
        (50000.0, 10000.0),
        (70000.0, 10000.0),
        (90000.0, 10000.0),
        (10000.0, 30000.0),
        (30000.0, 30000.0),
        (50000.0, 30000.0),
        (70000.0, 30000.0),
        (90000.0, 30000.0),
        (10000.0, 50000.0),
        (30000.0, 50000.0),
        (50000.0, 50000.0),
        (70000.0, 50000.0),
        (90000.0, 50000.0),
        (10000.0, 70000.0),
        (30000.0, 70000.0),
        (50000.0, 70000.0),
        (70000.0, 70000.0),
        (90000.0, 70000.0),
        (10000.0, 90000.0),
        (30000.0, 90000.0),
        (50000.0, 90000.0),
        (70000.0, 90000.0),
        (90000.0, 90000.0),
        (10000.0, 110000.0),
        (30000.0, 110000.0),
        (50000.0, 110000.0),
        (70000.0, 110000.0),
        (90000.0, 110000.0),
    ];

    for (cx, cy) in centers {
        let x = Array1::linspace(cx - 5000.0, cx + 5000.0, surface_res);
        let y = Array1::linspace(cy - 5000.0, cy + 5000.0, surface_res);
        let (x_mesh, y_mesh) = meshgrid((&x, &y), MeshIndex::IJ);
        let (z_t, z_b) = (
            Array2::from_elem((surface_res, surface_res), 0.0),
            Array2::from_elem((surface_res, surface_res), 5000.0),
        );

        let poly = generate_ngon(poly_vertices, cx, cy, 4000.0);
        basin_geometries.push(LayerGeometry::build(
            &poly,
            x_mesh,
            y_mesh,
            z_t.view(),
            z_b.view(),
        ));
    }

    let mut layers_map = BTreeMap::new();
    layers_map.insert(OrderedFloat(0.0), mock_quality());
    let layers = (0..basin_geometries.len())
        .map(|_| Model::Layered {
            layers: layers_map.clone(),
        })
        .collect();

    // Combine all basins into one LayerTree
    let basins_tree = LayerTree::new(basin_geometries, layers);
    let basins_model = Arc::new(ModelTree::layered_model(basins_tree));

    // 3. The Blend Model
    let blend_dist = 10000.0;
    let model = ModelTree::Blend {
        left: basins_model,
        right: background_tomo,
        distance: blend_dist,
    };

    // 4. Realistic Query Points
    let queries = [
        // Point deep inside the first basin (Distance < 0 or very small)
        Point3::new(10000.0, 10000.0, 1000.0),
        // Point in the Blend Zone (e.g., 5km away from the 4km radius basin)
        // Distance will be ~1000m, which is < 10000m blend distance
        Point3::new(15000.0, 10000.0, 1000.0),
        // Point far outside in pure Tomography territory
        Point3::new(50000.0, 50000.0, 1000.0),
    ];

    group.bench_with_input(
        BenchmarkId::new("Blended_Basin_Tomo", format!("P:{}", poly_vertices)),
        &model,
        |b, m| {
            b.iter(|| {
                for q in queries.iter() {
                    black_box(m.query(black_box(*q)));
                }
            })
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_realistic_blended_model,
    bench_complex_stacked_model,
    bench_mesh_queries,
    bench_layer_poly_complexity,
    bench_layer_surface_complexity,
);
criterion_main!(benches);
