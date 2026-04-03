use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
use geo::{Coord, LineString, Polygon};
use nalgebra::Point3;
use ndarray::{Array1, Array2};
use ordered_float::OrderedFloat;
use std::collections::BTreeMap;
use std::hint::black_box;

use nzcvm::layers::{LayerGeometry, LayerTree, Model};
use nzcvm::mesh::MeshModel;
use nzcvm::model::ModelTree;
use nzcvm::quality::Quality;

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
    let (z_t, z_b) = (
        Array2::from_elem((n, n), 0.0),
        Array2::from_elem((n, n), 50.0),
    );

    for v_count in [4, 16, 64, 256, 1024, 4096].iter() {
        let poly = generate_ngon(*v_count, 50.0, 50.0, 40.0);
        let geom = LayerGeometry::new(&poly, x.clone(), y.clone(), z_t.clone(), z_b.clone());
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
        let (z_t, z_b) = (
            Array2::from_elem((*n, *n), 0.0),
            Array2::from_elem((*n, *n), 50.0),
        );

        let geom = LayerGeometry::new(&poly, x, y, z_t, z_b);
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

criterion_group!(
    benches,
    bench_mesh_queries,
    bench_layer_poly_complexity,
    bench_layer_surface_complexity
);
criterion_main!(benches);
