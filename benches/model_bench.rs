use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
use nalgebra::Point3;
use nzcvm::mesh::MeshModel;
use nzcvm::quality::Quality;
use pprof::criterion::{Output, PProfProfiler};
use std::hint::black_box;

fn mock_quality() -> Quality {
    Quality {
        rho: 2500.0,
        vp: 3000.0,
        vs: 1500.0,
        qp: 100.0,
        qs: 50.0,
        alpha: 1.0,
    }
}

fn bench_mesh_queries(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let mut group = c.benchmark_group("Mesh_Point_Query");
    group.plot_config(plot_config);

    for size in [160].iter() {
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

        // Multiple query points to isolate spatial bias
        let queries = [
            Point3::new(0.1, 0.1, 0.1),
            Point3::new(n as f32 / 2.0, n as f32 / 2.0, n as f32 / 2.0),
            Point3::new(n as f32 - 1.1, n as f32 - 1.1, n as f32 - 1.1),
        ];

        group.bench_with_input(
            BenchmarkId::from_parameter(total_vertices),
            &mesh,
            |b, m: &MeshModel| {
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

criterion_group! {
    name = benches;
    config = Criterion::default()
        .with_profiler(PProfProfiler::new(500, Output::Flamegraph(None)));
    targets =
        bench_mesh_queries,

}

criterion_main!(benches);
