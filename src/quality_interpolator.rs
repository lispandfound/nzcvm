use crate::quality::Quality;
use nalgebra::Point3;
use rstar::primitives::GeomWithData;
use rstar::RTree;

type QualityDatabase = RTree<GeomWithData<[f32; 3], Quality>>;

pub fn query(
    database: &QualityDatabase,
    point: &Point3<f32>,
    n_neighbours: Option<usize>,
    eps: f32,
) -> Option<Quality> {
    let mut neighbours = database
        .nearest_neighbor_iter_with_distance_2(&[point.x, point.y, point.z])
        .take(n_neighbours.unwrap_or(database.size()));

    let (initial_entry, initial_dist_sq) = neighbours.next()?;

    if initial_dist_sq < eps * eps {
        return Some(initial_entry.data);
    }

    let initial_weight = 1.0 / initial_dist_sq;
    let initial_q = initial_entry.data;

    let (sum_q, total_weight) = neighbours.fold(
        (
            Quality {
                rho: initial_q.rho * initial_weight,
                vp: initial_q.vp * initial_weight,
                vs: initial_q.vs * initial_weight,
                qp: initial_q.qp * initial_weight,
                qs: initial_q.qs * initial_weight,
            },
            initial_weight,
        ),
        |(acc_q, acc_w), (entry, dist_sq)| {
            let w = 1.0 / dist_sq;
            let q = entry.data;
            (
                Quality {
                    rho: acc_q.rho + q.rho * w,
                    vp: acc_q.vp + q.vp * w,
                    vs: acc_q.vs + q.vs * w,
                    qp: acc_q.qp + q.qp * w,
                    qs: acc_q.qs + q.qs * w,
                },
                acc_w + w,
            )
        },
    );

    Some(Quality {
        rho: sum_q.rho / total_weight,
        vp: sum_q.vp / total_weight,
        vs: sum_q.vs / total_weight,
        qp: sum_q.qp / total_weight,
        qs: sum_q.qs / total_weight,
    })
}
