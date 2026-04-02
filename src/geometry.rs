use nalgebra::Point3;
use ndarray::ArrayView1;
#[inline(always)]
fn line_to_point_dist_sq(px: f32, py: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let line_len_sq = dx * dx + dy * dy;

    if line_len_sq == 0.0 {
        return (px - x1).powi(2) + (py - y1).powi(2);
    }

    // Projection scalar t = [(P-A) . (B-A)] / |B-A|^2
    let t = ((px - x1) * dx + (py - y1) * dy) / line_len_sq;
    let t = t.clamp(0.0, 1.0);

    let closest_x = x1 + t * dx;
    let closest_y = y1 + t * dy;

    (px - closest_x).powi(2) + (py - closest_y).powi(2)
}

pub fn polygon_distance_sq(point: Point3<f32>, coords: &[f32]) -> f32 {
    let (px, py) = (point.x, point.y);
    let mut min_d2 = f32::MAX;
    let mut is_inside = false;

    let n = coords.len();

    for i in (0..n).step_by(2) {
        // Handle wrapping for the closing edge
        let (x1, y1) = (coords[i], coords[i + 1]);
        let (x2, y2) = if i + 2 < n {
            (coords[i + 2], coords[i + 3])
        } else {
            (coords[0], coords[1])
        };

        // 1. DISTANCE CALCULATION
        let d2 = line_to_point_dist_sq(px, py, x1, y1, x2, y2);
        if d2 < min_d2 {
            min_d2 = d2;
        }

        if ((y1 > py) != (y2 > py)) && (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1) {
            is_inside = !is_inside;
        }

        if min_d2 < 1e-6 {
            return 0.0;
        }
    }

    if is_inside {
        0.0 // Point is inside the polygon
    } else {
        min_d2 // Point is outside, return squared distance
    }
}

pub fn point_triangle_distance_sq(
    q: Point3<f32>,
    p1: Point3<f32>,
    p2: Point3<f32>,
    p3: Point3<f32>,
) -> f32 {
    let ab = p2 - p1;
    let ac = p3 - p1;
    let aq = q - p1;

    // Check if q is in vertex region outside p1
    let d1 = ab.dot(&aq);
    let d2 = ac.dot(&aq);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (q - p1).norm_squared();
    }

    // Check if q is in vertex region outside p2
    let bq = q - p2;
    let d3 = ab.dot(&bq);
    let d4 = ac.dot(&bq);
    if d3 >= 0.0 && d4 <= d3 {
        return (q - p2).norm_squared();
    }

    // Check if q is in edge region of ab, if so return distance to edge ab
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (q - (p1 + ab * v)).norm_squared();
    }

    // Check if q is in vertex region outside p3
    let cq = q - p3;
    let d5 = ab.dot(&cq);
    let d6 = ac.dot(&cq);
    if d6 >= 0.0 && d5 <= d6 {
        return (q - p3).norm_squared();
    }

    // Check if q is in edge region of ac, if so return distance to edge ac
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (q - (p1 + ac * w)).norm_squared();
    }

    // Check if q is in edge region of bc, if so return distance to edge bc
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (q - (p2 + (p3 - p2) * w)).norm_squared();
    }

    // P is inside the face region. Compute distance via barycentric coordinates
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let projection = p1 + ab * v + ac * w;
    (q - projection).norm_squared()
}
