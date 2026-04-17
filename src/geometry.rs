use crate::real::Real;
use nalgebra::Point3;

/// Calculates the squared distance from a 3D point to a triangle.
pub fn point_triangle_distance_sq(
    q: Point3<Real>,
    p1: Point3<Real>,
    p2: Point3<Real>,
    p3: Point3<Real>,
) -> Real {
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

    // Check if q is in edge region of ab
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

    // Check if q is in edge region of ac
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (q - (p1 + ac * w)).norm_squared();
    }

    // Check if q is in edge region of bc
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
