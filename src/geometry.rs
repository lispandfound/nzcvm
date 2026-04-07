use nalgebra::{Point2, Point3, RealField, Vector2};

/// Calculates the squared distance from a point to a line segment.
pub fn line_to_point_dist_sq<F: RealField + Copy>(p: Point2<F>, a: Point2<F>, b: Point2<F>) -> F {
    let closest = closest_point_to_line(p, a, b);

    (p - closest).norm_squared()
}

/// Returns the closest point on a line segment to a given point.
pub fn closest_point_to_line<F: RealField + Copy>(
    p: Point2<F>,
    a: Point2<F>,
    b: Point2<F>,
) -> Point2<F> {
    let ab = b - a;
    let ap = p - a;
    let line_len_sq = ab.norm_squared();

    if line_len_sq == F::zero() {
        return p;
    }

    // Projection scalar t clamped to [0, 1] to stay on the segment
    let t = (ap.dot(&ab) / line_len_sq).clamp(F::zero(), F::one());
    a + ab * t
}

/// Calculates the squared distance from a 3D point to a triangle.
pub fn point_triangle_distance_sq<F: RealField + Copy>(
    q: Point3<F>,
    p1: Point3<F>,
    p2: Point3<F>,
    p3: Point3<F>,
) -> F {
    let ab = p2 - p1;
    let ac = p3 - p1;
    let aq = q - p1;

    // Check if q is in vertex region outside p1
    let d1 = ab.dot(&aq);
    let d2 = ac.dot(&aq);
    if d1 <= F::zero() && d2 <= F::zero() {
        return (q - p1).norm_squared();
    }

    // Check if q is in vertex region outside p2
    let bq = q - p2;
    let d3 = ab.dot(&bq);
    let d4 = ac.dot(&bq);
    if d3 >= F::zero() && d4 <= d3 {
        return (q - p2).norm_squared();
    }

    // Check if q is in edge region of ab
    let vc = d1 * d4 - d3 * d2;
    if vc <= F::zero() && d1 >= F::zero() && d3 <= F::zero() {
        let v = d1 / (d1 - d3);
        return (q - (p1 + ab * v)).norm_squared();
    }

    // Check if q is in vertex region outside p3
    let cq = q - p3;
    let d5 = ab.dot(&cq);
    let d6 = ac.dot(&cq);
    if d6 >= F::zero() && d5 <= d6 {
        return (q - p3).norm_squared();
    }

    // Check if q is in edge region of ac
    let vb = d5 * d2 - d1 * d6;
    if vb <= F::zero() && d2 >= F::zero() && d6 <= F::zero() {
        let w = d2 / (d2 - d6);
        return (q - (p1 + ac * w)).norm_squared();
    }

    // Check if q is in edge region of bc
    let va = d3 * d6 - d5 * d4;
    if va <= F::zero() && (d4 - d3) >= F::zero() && (d5 - d6) >= F::zero() {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (q - (p2 + (p3 - p2) * w)).norm_squared();
    }

    // P is inside the face region. Compute distance via barycentric coordinates
    let denom = F::one() / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let projection = p1 + ab * v + ac * w;

    (q - projection).norm_squared()
}
