use crate::real::Real;
use nalgebra::{Affine3, Matrix4, Point3, Rotation3, Translation3, Unit, Vector3};

pub trait CoordinateSystem {
    fn coordinates(
        &self,
        index: (usize, usize, usize),
        dx: Real,
        dy: Real,
        dz: Real,
    ) -> Point3<Real>;
}

pub struct LinearSystem {
    transform: Affine3<Real>,
}

impl CoordinateSystem for LinearSystem {
    fn coordinates(
        &self,
        index: (usize, usize, usize),
        dx: Real,
        dy: Real,
        dz: Real,
    ) -> Point3<Real> {
        let (i, j, k) = index;
        let point = Point3::new(i as Real * dx, j as Real * dy, k as Real * dz);

        self.transform.transform_point(&point)
    }
}

pub fn sw4_transform(azimuth_deg: Real, x_origin: Real, y_origin: Real) -> LinearSystem {
    let swap = Affine3::from_matrix_unchecked(Matrix4::new(
        0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ));

    let axis = Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0));
    let rotation = Rotation3::from_axis_angle(&axis, -azimuth_deg.to_radians());
    let rot_affine: Affine3<Real> = nalgebra::convert(rotation);

    let translation = Translation3::new(x_origin, y_origin, 0.0);
    let trans_affine: Affine3<Real> = nalgebra::convert(translation);

    let combined_transform = trans_affine * rot_affine * swap;

    LinearSystem {
        transform: combined_transform,
    }
}
