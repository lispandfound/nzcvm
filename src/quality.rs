use crate::real::Real;
use deepsize::DeepSizeOf;
use ndarray::{Array1, ArrayView1};
use std::ops::{Add, Mul};

/// Seismic material properties at a single point.
///
/// `alpha` is the opacity weight used when blending overlapping models;
/// it follows the Porter-Duff "over" compositing rule in [`Quality::blend`].
#[derive(Clone, Debug, Copy, PartialEq, DeepSizeOf, serde::Serialize, serde::Deserialize)]
pub struct Quality {
    pub rho: Real,
    pub vp: Real,
    pub vs: Real,
    pub qp: Real,
    pub qs: Real,
    /// Opacity weight in `[0, 1]`.  A value of `1.0` means fully opaque:
    /// higher-priority models beneath are ignored.
    pub alpha: Real,
}

impl Quality {
    /// Composite `self` over `rhs` using the Porter-Duff "over" operator.
    ///
    /// The resulting `alpha` is `self.alpha + rhs.alpha * (1 - self.alpha)`.
    /// Material properties are blended proportionally.
    ///
    /// # Examples
    ///
    /// A fully-opaque quality blended with anything stays unchanged:
    ///
    /// ```
    /// use nzcvm::quality::Quality;
    /// let a = Quality { rho: 2700.0, vp: 6000.0, vs: 3500.0, qp: 200.0, qs: 100.0, alpha: 1.0 };
    /// let b = Quality { rho: 1000.0, vp: 1500.0, vs: 0.0, qp: 50.0, qs: 25.0, alpha: 0.5 };
    /// let blended = a.blend(&b);
    /// assert!((blended.rho - 2700.0).abs() < 1e-3);
    /// assert!((blended.alpha - 1.0).abs() < 1e-3);
    /// ```
    pub fn blend(&self, rhs: &Quality) -> Quality {
        let alpha = self.alpha + rhs.alpha * (1.0 - self.alpha);
        let a0 = self.alpha / alpha;
        let a1 = rhs.alpha * (1.0 - self.alpha) / alpha;
        Self {
            rho: a0 * self.rho + a1 * rhs.rho,
            vp: a0 * self.vp + a1 * rhs.vp,
            vs: a0 * self.vs + a1 * rhs.vs,
            qp: a0 * self.qp + a1 * rhs.qp,
            qs: a0 * self.qs + a1 * rhs.qs,
            alpha,
        }
    }
}

impl From<Quality> for Array1<Real> {
    fn from(val: Quality) -> Self {
        Array1::from_iter([val.rho, val.vp, val.vs, val.qp, val.qs])
    }
}

impl From<ArrayView1<'_, Real>> for Quality {
    fn from(arr: ArrayView1<'_, Real>) -> Self {
        Quality {
            rho: arr[0],
            vp: arr[1],
            vs: arr[2],
            qp: arr[3],
            qs: arr[4],
            alpha: arr[5],
        }
    }
}

// TODO: How should qp/qs arithmetic be handled?
impl Add for Quality {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            rho: self.rho + rhs.rho,
            vp: self.vp + rhs.vp,
            vs: self.vs + rhs.vs,
            qp: self.qp + rhs.qp,
            qs: self.qs + rhs.qs,
            alpha: self.alpha + rhs.alpha,
        }
    }
}

impl Mul<Real> for Quality {
    type Output = Self;

    fn mul(self, rhs: Real) -> Self::Output {
        Self {
            rho: self.rho * rhs,
            vp: self.vp * rhs,
            vs: self.vs * rhs,
            qp: self.qp * rhs,
            qs: self.qs * rhs,
            alpha: self.alpha * rhs,
        }
    }
}

impl Mul<Quality> for Real {
    type Output = Quality;

    fn mul(self, rhs: Quality) -> Self::Output {
        rhs * self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn q(v: Real) -> Quality {
        Quality { rho: v, vp: v, vs: v, qp: v, qs: v, alpha: v }
    }

    #[test]
    fn test_quality_add() {
        let a = q(1.0);
        let b = q(2.0);
        let c = a + b;
        assert_relative_eq!(c.rho, 3.0);
        assert_relative_eq!(c.vp, 3.0);
        assert_relative_eq!(c.alpha, 3.0);
    }

    #[test]
    fn test_quality_mul_scalar() {
        let a = q(4.0);
        let b = a * 0.5;
        assert_relative_eq!(b.rho, 2.0);
        assert_relative_eq!(b.vs, 2.0);
    }

    #[test]
    fn test_scalar_mul_quality() {
        let a = q(4.0);
        let b = 0.5 * a;
        assert_relative_eq!(b.rho, 2.0);
    }

    #[test]
    fn test_blend_identity_alpha_one() {
        let a = Quality { rho: 10.0, vp: 20.0, vs: 30.0, qp: 40.0, qs: 50.0, alpha: 1.0 };
        let b = Quality { rho: 99.0, vp: 99.0, vs: 99.0, qp: 99.0, qs: 99.0, alpha: 0.5 };
        let blended = a.blend(&b);
        assert_relative_eq!(blended.rho, a.rho, epsilon = 1e-5);
        assert_relative_eq!(blended.alpha, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_blend_commutative_alpha() {
        let a = Quality { rho: 1.0, vp: 1.0, vs: 1.0, qp: 1.0, qs: 1.0, alpha: 0.6 };
        let b = Quality { rho: 2.0, vp: 2.0, vs: 2.0, qp: 2.0, qs: 2.0, alpha: 0.4 };
        let ab = a.blend(&b);
        let ba = b.blend(&a);
        let expected_alpha_ab = 0.6 + 0.4 * (1.0 - 0.6);
        let expected_alpha_ba = 0.4 + 0.6 * (1.0 - 0.4);
        assert_relative_eq!(ab.alpha, expected_alpha_ab as Real, epsilon = 1e-5);
        assert_relative_eq!(ba.alpha, expected_alpha_ba as Real, epsilon = 1e-5);
    }

    #[test]
    fn test_blend_two_equal_half_alpha() {
        let q1 = Quality { rho: 10.0, vp: 20.0, vs: 30.0, qp: 40.0, qs: 50.0, alpha: 0.5 };
        let q2 = Quality { rho: 10.0, vp: 20.0, vs: 30.0, qp: 40.0, qs: 50.0, alpha: 0.5 };
        let blended = q1.blend(&q2);
        assert_relative_eq!(blended.rho, 10.0, epsilon = 1e-5);
        assert_relative_eq!(blended.vp, 20.0, epsilon = 1e-5);
        assert_relative_eq!(blended.alpha, 0.75, epsilon = 1e-5);
    }

    #[test]
    fn test_from_array1() {
        let arr = ndarray::array![1.0, 2.0, 3.0, 4.0, 5.0, 0.8];
        let q = Quality::from(arr.view());
        assert_relative_eq!(q.rho, 1.0);
        assert_relative_eq!(q.vp, 2.0);
        assert_relative_eq!(q.vs, 3.0);
        assert_relative_eq!(q.qp, 4.0);
        assert_relative_eq!(q.qs, 5.0);
        assert_relative_eq!(q.alpha, 0.8);
    }

    #[test]
    fn test_into_array1() {
        let quality = Quality { rho: 1.0, vp: 2.0, vs: 3.0, qp: 4.0, qs: 5.0, alpha: 0.0 };
        let arr: ndarray::Array1<Real> = quality.into();
        assert_eq!(arr.len(), 5);
        assert_relative_eq!(arr[0], 1.0);
        assert_relative_eq!(arr[4], 5.0);
    }
}
