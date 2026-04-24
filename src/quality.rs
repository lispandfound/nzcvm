use crate::real::Real;
use deepsize::DeepSizeOf;
use ndarray::{Array1, ArrayView1};
use std::ops::{Add, Mul};

#[derive(Clone, Debug, Copy, PartialEq, DeepSizeOf)]
pub struct Quality {
    pub rho: Real,
    pub vp: Real,
    pub vs: Real,
    pub qp: Real,
    pub qs: Real,
    pub alpha: Real,
}

impl Quality {
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
            alpha: alpha,
        }
    }
}

impl Into<Array1<Real>> for Quality {
    fn into(self) -> Array1<Real> {
        Array1::from_iter([self.rho, self.vp, self.vs, self.qp, self.qs].into_iter())
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
