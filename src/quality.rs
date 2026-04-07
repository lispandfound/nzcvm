use crate::real::Real;
use ndarray::Array1;
use std::ops::{Add, Mul};

#[derive(Clone, Debug, Copy)]
pub struct Quality {
    pub rho: Real,
    pub vp: Real,
    pub vs: Real,
    pub qp: Real,
    pub qs: Real,
}

impl Into<Array1<Real>> for Quality {
    fn into(self) -> Array1<Real> {
        Array1::from_iter([self.rho, self.vp, self.vs, self.qp, self.qs].into_iter())
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
        }
    }
}

impl Mul<Quality> for Real {
    type Output = Quality;

    fn mul(self, rhs: Quality) -> Self::Output {
        rhs * self
    }
}
