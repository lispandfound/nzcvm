use std::ops::{Add, Mul};

#[derive(Clone, Debug, Copy)]
pub struct Quality {
    pub rho: f32,
    pub vp: f32,
    pub vs: f32,
    pub qp: f32,
    pub qs: f32,
}

// Enables: quality_a + quality_b
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

// Enables: quality * 0.5
impl Mul<f32> for Quality {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            rho: self.rho * rhs,
            vp: self.vp * rhs,
            vs: self.vs * rhs,
            qp: self.qp * rhs,
            qs: self.qs * rhs,
        }
    }
}

// Optional: Enables 0.5 * quality (LHS multiplication)
impl Mul<Quality> for f32 {
    type Output = Quality;

    fn mul(self, rhs: Quality) -> Self::Output {
        rhs * self
    }
}
