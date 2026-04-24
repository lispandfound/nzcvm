use crate::quality::Quality;
use crate::real::Real;
use crate::simplex::Simplex;
use deepsize::{Context, DeepSizeOf};
use enum_dispatch::enum_dispatch;
use nalgebra::Scalar;
use nalgebra::{Point3, Point4};

pub enum ModelExplanation {
    Constant(ConstantModel<Quality>),
    Interpolate(InterpolateModel<Quality>),
}

#[enum_dispatch]
pub trait Queryable {
    fn quality_at(&self, qualities: &[Quality], simplex: &Simplex, point: &Point3<Real>)
        -> Quality;
    fn explanation(&self, qualities: &[Quality]) -> ModelExplanation;
}

#[enum_dispatch(Queryable)]
pub enum Model {
    Constant(ConstantModel<usize>),
    Interpolate(InterpolateModel<usize>),
}

impl DeepSizeOf for Model {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

pub struct ConstantModel<T> {
    pub quality: T,
}

impl Queryable for ConstantModel<usize> {
    fn quality_at(
        &self,
        qualities: &[Quality],
        _simplex: &Simplex,
        _point: &Point3<Real>,
    ) -> Quality {
        qualities[self.quality]
    }

    fn explanation(&self, qualities: &[Quality]) -> ModelExplanation {
        ModelExplanation::Constant(ConstantModel {
            quality: qualities[self.quality],
        })
    }
}

pub struct InterpolateModel<T: Scalar> {
    pub qualities: Point4<T>,
}

impl Queryable for InterpolateModel<usize> {
    fn quality_at(
        &self,
        qualities: &[Quality],
        simplex: &Simplex,
        point: &Point3<Real>,
    ) -> Quality {
        let bary = simplex.barycentric_coordinates(*point);
        let q0 = qualities[self.qualities.w];
        let q1 = qualities[self.qualities.x];
        let q2 = qualities[self.qualities.y];
        let q3 = qualities[self.qualities.z];
        q0 * bary.w + q1 * bary.x + q2 * bary.y + q3 * bary.z
    }

    fn explanation(&self, qualities: &[Quality]) -> ModelExplanation {
        ModelExplanation::Interpolate(InterpolateModel {
            qualities: self.qualities.map(|x| qualities[x]),
        })
    }
}
