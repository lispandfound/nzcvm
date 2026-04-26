use crate::quality::Quality;
use crate::real::Real;
use crate::simplex::Simplex;
use deepsize::{Context, DeepSizeOf};
use enum_dispatch::enum_dispatch;
use nalgebra::Scalar;
use nalgebra::{Point3, Point4};
use ndarray::ArrayView2;

/// Diagnostic output from a model query, carrying the per-vertex qualities
/// used in the computation.
pub enum ModelExplanation {
    Constant(ConstantModel<Quality>),
    Interpolate(InterpolateModel<Quality>),
}

/// A type that can report the seismic quality at a point inside a simplex.
#[enum_dispatch]
pub trait Queryable {
    /// Return the quality at `point` inside `simplex`, looking up vertex
    /// properties from the `(N, 6)` qualities array.
    fn quality_at(
        &self,
        qualities: ArrayView2<'_, Real>,
        simplex: &Simplex,
        point: &Point3<Real>,
    ) -> Quality;
    /// Return a diagnostic description of this model's contribution.
    fn explanation(&self, qualities: ArrayView2<'_, Real>) -> ModelExplanation;
}

/// Per-simplex model variant: either constant or barycentric interpolation.
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

/// Model that returns the same quality regardless of position within the simplex.
pub struct ConstantModel<T> {
    /// Index into the qualities array (or the quality itself when `T = Quality`).
    pub quality: T,
}

impl Queryable for ConstantModel<usize> {
    fn quality_at(
        &self,
        qualities: ArrayView2<'_, Real>,
        _simplex: &Simplex,
        _point: &Point3<Real>,
    ) -> Quality {
        Quality::from(qualities.row(self.quality))
    }

    fn explanation(&self, qualities: ArrayView2<'_, Real>) -> ModelExplanation {
        ModelExplanation::Constant(ConstantModel {
            quality: Quality::from(qualities.row(self.quality)),
        })
    }
}

/// Model that interpolates quality using barycentric coordinates within the simplex.
pub struct InterpolateModel<T: Scalar> {
    /// Indices of the four vertex qualities (or the qualities themselves when
    /// `T = Quality`), stored in `(x, y, z, w)` order matching the simplex
    /// vertices.
    pub qualities: Point4<T>,
}

impl Queryable for InterpolateModel<usize> {
    fn quality_at(
        &self,
        qualities: ArrayView2<'_, Real>,
        simplex: &Simplex,
        point: &Point3<Real>,
    ) -> Quality {
        let bary = simplex.barycentric_coordinates(*point);
        let q0 = Quality::from(qualities.row(self.qualities.w));
        let q1 = Quality::from(qualities.row(self.qualities.x));
        let q2 = Quality::from(qualities.row(self.qualities.y));
        let q3 = Quality::from(qualities.row(self.qualities.z));
        q0 * bary.w + q1 * bary.x + q2 * bary.y + q3 * bary.z
    }

    fn explanation(&self, qualities: ArrayView2<'_, Real>) -> ModelExplanation {
        ModelExplanation::Interpolate(InterpolateModel {
            qualities: self.qualities.map(|x| Quality::from(qualities.row(x))),
        })
    }
}
