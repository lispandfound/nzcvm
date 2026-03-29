use ndarray::{Array, Array4, ShapeError};

use crate::geomodelgrid::*;

pub fn uniform_values(
    shape: (usize, usize, usize),
    vp: f32,
    vs: f32,
    rho: f32,
    qp: f32,
    qs: f32,
) -> Result<Array4<f32>, &'static str> {
    let (nx, ny, nz) = shape;
    let nv = 7; // 5 media values plus fault block id and zone id
    let final_shape = (nx, ny, nz, nv);

    Array::from_vec(vec![rho, vp, vs, qp, qs, 0.0, 0.0])
        .broadcast(final_shape)
        .map(|view| view.to_owned())
        .ok_or("Could not broadcast shapes")
}
