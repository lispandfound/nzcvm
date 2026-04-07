#[cfg(feature = "high_precision")]
pub type Real = f64;

#[cfg(not(feature = "high_precision"))]
pub type Real = f32;
