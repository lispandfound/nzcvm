/// Floating-point scalar type used throughout the velocity model.
///
/// Defaults to `f32` for reduced memory footprint. Compile with the
/// `high_precision` feature to switch to `f64`.
///
/// # Examples
///
/// ```
/// use nzcvm::real::Real;
/// let v: Real = 3500.0;
/// assert_eq!(v, 3500.0_f32);
/// ```
#[cfg(feature = "high_precision")]
pub type Real = f64;

/// Floating-point scalar type used throughout the velocity model.
///
/// Defaults to `f32` for reduced memory footprint. Compile with the
/// `high_precision` feature to switch to `f64`.
///
/// # Examples
///
/// ```
/// use nzcvm::real::Real;
/// let v: Real = 3500.0;
/// assert_eq!(v, 3500.0_f32);
/// ```
#[cfg(not(feature = "high_precision"))]
pub type Real = f32;
