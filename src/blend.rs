//! Blend-mode dispatch for vectorised model queries.
//!
//! This module defines the [`Blend`] trait and two concrete implementations —
//! [`Erase`] and [`Over`] — that are dispatched through [`BlendDispatch`]
//! using the [`enum_dispatch`] crate.  This keeps the inner query loop
//! branch-free: the blend mode is resolved once before entering the loop, and
//! then `blend.apply(...)` is inlined by the compiler.
//!
//! The Python-facing [`BlendMode`] C-like enum (defined in `lib.rs`) is
//! converted to a [`BlendDispatch`] value once per `query_many` call.

use crate::model_tree::ModelTree;
use crate::quality::Quality;
use crate::query::Query;
use crate::real::Real;
use enum_dispatch::enum_dispatch;
use nalgebra::Point3;
use ndarray::ArrayView1;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Apply a compositing operation to a single buffer row.
///
/// Implementations are called once per spatial point inside the
/// `query_many` loop.  The method receives:
///
/// * `tree`  — the model tree to query.
/// * `pt`    — the 3-D query point.
/// * `row`   — the current `[rho, vp, vs, qp, qs, alpha]` row from the
///   output buffer.  Used by [`Over`] as the "existing" quality.
/// * `lo`, `hi` — inclusive priority range filter.
///
/// Returns `Some(quality)` when the query hit a model in range, or
/// `None` if no model covers the point within the requested priority range.
#[enum_dispatch]
pub trait Blend {
    fn apply(
        &self,
        tree: &ModelTree,
        pt: Point3<Real>,
        row: ArrayView1<'_, Real>,
        lo: u8,
        hi: u8,
    ) -> Option<Quality>;
}

// ---------------------------------------------------------------------------
// Implementations
// ---------------------------------------------------------------------------

/// Overwrite blend: replaces the buffer row with the query result.
///
/// Points not covered by any matching model leave the row unchanged.
#[derive(Debug, Clone, Copy)]
pub struct Erase;

/// Over blend: Porter-Duff "over" composite of the query result *over*
/// the existing buffer row.
#[derive(Debug, Clone, Copy)]
pub struct Over;

impl Blend for Erase {
    #[inline]
    fn apply(
        &self,
        tree: &ModelTree,
        pt: Point3<Real>,
        _row: ArrayView1<'_, Real>,
        lo: u8,
        hi: u8,
    ) -> Option<Quality> {
        tree.query_bounded(pt, lo, hi)
    }
}

impl Blend for Over {
    #[inline]
    fn apply(
        &self,
        tree: &ModelTree,
        pt: Point3<Real>,
        row: ArrayView1<'_, Real>,
        lo: u8,
        hi: u8,
    ) -> Option<Quality> {
        let existing = Some(Quality::from(row));
        tree.query_bounded_into(pt, existing, lo, hi)
    }
}

// ---------------------------------------------------------------------------
// Dispatch enum
// ---------------------------------------------------------------------------

/// Internal dispatch enum created from the Python-facing `BlendMode`.
///
/// Uses `enum_dispatch` so the compiler can inline the correct blend
/// implementation without a `match` inside the hot query loop.
#[enum_dispatch(Blend)]
#[derive(Debug, Clone, Copy)]
pub enum BlendDispatch {
    Erase,
    Over,
}
