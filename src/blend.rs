use crate::model_tree::ModelTree;
use crate::quality::Quality;
use crate::query::Query;
use crate::real::Real;
use enum_dispatch::enum_dispatch;
use nalgebra::Point3;
use ndarray::ArrayView1;

#[enum_dispatch]
pub trait Blend {
    fn apply(&self, tree: &ModelTree, pt: Point3<Real>, row: ArrayView1<'_, Real>, lo: u8, hi: u8) -> Option<Quality>;
}

/// Overwrite: replaces the buffer row with the query result.
#[derive(Debug, Clone, Copy)]
pub struct Erase;

/// Porter-Duff "over": composites the query result over the existing row.
#[derive(Debug, Clone, Copy)]
pub struct Over;

impl Blend for Erase {
    #[inline]
    fn apply(&self, tree: &ModelTree, pt: Point3<Real>, _row: ArrayView1<'_, Real>, lo: u8, hi: u8) -> Option<Quality> {
        tree.query(pt, None, lo, hi)
    }
}

impl Blend for Over {
    #[inline]
    fn apply(&self, tree: &ModelTree, pt: Point3<Real>, row: ArrayView1<'_, Real>, lo: u8, hi: u8) -> Option<Quality> {
        tree.query(pt, Some(Quality::from(row)), lo, hi)
    }
}

/// Dispatch enum — resolved once per `query_many` call so the hot loop is branch-free.
#[enum_dispatch(Blend)]
#[derive(Debug, Clone, Copy)]
pub enum BlendDispatch {
    Erase,
    Over,
}
