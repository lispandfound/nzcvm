use crate::quality::Quality;
use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub trait Blend {
    /// Combine an existing (buffer) quality with a freshly-queried one.
    ///
    /// `existing` is the current value in the output buffer row (may be
    /// `None` if the buffer is zero-filled and the caller passes nothing).
    fn apply(&self, existing: Option<Quality>, new: Quality) -> Quality;
}

/// Overwrite: replaces the buffer row with the query result.
#[derive(Debug, Clone, Copy)]
pub struct Erase;

/// Porter-Duff "over": composites the query result over the existing row.
#[derive(Debug, Clone, Copy)]
pub struct Over;

impl Blend for Erase {
    #[inline]
    fn apply(&self, _existing: Option<Quality>, new: Quality) -> Quality {
        new
    }
}

impl Blend for Over {
    #[inline]
    fn apply(&self, existing: Option<Quality>, new: Quality) -> Quality {
        match existing {
            None => new,
            Some(e) => e.blend(&new),
        }
    }
}

/// Dispatch enum — resolved once per `query_many` call so the hot loop is branch-free.
#[enum_dispatch(Blend)]
#[derive(Debug, Clone, Copy)]
pub enum BlendDispatch {
    Erase,
    Over,
}
