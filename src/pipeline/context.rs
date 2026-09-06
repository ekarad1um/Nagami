//! Read-only context handed to every [`super::Pass::run`] invocation.

use std::cell::RefCell;

use crate::config::Config;
use crate::name_map::NameLog;

/// Per-pass execution context: the active configuration and the rename log.
#[derive(Debug, Clone, Copy)]
pub struct PassContext<'a> {
    /// The active compaction configuration.
    pub config: &'a Config,
    /// Where the rename pass logs module-scope renames (`RefCell`: the
    /// context is shared read-only); `None` disables recording.
    pub name_log: Option<&'a RefCell<NameLog>>,
}
