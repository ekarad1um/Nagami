//! Read-only context handed to every [`super::Pass::run`] invocation.

use std::cell::RefCell;
use std::path::Path;

use crate::config::Config;
use crate::name_map::NameLog;

/// Per-pass execution context.  Bundles the active configuration and
/// the resolved trace directory so passes do not individually reach
/// into [`Config`] for paths that the pipeline driver already owns.
#[derive(Debug, Clone, Copy)]
pub struct PassContext<'a> {
    /// The active compaction configuration.
    pub config: &'a Config,
    /// Directory the pipeline allocated for this run's trace output,
    /// or `None` when tracing is disabled.  See
    /// `pipeline::allocate_trace_run_dir` for the allocation scheme.
    pub trace_run_dir: Option<&'a Path>,
    /// Where the rename pass logs module-scope renames (`RefCell`: the
    /// context is shared read-only); `None` disables recording.
    pub name_log: Option<&'a RefCell<NameLog>>,
}
