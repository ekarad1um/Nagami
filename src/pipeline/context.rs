//! Read-only context handed to every [`super::Pass::run`] invocation.

use std::cell::RefCell;

use crate::config::Config;
use crate::name_map::NameLog;

/// Per-pass execution context: the active configuration, the module's
/// validation info and the rename log.
#[derive(Debug, Clone, Copy)]
pub struct PassContext<'a> {
    /// The active compaction configuration.
    pub config: &'a Config,
    /// naga's analysis of the module AS HANDED TO THE PASS; the driver
    /// re-validates after every accepted change, so it is never stale on
    /// entry, and a pass that mutates the module must not read it after.
    pub info: &'a naga::valid::ModuleInfo,
    /// Where the rename pass logs module-scope renames (`RefCell`: the
    /// context is shared read-only); `None` disables recording.
    pub name_log: Option<&'a RefCell<NameLog>>,
    /// What the tail's render leaves out and leaves behind; `None` keeps
    /// the census alias plan.
    pub tail: Option<&'a TailRender>,
}

/// The tail's rename renders the module once to rank names by the text;
/// that render leaves out the preamble's declarations as the shipped one
/// does, and leaves its type spellings behind for the alias plan of the
/// emission that ships (`GenerateOptions::type_uses`).
#[derive(Debug, Default)]
pub struct TailRender {
    /// The module-scope names the preamble declares.
    pub preamble_names: std::collections::HashSet<String>,
    /// The type spellings of the rendered module, once rendered.
    pub type_uses: RefCell<Option<crate::generator::TypeUses>>,
}

#[cfg(test)]
impl PassContext<'_> {
    /// One run of `pass` as the driver stages it: `module` validated, its
    /// info in the context, no name log.
    pub(crate) fn run_pass(
        pass: &mut dyn super::Pass,
        module: &mut naga::Module,
        config: &Config,
    ) -> Result<bool, crate::error::Error> {
        let info = crate::io::validate_module(module)?;
        pass.run(
            module,
            &PassContext {
                config,
                info: &info,
                name_log: None,
                tail: None,
            },
        )
    }
}
