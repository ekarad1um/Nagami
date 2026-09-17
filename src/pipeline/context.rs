//! Read-only context handed to every [`super::Pass::run`] invocation.

use std::cell::{Cell, Ref, RefCell};
use std::rc::Rc;

use crate::config::Config;
use crate::error::Error;
use crate::name_map::NameLog;
use crate::passes::expr_util::{IndexBound, access_static_lengths};

/// Per-pass execution context: the active configuration, the module's
/// validation info and the rename log.
#[derive(Debug, Clone, Copy)]
pub struct PassContext<'a> {
    /// The active compaction configuration.
    pub config: &'a Config,
    /// naga's analysis of the module, read through [`Self::info`].
    pub info: &'a ModuleInfoCell,
    /// Where the rename pass logs module-scope renames (`RefCell`: the
    /// context is shared read-only); `None` disables recording.
    pub name_log: Option<&'a RefCell<NameLog>>,
    /// What the tail's render leaves out and leaves behind; `None` keeps
    /// the census alias plan.
    pub tail: Option<&'a TailRender>,
    /// Analyses of the module as handed to the pass, shared with the
    /// passes before it that left the module unchanged.
    pub analyses: &'a AnalysisCache,
}

impl PassContext<'_> {
    /// naga's analysis of `module`, validated now if no pass has read it
    /// since the module last changed.  A pass that mutated `module` before
    /// asking gets the analysis of what it passes in, not of what it was
    /// handed: the readers price by the generator, which reads expression
    /// types only, and a statement edit moves none.  `Err` means `module`
    /// is invalid; the driver then re-runs the pipeline validating after
    /// every pass, which names and rolls back the pass that produced it.
    pub fn info(&self, module: &naga::Module) -> Result<Ref<'_, naga::valid::ModuleInfo>, Error> {
        self.info.get(module)
    }

    /// [`access_static_lengths`] of `function`, the body at `body` in
    /// `all_functions` order, computed once per module state.  For a
    /// function the pass has not edited yet: after its own rewrite a pass
    /// sizes the lengths itself.
    pub(crate) fn access_lens(
        &self,
        body: usize,
        function: &naga::Function,
        module: &naga::Module,
    ) -> AccessLens {
        let mut lens = self.analyses.access_lens.borrow_mut();
        if lens.len() <= body {
            lens.resize(body + 1, None);
        }
        Rc::clone(lens[body].get_or_insert_with(|| access_static_lengths(function, module).into()))
    }
}

/// Per-body analyses the passes read at entry, kept while the module is
/// unchanged: the driver empties the cache on every accepted change, so a
/// hit describes the module as handed to the pass.  The typifier walk
/// behind each entry was repeated by every pass of a sweep, most of which
/// changed nothing between them.
#[derive(Debug, Default)]
pub struct AnalysisCache {
    access_lens: RefCell<Vec<Option<AccessLens>>>,
}

/// One body's [`access_static_lengths`], shared by reference.
pub(crate) type AccessLens = Rc<[Option<IndexBound>]>;

impl AnalysisCache {
    /// The module changed: every entry is stale.
    pub(super) fn clear(&self) {
        self.access_lens.borrow_mut().clear();
    }
}

/// The module's validation info, computed when first read and dropped on
/// every accepted change: most passes never read it, so validating after
/// each of them was most of what the driver spent on validation.
#[derive(Debug, Default)]
pub struct ModuleInfoCell {
    info: RefCell<Option<naga::valid::ModuleInfo>>,
    /// Set by a validation here that failed, so the driver can tell an
    /// invalid module from any other error a pass returns.
    failed: Cell<bool>,
    #[cfg(test)]
    pub(crate) validations: Cell<usize>,
}

impl ModuleInfoCell {
    /// A cell holding `info`, which describes the module as it is now.
    pub fn new(info: naga::valid::ModuleInfo) -> Self {
        Self {
            info: RefCell::new(Some(info)),
            ..Default::default()
        }
    }

    /// The module changed: the next read validates it.
    pub(super) fn invalidate(&self) {
        *self.info.borrow_mut() = None;
    }

    /// Info the driver validated itself.
    pub(super) fn set(&self, info: naga::valid::ModuleInfo) {
        *self.info.borrow_mut() = Some(info);
    }

    pub(super) fn failed(&self) -> bool {
        self.failed.get()
    }

    /// The info, validating `module` when the cell is empty.
    pub(super) fn get(
        &self,
        module: &naga::Module,
    ) -> Result<Ref<'_, naga::valid::ModuleInfo>, Error> {
        if self.info.borrow().is_none() {
            let info = self.validate(module)?;
            *self.info.borrow_mut() = Some(info);
        }
        Ok(Ref::map(self.info.borrow(), |info| {
            info.as_ref().expect("filled above")
        }))
    }

    pub(super) fn take_or_validate(
        &self,
        module: &naga::Module,
    ) -> Result<naga::valid::ModuleInfo, Error> {
        match self.info.borrow_mut().take() {
            Some(info) => Ok(info),
            None => self.validate(module),
        }
    }

    fn validate(&self, module: &naga::Module) -> Result<naga::valid::ModuleInfo, Error> {
        #[cfg(test)]
        self.validations.set(self.validations.get() + 1);
        crate::io::validate_module(module).inspect_err(|_| self.failed.set(true))
    }
}

/// The tail's rename renders the module once to rank names by the text;
/// that render leaves out the preamble's declarations as the shipped one
/// does, and leaves itself behind for the emission that ships.
#[derive(Debug, Default)]
pub struct TailRender {
    /// The module-scope names the preamble declares.
    pub preamble_names: std::collections::HashSet<String>,
    /// Set by the rename that renders.
    pub render: RefCell<Option<Rendered>>,
}

/// The tail's render of the module: its type spellings price the alias
/// plan of the emission that ships (`GenerateOptions::type_uses`) and its
/// analyses are that emission's (`generator::generate_after`); when
/// weighing moved no name it is the shipped text itself unless its own
/// spellings change the type plan (`generator::generate_reusing`).
#[derive(Debug)]
pub struct Rendered {
    /// The text with its type spellings and analyses.
    pub emission: crate::generator::Emission,
    /// Whether weighing kept every name the render spelled.
    pub names_kept: bool,
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
        let info = ModuleInfoCell::new(crate::io::validate_module(module)?);
        let analyses = AnalysisCache::default();
        pass.run(
            module,
            &PassContext {
                config,
                info: &info,
                name_log: None,
                tail: None,
                analyses: &analyses,
            },
        )
    }
}
