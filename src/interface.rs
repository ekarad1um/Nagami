//! The host interface - what pipeline creation, bind groups, pipeline
//! constants and inter-stage matching read - as canonical text rendered
//! from the IR, so the module that entered the pipeline and the module the
//! shipped text re-parses to compare line by line.  Every line is
//! structural (a type is its layout, a binding its attributes, an override
//! its id or name), so renaming and respelling leave it alone.
//!
//! Per entry point: its I/O, its stage attributes (workgroup size, early
//! depth test, mesh and payload types) and the resources it statically
//! accesses - WGSL's term: referenced anywhere in its call graph, dead code
//! and the pins of `crate::pins` included, what an automatic pipeline
//! layout lists.  Module-wide: the overrides and the diagnostic filters.
//! Without entry points: every declaration, a library's exports being its
//! boundary.  Names only where a host reads them: under
//! `preserve_interface` the bound globals, the overrides and the struct
//! types a bound global reaches (a library's functions and other structs
//! rename like any, the name map covering them), and preserve-listed
//! symbols, whose signatures are then a contract.  Directives are read
//! nowhere here: the emitter derives them from the IR and the re-parse
//! judges them.

use std::fmt::Write as _;

use crate::config::{Config, FloatPrecision};
use crate::error::Error;
use crate::generator::syntax;
use crate::handle_set::HandleSet;
use crate::ir::visit::for_each_statement;
use crate::passes::expr_util::lit_key;

/// What the rendering reads beyond the structural interface.
#[derive(Clone, Default)]
pub(crate) struct Options {
    /// Names a by-name host reads: bound globals, overrides and the struct
    /// types and members they reach ([`Config::preserve_interface`]).
    pub names: bool,
    /// Compare float literals (override defaults, preserved constants) by
    /// value; off under a lossy float precision, where the text rounds them.
    pub float_values: bool,
    /// Preserve-listed symbols: a module-scope declaration of that name
    /// keeps its kind and signature.
    pub symbols: Vec<String>,
}

impl Options {
    pub(crate) fn from_config(config: &Config) -> Self {
        Self {
            names: config.preserve_interface,
            float_values: config.float_precision == FloatPrecision::default(),
            symbols: config.preserve_symbols.clone(),
        }
    }
}

/// Sorted lines, a multiset: two library functions can share a signature.
pub(crate) struct Interface {
    lines: Vec<String>,
    options: Options,
}

impl Interface {
    pub(crate) fn of(module: &naga::Module, options: Options) -> Self {
        let named = if options.names {
            host_named_structs(module)
        } else {
            HandleSet::default()
        };
        let mut render = Render {
            module,
            options: &options,
            named,
            out: Vec::new(),
        };
        render.module_lines();
        let mut lines = render.out;
        lines.sort_unstable();
        Interface { lines, options }
    }

    /// Every line only one side has: `'-'` for `self`, `'+'` for `other`.
    pub(crate) fn differences<'a>(&'a self, other: &'a Interface) -> Vec<(char, &'a str)> {
        let (mut i, mut j) = (0, 0);
        let mut diff = Vec::new();
        while i < self.lines.len() || j < other.lines.len() {
            match (self.lines.get(i), other.lines.get(j)) {
                (Some(a), Some(b)) if a == b => {
                    i += 1;
                    j += 1;
                }
                (Some(a), Some(b)) if a < b => {
                    diff.push(('-', a.as_str()));
                    i += 1;
                }
                (Some(a), None) => {
                    diff.push(('-', a.as_str()));
                    i += 1;
                }
                (_, Some(b)) => {
                    diff.push(('+', b.as_str()));
                    j += 1;
                }
                (None, None) => unreachable!("the loop condition leaves a line on one side"),
            }
        }
        diff
    }

    /// The [`Self::differences`], rendered; `None` when there are none.
    pub(crate) fn mismatch(&self, other: &Interface) -> Option<String> {
        let mut diff = String::new();
        for (sign, line) in self.differences(other) {
            diff.push_str("\n  ");
            diff.push(sign);
            diff.push(' ');
            diff.push_str(line);
        }
        (!diff.is_empty()).then_some(diff)
    }

    /// `module` against the module `self` was rendered from; the message
    /// wants its subject prefixed (`pass 'x' changed the host interface`).
    pub(crate) fn check(&self, module: &naga::Module) -> Result<(), Error> {
        match self.mismatch(&Interface::of(module, self.options.clone())) {
            None => Ok(()),
            Some(diff) => Err(Error::Validation(format!(
                "changed the host interface (- before, + after):{diff}"
            ))),
        }
    }

    /// Parse and validate `text`, then [`Self::check`] the module it yields.
    pub(crate) fn check_text(&self, text: &str) -> Result<(), Error> {
        let module = crate::io::parse_and_validate_wgsl_text(text)?;
        self.check(&module)
            .map_err(|e| Error::Validation(format!("the text {e}")))
    }
}

/// The struct types a host reads by name under `preserve_interface`: the
/// ones a bound global's type reaches through arrays, binding arrays,
/// pointers and members.  The option's preserve list is built from it.
pub(crate) fn host_named_structs(module: &naga::Module) -> HandleSet<naga::Type> {
    let mut pending: Vec<naga::Handle<naga::Type>> = module
        .global_variables
        .iter()
        .filter(|(_, global)| global.binding.is_some())
        .map(|(_, global)| global.ty)
        .collect();
    let mut seen = HandleSet::default();
    let mut structs = HandleSet::default();
    while let Some(ty) = pending.pop() {
        if !seen.insert(ty) {
            continue;
        }
        match &module.types[ty].inner {
            naga::TypeInner::Struct { members, .. } => {
                structs.insert(ty);
                for member in members {
                    pending.push(member.ty);
                }
            }
            naga::TypeInner::Array { base, .. }
            | naga::TypeInner::BindingArray { base, .. }
            | naga::TypeInner::Pointer { base, .. } => pending.push(*base),
            _ => {}
        }
    }
    structs
}

/// Whether a host sees `global`: a resource, a uniform or storage buffer,
/// an immediate - every space but the shader's own.
pub(crate) fn host_visible(global: &naga::GlobalVariable) -> bool {
    !matches!(
        global.space,
        naga::AddressSpace::Function | naga::AddressSpace::Private | naga::AddressSpace::WorkGroup
    )
}

/// The globals a function statically accesses in WGSL's sense: referenced
/// by an expression anywhere in its call graph, dead code, phony
/// assignments and pins included.  Computed here rather than read from
/// naga's analysis, which attributes no access to a global passed by
/// pointer.  Call graphs are memoized per function, so a walk over every
/// function of a module is one pass over its arenas.
pub(crate) struct StaticUses<'m> {
    module: &'m naga::Module,
    memo: Vec<Option<HandleSet<naga::GlobalVariable>>>,
}

impl<'m> StaticUses<'m> {
    pub(crate) fn new(module: &'m naga::Module) -> Self {
        Self {
            module,
            memo: vec![None; module.functions.len()],
        }
    }

    /// The uses of `function`, which may be an entry point's: its own
    /// arena's globals and its callees'.
    pub(crate) fn of(&mut self, function: &naga::Function) -> HandleSet<naga::GlobalVariable> {
        let mut uses = self.of_callees(function);
        for (_, expr) in function.expressions.iter() {
            if let naga::Expression::GlobalVariable(global) = *expr {
                uses.insert(global);
            }
        }
        uses
    }

    /// The uses of `function`'s callees' call graphs: what their texts
    /// answer for (`crate::pins`).
    pub(crate) fn of_callees(
        &mut self,
        function: &naga::Function,
    ) -> HandleSet<naga::GlobalVariable> {
        let mut callees = Vec::new();
        for_each_statement(&function.body, &mut |stmt| {
            if let naga::Statement::Call { function, .. } = stmt {
                callees.push(*function);
            }
        });
        let mut uses = HandleSet::default();
        for callee in callees {
            uses.extend(self.of_function(callee).iter().copied());
        }
        uses
    }

    pub(crate) fn of_function(
        &mut self,
        handle: naga::Handle<naga::Function>,
    ) -> &HandleSet<naga::GlobalVariable> {
        if self.memo[handle.index()].is_none() {
            let module = self.module;
            let uses = self.of(&module.functions[handle]);
            self.memo[handle.index()] = Some(uses);
        }
        self.memo[handle.index()].as_ref().expect("just computed")
    }
}

struct Render<'a> {
    module: &'a naga::Module,
    options: &'a Options,
    /// [`host_named_structs`] when names are read, else empty.
    named: HandleSet<naga::Type>,
    out: Vec<String>,
}

impl Render<'_> {
    fn module_lines(&mut self) {
        let module = self.module;
        let mut static_uses = StaticUses::new(module);
        for ep in &module.entry_points {
            let head = format!("entry {} {}", stage_name(ep.stage), ep.name);
            let declaration = self.entry_declaration(ep, &head);
            self.out.push(declaration);
            for arg in &ep.function.arguments {
                self.io_lines(&head, "in", arg.ty, arg.binding.as_ref());
            }
            if let Some(result) = &ep.function.result {
                self.io_lines(&head, "out", result.ty, result.binding.as_ref());
            }
            let uses = static_uses.of(&ep.function);
            self.use_lines(&head, &uses);
        }
        if module.entry_points.is_empty() {
            for (_, global) in module.global_variables.iter() {
                let names = self.options.names && global.binding.is_some();
                if let Some(line) = self.global_key(global, names) {
                    self.out.push(format!("global {line}"));
                }
            }
            // A library function's call graph is part of every entry
            // point a host composes it into.
            for (handle, function) in module.functions.iter() {
                let head = format!("fn {}", self.function_key(function, false));
                let uses = static_uses.of_function(handle).clone();
                self.use_lines(&head, &uses);
                self.out.push(head);
            }
            for (handle, ty) in module.types.iter() {
                if matches!(ty.inner, naga::TypeInner::Struct { .. }) {
                    let line = self.type_key(handle);
                    self.out.push(format!("type {line}"));
                }
            }
        } else if self.options.names {
            for (_, global) in module.global_variables.iter() {
                if global.binding.is_some()
                    && let Some(line) = self.global_key(global, true)
                {
                    self.out.push(format!("global {line}"));
                }
            }
        }
        for (handle, over) in module.overrides.iter() {
            // naga's own, for an override-expression array size: no key a
            // host could set.
            if over.name.is_none() && over.id.is_none() {
                continue;
            }
            let mut line = format!("override {}", self.override_key(handle));
            if self.options.names
                && over.id.is_some()
                && let Some(name) = &over.name
            {
                line.push_str(" named ");
                line.push_str(name);
            }
            line.push(':');
            line.push_str(&self.type_key(over.ty));
            match over.init {
                Some(init) => {
                    line.push('=');
                    self.expr_key(init, &mut line);
                }
                None => line.push_str(" required"),
            }
            self.out.push(line);
        }
        let mut next = module.diagnostic_filter_leaf;
        while let Some(handle) = next {
            let node = &module.diagnostic_filters[handle];
            self.out.push(format!(
                "diagnostic({},{})",
                syntax::severity_name(node.inner.new_severity),
                syntax::triggering_rule_name(&node.inner.triggering_rule)
            ));
            next = node.parent;
        }
        self.symbol_lines();
    }

    /// Preserve-listed globals, functions and overrides: the compaction
    /// anchor keeps them, so their kind and signature are the contract.
    fn symbol_lines(&mut self) {
        let module = self.module;
        let listed = |name: &Option<String>| {
            name.as_deref()
                .is_some_and(|n| self.options.symbols.iter().any(|s| s == n))
        };
        for (_, global) in module.global_variables.iter() {
            if listed(&global.name) {
                let mut line = format!(
                    "symbol var {}:{}",
                    global.name.as_deref().unwrap_or_default(),
                    self.type_key(global.ty)
                );
                if let Some(binding) = &global.binding {
                    let _ = write!(
                        line,
                        " @group({})@binding({})",
                        binding.group, binding.binding
                    );
                }
                self.out.push(line);
            }
        }
        for (_, function) in module.functions.iter() {
            if listed(&function.name) {
                let line = self.function_key(function, true);
                self.out.push(format!("symbol fn {line}"));
            }
        }
        for (_, over) in module.overrides.iter() {
            if listed(&over.name) {
                self.out.push(format!(
                    "symbol override {}",
                    over.name.as_deref().unwrap_or_default()
                ));
            }
        }
    }

    fn entry_declaration(&self, ep: &naga::EntryPoint, head: &str) -> String {
        let mut line = head.to_owned();
        if matches!(
            ep.stage,
            naga::ShaderStage::Compute | naga::ShaderStage::Task | naga::ShaderStage::Mesh
        ) {
            line.push_str(" @workgroup_size(");
            for axis in 0..3 {
                if axis > 0 {
                    line.push(',');
                }
                match ep.workgroup_size_overrides.as_ref().and_then(|o| o[axis]) {
                    Some(expr) => self.expr_key(expr, &mut line),
                    None => {
                        let _ = write!(line, "{}", ep.workgroup_size[axis]);
                    }
                }
            }
            line.push(')');
        }
        if let Some(test) = ep.early_depth_test {
            line.push(' ');
            line.push_str(syntax::early_depth_test_attr(test));
        }
        if let Some(mesh) = &ep.mesh_info {
            let _ = write!(line, " mesh(topology{},", mesh.topology as u8);
            match mesh.max_vertices_override {
                Some(expr) => self.expr_key(expr, &mut line),
                None => {
                    let _ = write!(line, "{}", mesh.max_vertices);
                }
            }
            line.push(',');
            match mesh.max_primitives_override {
                Some(expr) => self.expr_key(expr, &mut line),
                None => {
                    let _ = write!(line, "{}", mesh.max_primitives);
                }
            }
            let _ = write!(
                line,
                ",{},{})",
                self.type_key(mesh.vertex_output_type),
                self.type_key(mesh.primitive_output_type)
            );
        }
        if let Some(payload) = ep.task_payload {
            let _ = write!(
                line,
                " task_payload {}",
                self.type_key(self.module.global_variables[payload].ty)
            );
        }
        if let Some(payload) = ep.incoming_ray_payload {
            let _ = write!(
                line,
                " incoming_ray_payload {}",
                self.type_key(self.module.global_variables[payload].ty)
            );
        }
        line
    }

    /// One line per bound value; an unbound struct argument or result is
    /// its members' bindings.
    fn io_lines(
        &mut self,
        head: &str,
        direction: &str,
        ty: naga::Handle<naga::Type>,
        binding: Option<&naga::Binding>,
    ) {
        match binding {
            Some(binding) => {
                let line = format!(
                    "{head} {direction} {} {}",
                    self.binding_key(binding, ty),
                    self.type_key(ty)
                );
                self.out.push(line);
            }
            None => {
                if let naga::TypeInner::Struct { members, .. } = &self.module.types[ty].inner {
                    for member in members {
                        self.io_lines(head, direction, member.ty, member.binding.as_ref());
                    }
                }
            }
        }
    }

    /// The attributes as the emitter spells them, on a binding brought to
    /// the spec's defaults: `@interpolate(perspective)` written out and the
    /// default it re-parses to are one binding, `flat` and `flat, first`
    /// too.
    fn binding_key(&self, binding: &naga::Binding, ty: naga::Handle<naga::Type>) -> String {
        let mut binding = binding.clone();
        if let naga::Binding::Location {
            interpolation,
            sampling,
            ..
        } = &mut binding
        {
            let float = self.module.types[ty].inner.scalar_kind() == Some(naga::ScalarKind::Float);
            let interpolation = *interpolation.get_or_insert(if float {
                naga::Interpolation::Perspective
            } else {
                naga::Interpolation::Flat
            });
            match interpolation {
                naga::Interpolation::Flat => {
                    sampling.get_or_insert(naga::Sampling::First);
                }
                naga::Interpolation::PerVertex => *sampling = None,
                naga::Interpolation::Perspective | naga::Interpolation::Linear => {
                    sampling.get_or_insert(naga::Sampling::Center);
                }
            }
        }
        syntax::binding_attrs(&binding, true)
            .map_or_else(|_| "?".to_owned(), |attrs| attrs.trim_end().to_owned())
    }

    /// One `<head> uses <global>` line per host-visible global in `uses`.
    fn use_lines(&mut self, head: &str, uses: &HandleSet<naga::GlobalVariable>) {
        for (handle, global) in self.module.global_variables.iter() {
            if uses.contains(handle)
                && let Some(line) = self.global_key(global, false)
            {
                self.out.push(format!("{head} uses {line}"));
            }
        }
    }

    /// A host-visible global: binding, address space and type; `None` for
    /// the spaces no host sees.
    fn global_key(&self, global: &naga::GlobalVariable, names: bool) -> Option<String> {
        if !host_visible(global) {
            return None;
        }
        let space = match global.space {
            naga::AddressSpace::Handle => String::new(),
            naga::AddressSpace::Storage { access } => {
                format!("<storage,{}>", syntax::storage_access(access))
            }
            space => format!("<{}>", syntax::address_space(space)),
        };
        let mut line = String::new();
        if let Some(binding) = &global.binding {
            let _ = write!(
                line,
                "@group({})@binding({}) ",
                binding.group, binding.binding
            );
        }
        line.push_str("var");
        line.push_str(&space);
        if names && let Some(name) = &global.name {
            line.push(' ');
            line.push_str(name);
        }
        line.push(':');
        line.push_str(&self.type_key(global.ty));
        Some(line)
    }

    fn function_key(&self, function: &naga::Function, names: bool) -> String {
        let mut line = String::new();
        if names && let Some(name) = &function.name {
            line.push_str(name);
        }
        line.push('(');
        for (i, arg) in function.arguments.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            line.push_str(&self.type_key(arg.ty));
        }
        line.push(')');
        if let Some(result) = &function.result {
            line.push_str("->");
            line.push_str(&self.type_key(result.ty));
        }
        line
    }

    /// A type by structure: a struct is its members' offsets and its span,
    /// an array its stride, so `@align` / `@size` count by their effect.
    fn type_key(&self, handle: naga::Handle<naga::Type>) -> String {
        use naga::TypeInner as T;
        let ty = &self.module.types[handle];
        match &ty.inner {
            T::Scalar(scalar) => scalar_key(*scalar),
            T::Vector { size, scalar } => format!("vec{}<{}>", *size as u8, scalar_key(*scalar)),
            T::Matrix {
                columns,
                rows,
                scalar,
            } => format!(
                "mat{}x{}<{}>",
                *columns as u8,
                *rows as u8,
                scalar_key(*scalar)
            ),
            T::Atomic(scalar) => format!("atomic<{}>", scalar_key(*scalar)),
            T::Array { base, size, stride } => format!(
                "array<{},{}>@stride({stride})",
                self.type_key(*base),
                self.array_size(*size)
            ),
            T::BindingArray { base, size } => format!(
                "binding_array<{},{}>",
                self.type_key(*base),
                self.array_size(*size)
            ),
            T::Struct { members, span } => {
                let named = self.named.contains(handle);
                let mut line = String::from("struct");
                if named && let Some(name) = &ty.name {
                    line.push(' ');
                    line.push_str(name);
                }
                line.push('{');
                for (i, member) in members.iter().enumerate() {
                    if i > 0 {
                        line.push(',');
                    }
                    if named && let Some(name) = &member.name {
                        line.push_str(name);
                        line.push(':');
                    }
                    let _ = write!(line, "{}@{}", self.type_key(member.ty), member.offset);
                }
                let _ = write!(line, "}}@size({span})");
                line
            }
            T::Image {
                dim,
                arrayed,
                class,
            } => syntax::image_type(*dim, *arrayed, *class).unwrap_or_else(|_| "image?".to_owned()),
            T::Sampler { comparison } => if *comparison {
                "sampler_comparison"
            } else {
                "sampler"
            }
            .to_owned(),
            T::AccelerationStructure { vertex_return } => format!(
                "acceleration_structure{}",
                vertex_return_suffix(*vertex_return)
            ),
            T::RayQuery { vertex_return } => {
                format!("ray_query{}", vertex_return_suffix(*vertex_return))
            }
            T::Pointer { base, space } => format!(
                "ptr<{},{}>",
                syntax::address_space(*space),
                self.type_key(*base)
            ),
            T::ValuePointer { .. } | T::CooperativeMatrix { .. } => {
                syntax::type_inner_kind(&ty.inner).to_owned()
            }
        }
    }

    fn array_size(&self, size: naga::ArraySize) -> String {
        match size {
            naga::ArraySize::Constant(n) => n.to_string(),
            naga::ArraySize::Dynamic => "dynamic".to_owned(),
            naga::ArraySize::Pending(over) => format!("override({})", self.override_key(over)),
        }
    }

    /// The pipeline-constant key: the `@id`, else the name.
    fn override_key(&self, handle: naga::Handle<naga::Override>) -> String {
        let over = &self.module.overrides[handle];
        match over.id {
            Some(id) => format!("id({id})"),
            None => over.name.clone().unwrap_or_default(),
        }
    }

    /// A global expression by structure: constants dissolve into their
    /// values (they are not interface), overrides stand by their key.
    fn expr_key(&self, handle: naga::Handle<naga::Expression>, out: &mut String) {
        use naga::Expression as E;
        let list = |this: &Self, out: &mut String, items: &[naga::Handle<naga::Expression>]| {
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                this.expr_key(*item, out);
            }
            out.push(')');
        };
        match &self.module.global_expressions[handle] {
            E::Literal(literal) => out.push_str(&self.literal_key(*literal)),
            E::Constant(constant) => self.expr_key(self.module.constants[*constant].init, out),
            E::Override(over) => {
                let _ = write!(out, "override({})", self.override_key(*over));
            }
            // A scalar's zero value and its literal zero are one default:
            // `override o: f32 = f32()` ships as `=0`.
            E::ZeroValue(ty) => {
                let scalar_zero = match self.module.types[*ty].inner {
                    naga::TypeInner::Scalar(scalar) => naga::Literal::zero(scalar),
                    _ => None,
                };
                match scalar_zero {
                    Some(zero) => out.push_str(&self.literal_key(zero)),
                    None => {
                        let _ = write!(out, "{}()", self.type_key(*ty));
                    }
                }
            }
            E::Compose { ty, components } => {
                let _ = write!(out, "{}(", self.type_key(*ty));
                list(self, out, components);
            }
            E::Splat { size, value } => {
                let _ = write!(out, "splat{}(", *size as u8);
                list(self, out, &[*value]);
            }
            E::Unary { op, expr } => {
                let _ = write!(out, "unary{}(", *op as u8);
                list(self, out, &[*expr]);
            }
            E::Binary { op, left, right } => {
                let _ = write!(out, "binary{}(", *op as u8);
                list(self, out, &[*left, *right]);
            }
            E::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                let _ = write!(out, "{}(", syntax::math_name(*fun));
                let args: Vec<_> = [Some(*arg), *arg1, *arg2, *arg3]
                    .into_iter()
                    .flatten()
                    .collect();
                list(self, out, &args);
            }
            E::As {
                expr,
                kind,
                convert,
            } => {
                let _ = write!(
                    out,
                    "as{}{}(",
                    kind_key(*kind),
                    convert.map_or(0, u32::from)
                );
                list(self, out, &[*expr]);
            }
            E::Select {
                condition,
                accept,
                reject,
            } => {
                out.push_str("select(");
                list(self, out, &[*condition, *accept, *reject]);
            }
            E::Relational { fun, argument } => {
                let _ = write!(out, "relational{}(", *fun as u8);
                list(self, out, &[*argument]);
            }
            E::Access { base, index } => {
                out.push_str("access(");
                list(self, out, &[*base, *index]);
            }
            E::AccessIndex { base, index } => {
                let _ = write!(out, "index{index}(");
                list(self, out, &[*base]);
            }
            E::Swizzle {
                size,
                vector,
                pattern,
            } => {
                let _ = write!(out, "swizzle{}", *size as u8);
                for component in &pattern[..*size as usize] {
                    let _ = write!(out, "{}", *component as u8);
                }
                out.push('(');
                list(self, out, &[*vector]);
            }
            E::Load { pointer } => {
                out.push_str("load(");
                list(self, out, &[*pointer]);
            }
            _ => out.push_str("other"),
        }
    }

    fn literal_key(&self, literal: naga::Literal) -> String {
        use naga::Literal as L;
        if !self.options.float_values
            && matches!(
                literal,
                L::F64(_) | L::F32(_) | L::F16(_) | L::AbstractFloat(_)
            )
        {
            return "float".to_owned();
        }
        let (kind, bits) = lit_key(literal);
        format!("lit{kind}:{bits:x}")
    }
}

fn kind_key(kind: naga::ScalarKind) -> &'static str {
    match kind {
        naga::ScalarKind::Sint => "i",
        naga::ScalarKind::Uint => "u",
        naga::ScalarKind::Float => "f",
        naga::ScalarKind::Bool => "bool",
        naga::ScalarKind::AbstractInt => "abstract_int",
        naga::ScalarKind::AbstractFloat => "abstract_float",
    }
}

fn scalar_key(scalar: naga::Scalar) -> String {
    match scalar.kind {
        naga::ScalarKind::Bool => "bool".to_owned(),
        kind => format!("{}{}", kind_key(kind), u32::from(scalar.width) * 8),
    }
}

fn stage_name(stage: naga::ShaderStage) -> &'static str {
    match stage {
        naga::ShaderStage::Vertex => "vertex",
        naga::ShaderStage::Fragment => "fragment",
        naga::ShaderStage::Compute => "compute",
        naga::ShaderStage::Task => "task",
        naga::ShaderStage::Mesh => "mesh",
        naga::ShaderStage::RayGeneration => "ray_generation",
        naga::ShaderStage::AnyHit => "any_hit",
        naga::ShaderStage::ClosestHit => "closest_hit",
        naga::ShaderStage::Miss => "miss",
    }
}

fn vertex_return_suffix(vertex_return: bool) -> &'static str {
    if vertex_return { "<vertex_return>" } else { "" }
}

#[cfg(test)]
#[path = "interface_tests.rs"]
mod tests;
