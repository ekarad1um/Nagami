//! Module-level emission driver: [`Generator::generate_module`] emits every
//! module-scope declaration in the order WGSL requires (directives, aliases,
//! structs, constants, overrides, globals, functions, entry points), each
//! section gated on liveness and on the alias plan computed up front.

use crate::error::Error;

use super::core::{ExprTypes, FunctionCtx, Generator};
use super::syntax::{address_space, binding_attrs, storage_access};

mod call_inline;
mod defer_vars;
mod local_resolve;
mod must_bind;
mod ref_counts;

pub(super) use defer_vars::find_deferrable_vars;
pub(super) use local_resolve::local_var_in_stmts;

use crate::handle_set::HandleSet;
use call_inline::{compute_pure_functions, find_inlineable_calls};
use defer_vars::find_for_loop_vars;
use must_bind::compute_must_bind_loads;
use ref_counts::{compute_expression_ref_counts, discount_initializer_refs};

/// The `enable` directives the emitted text needs.  Mirrors naga's own
/// writer scan except where noted: a liveness gate on 16-bit scalars, the
/// expression scans for 16-bit values that register no standalone type, and
/// `ray_query` (naga's writer has no ray-query arm at all).
#[derive(Default)]
struct Enables {
    f16: bool,
    int16: bool,
    dual_source_blending: bool,
    clip_distances: bool,
    mesh_shaders: bool,
    binding_array: bool,
    draw_index: bool,
    primitive_index: bool,
    cooperative_matrix: bool,
    ray_tracing: bool,
    ray_query: bool,
    ray_query_vertex_return: bool,
}

impl Enables {
    fn note_binding(&mut self, binding: &naga::Binding) {
        match *binding {
            naga::Binding::Location {
                blend_src: Some(_), ..
            } => self.dual_source_blending = true,
            naga::Binding::BuiltIn(naga::BuiltIn::ClipDistances) => self.clip_distances = true,
            naga::Binding::BuiltIn(naga::BuiltIn::PrimitiveIndex) => self.primitive_index = true,
            naga::Binding::BuiltIn(naga::BuiltIn::DrawIndex) => self.draw_index = true,
            naga::Binding::Location {
                per_primitive: true,
                ..
            } => self.mesh_shaders = true,
            naga::Binding::BuiltIn(
                naga::BuiltIn::RayInvocationId
                | naga::BuiltIn::NumRayInvocations
                | naga::BuiltIn::InstanceCustomData
                | naga::BuiltIn::GeometryIndex
                | naga::BuiltIn::WorldRayOrigin
                | naga::BuiltIn::WorldRayDirection
                | naga::BuiltIn::ObjectRayOrigin
                | naga::BuiltIn::ObjectRayDirection
                | naga::BuiltIn::RayTmin
                | naga::BuiltIn::RayTCurrentMax
                | naga::BuiltIn::ObjectToWorld
                | naga::BuiltIn::WorldToObject,
            ) => self.ray_tracing = true,
            _ => {}
        }
    }

    fn scan(module: &naga::Module, live_types: &HandleSet<naga::Type>) -> Self {
        let mut e = Enables {
            mesh_shaders: module.uses_mesh_shaders(),
            ..Default::default()
        };
        let mut has_acceleration_structure = false;
        for (h, ty) in module.types.iter() {
            match ty.inner {
                // Liveness-gated, unlike naga's raw scan: the compactor roots
                // `special_types`, so a dead `__frexp_result_f16` would keep
                // `enable f16;` on text with no f16 token (non-idempotent, and
                // over-declares a device feature).  Under-detection is
                // fail-safe: the re-parse self-check rejects f16 text without
                // the enable.
                naga::TypeInner::Scalar(s)
                | naga::TypeInner::Vector { scalar: s, .. }
                | naga::TypeInner::Matrix { scalar: s, .. }
                    if live_types.contains(h) =>
                {
                    e.f16 |= s == naga::Scalar::F16;
                    e.int16 |= s == naga::Scalar::I16 || s == naga::Scalar::U16;
                }
                naga::TypeInner::Struct { ref members, .. } => {
                    for binding in members.iter().filter_map(|m| m.binding.as_ref()) {
                        e.note_binding(binding);
                    }
                }
                naga::TypeInner::CooperativeMatrix { .. } => e.cooperative_matrix = true,
                // `acceleration_structure` parses under either `wgpu_ray_query`
                // or `wgpu_ray_tracing_pipeline`; resolved once every pipeline
                // signal is known.
                naga::TypeInner::AccelerationStructure { vertex_return } => {
                    has_acceleration_structure = true;
                    e.ray_query_vertex_return |= vertex_return;
                }
                naga::TypeInner::RayQuery { vertex_return } => {
                    e.ray_query = true;
                    e.ray_query_vertex_return |= vertex_return;
                }
                // naga 30 requires this to parse a `binding_array<...>`; it is
                // naga-only and `run` strips it from the shipped text.
                naga::TypeInner::BindingArray { .. } => e.binding_array = true,
                _ => {}
            }
        }
        // A bare 16-bit literal or a value-changing cast to a 16-bit scalar
        // that survives folding (a runtime operand) registers no standalone
        // type yet still emits text the enable must cover.
        if !e.f16 {
            e.f16 = any_expression(module, |expr| {
                matches!(
                    expr,
                    naga::Expression::Literal(naga::Literal::F16(_))
                        | naga::Expression::As {
                            kind: naga::ScalarKind::Float,
                            convert: Some(2),
                            ..
                        }
                )
            });
        }
        if !e.int16 {
            e.int16 = any_expression(module, |expr| {
                matches!(
                    expr,
                    naga::Expression::Literal(naga::Literal::I16(_) | naga::Literal::U16(_))
                        | naga::Expression::As {
                            kind: naga::ScalarKind::Sint | naga::ScalarKind::Uint,
                            convert: Some(2),
                            ..
                        }
                )
            });
        }
        for ep in &module.entry_points {
            if let Some(res) = ep.function.result.as_ref().and_then(|r| r.binding.as_ref()) {
                e.note_binding(res);
            }
            for binding in ep
                .function
                .arguments
                .iter()
                .filter_map(|a| a.binding.as_ref())
            {
                e.note_binding(binding);
            }
        }
        if module.global_variables.iter().any(|(_, gv)| {
            matches!(
                gv.space,
                naga::AddressSpace::RayPayload | naga::AddressSpace::IncomingRayPayload
            )
        }) || module.entry_points.iter().any(|ep| {
            matches!(
                ep.stage,
                naga::ShaderStage::RayGeneration
                    | naga::ShaderStage::AnyHit
                    | naga::ShaderStage::ClosestHit
                    | naga::ShaderStage::Miss
            )
        }) {
            e.ray_tracing = true;
        }
        // With a pipeline signal the pipeline enable covers the type; the IR
        // records no other admitting directive, so a signal-free module is
        // deliberately rewritten to the query enable.
        if has_acceleration_structure && !e.ray_tracing {
            e.ray_query = true;
        }
        // No `enable subgroups;` is ever synthesised: naga's text front-end
        // cannot parse it, and tiny subgroup-only shaders would only grow.
        e
    }

    /// Directive lines in naga's writer order, matching the naga baseline.
    fn directives(&self) -> impl Iterator<Item = &'static str> {
        [
            (self.f16, "enable f16;"),
            (self.int16, "enable wgpu_int16;"),
            (self.dual_source_blending, "enable dual_source_blending;"),
            (self.clip_distances, "enable clip_distances;"),
            (self.mesh_shaders, "enable wgpu_mesh_shader;"),
            (self.binding_array, "enable wgpu_binding_array;"),
            (self.draw_index, "enable draw_index;"),
            (self.primitive_index, "enable primitive_index;"),
            (self.cooperative_matrix, "enable wgpu_cooperative_matrix;"),
            (self.ray_tracing, "enable wgpu_ray_tracing_pipeline;"),
            (self.ray_query, "enable wgpu_ray_query;"),
            (
                self.ray_query_vertex_return,
                "enable wgpu_ray_query_vertex_return;",
            ),
        ]
        .into_iter()
        .filter_map(|(on, text)| on.then_some(text))
    }
}

/// `true` when any expression in any arena (const-init, function, entry
/// point) satisfies `pred`.
fn any_expression(module: &naga::Module, pred: impl Fn(&naga::Expression) -> bool) -> bool {
    module
        .global_expressions
        .iter()
        .chain(crate::passes::expr_util::all_functions(module).flat_map(|f| f.expressions.iter()))
        .any(|(_, e)| pred(e))
}

/// naga's special / predeclared struct types (`RayDesc`, `RayIntersection`,
/// `__modf_result_*`, ...), per naga's own `is_builtin_wgsl_struct`; neither
/// declared nor renamed by the emitter.
pub(super) fn special_struct_handles(module: &naga::Module) -> HandleSet<naga::Type> {
    let st = &module.special_types;
    let mut set: HandleSet<naga::Type> = [
        st.ray_desc,
        st.ray_intersection,
        st.ray_vertex_return,
        st.external_texture_params,
        st.external_texture_transfer_function,
    ]
    .iter()
    .filter_map(|h| *h)
    .collect();
    set.extend(st.predeclared_types.values().copied());
    set
}

/// `true` when a constant's init text already spells its concrete type
/// (`Compose` / `ZeroValue` / `Splat` constructors, a concrete `Literal`'s
/// suffix), making `const NAME: T` redundant.  Single source of truth: the
/// constant emitter omits the annotation and the alias cost model counts the
/// declared type in exactly the complementary cases, or the alias-savings
/// estimate diverges from the output.
pub(super) fn const_init_has_explicit_type(init: &naga::Expression) -> bool {
    match init {
        naga::Expression::Compose { .. }
        | naga::Expression::ZeroValue(_)
        | naga::Expression::Splat { .. } => true,
        naga::Expression::Literal(lit) => !matches!(
            lit,
            naga::Literal::AbstractInt(_) | naga::Literal::AbstractFloat(_)
        ),
        _ => false,
    }
}

impl<'a> Generator<'a> {
    /// Emit the module: the per-function analyses and literal extraction, then
    /// every declaration section in spec order.
    pub(super) fn generate_module(&mut self) -> Result<(), Error> {
        let mut has_prev_section = false;

        self.ref_count_cache = crate::passes::expr_util::all_functions(self.module)
            .map(compute_expression_ref_counts)
            .collect();

        self.pure_functions = compute_pure_functions(self.module);

        self.scan_and_extract_literals();

        // One context for every declaration section: rebuilding it per
        // declaration clones every constant name, quadratic on a module that
        // is mostly constants.  Each emitted constant is named in it below,
        // exactly as `expr_to_const` names it for the function sections.
        let mut decl_ctx = self.module_ctx();

        macro_rules! section_gap {
            ($self:expr, $has_prev:expr) => {
                if $has_prev {
                    $self.push_newline();
                }
            };
        }

        let enables = Enables::scan(self.module, &self.live_types);
        for directive in enables.directives() {
            self.out.push_str(directive);
            self.push_newline();
            has_prev_section = true;
        }

        {
            let mut filters = Vec::new();
            let mut next = self.module.diagnostic_filter_leaf;
            while let Some(handle) = next {
                let node = &self.module.diagnostic_filters[handle];
                filters.push(&node.inner);
                next = node.parent;
            }
            // Definition order is parent -> child, the reverse of the chain walk.
            if !filters.is_empty() {
                section_gap!(self, has_prev_section);
                for filter in filters.iter().rev() {
                    self.out.push_str("diagnostic(");
                    self.out.push_str(severity_name(filter.new_severity));
                    self.out.push(',');
                    self.out
                        .push_str(&triggering_rule_name(&filter.triggering_rule));
                    self.out.push_str(");");
                    self.push_newline();
                }
                has_prev_section = true;
            }
        }

        if !self.type_alias_decls.is_empty() {
            section_gap!(self, has_prev_section);
            let assign_tok = self.assign_sep();
            for i in 0..self.type_alias_decls.len() {
                self.out.push_str("alias ");
                self.out.push_str(&self.type_alias_decls[i].0);
                self.out.push_str(assign_tok);
                self.out.push_str(&self.type_alias_decls[i].1);
                self.out.push(';');
                self.push_newline();
            }
            has_prev_section = true;
        }

        let preamble = self.options.preamble_names.clone();
        // naga's predeclared struct types are not declarable WGSL: a declaration
        // gives the user struct and the predeclared type different type-arena
        // handles, so constructor expressions fail validation.
        let special_type_handles = special_struct_handles(self.module);
        for (h, ty) in self.module.types.iter() {
            if let naga::TypeInner::Struct { members, span } = &ty.inner {
                if !self.live_types.contains(h) {
                    continue;
                }
                if special_type_handles.contains(h) {
                    continue;
                }
                // Host-addressable past the special-type gate, preamble-owned
                // included.
                self.map_visible_structs.insert(h);
                if !preamble.is_empty()
                    && let Some(name) = ty.name.as_deref()
                    && preamble.contains(name)
                {
                    continue;
                }
                section_gap!(self, has_prev_section);
                self.generate_struct(h, members, *span)?;
                self.push_newline();
                has_prev_section = true;
            }
        }

        let has_constants = self.module.constants.iter().any(|(h, c)| {
            self.live_constants.contains(h)
                && c.name.as_deref().is_some_and(|n| !preamble.contains(n))
        });
        if has_constants {
            section_gap!(self, has_prev_section);
        }
        for (h, c) in self.module.constants.iter() {
            if c.name.is_none() {
                continue;
            }
            if !self.live_constants.contains(h) {
                continue;
            }
            if !preamble.is_empty()
                && let Some(name) = c.name.as_deref()
                && preamble.contains(name)
            {
                continue;
            }
            self.out.push_str("const ");
            self.out.push_str(&self.constant_names[h.index()]);
            // `: T` only when the init text does not spell its own type.
            let init_expr = &self.module.global_expressions[c.init];
            let self_typed = const_init_has_explicit_type(init_expr);
            if !self_typed {
                self.push_colon();
                self.out.push_str(&self.type_ref(c.ty)?);
            }
            self.push_assign();
            let expr = self.emit_global_expr_in(c.init, self_typed, &mut decl_ctx)?;
            self.out.push_str(&expr);
            self.out.push(';');
            self.push_newline();
            // Later constants sharing this init handle emit the name, not the tree.
            self.expr_to_const.insert(c.init, h);
            decl_ctx
                .expr_names
                .insert(c.init, self.constant_names[h.index()].clone());
        }
        if has_constants {
            has_prev_section = true;
        }

        let has_overrides = self
            .module
            .overrides
            .iter()
            .any(|(_, ov)| ov.name.as_deref().is_none_or(|n| !preamble.contains(n)));
        if has_overrides {
            section_gap!(self, has_prev_section);
        }
        for (h, ov) in self.module.overrides.iter() {
            if !preamble.is_empty()
                && let Some(name) = ov.name.as_deref()
                && preamble.contains(name)
            {
                continue;
            }
            if let Some(id) = ov.id {
                self.out.push_str("@id(");
                self.out.push_str(&id.to_string());
                self.out
                    .push_str(if self.options.beautify { ") " } else { ")" });
            }
            self.out.push_str("override ");
            self.out.push_str(&self.override_names[h.index()]);
            self.push_colon();
            self.out.push_str(&self.type_ref(ov.ty)?);
            if let Some(init) = ov.init {
                self.push_assign();
                let text = self.emit_global_expr_in(init, false, &mut decl_ctx)?;
                self.out.push_str(&text);
            }
            self.out.push(';');
            self.push_newline();
        }
        if has_overrides {
            has_prev_section = true;
        }

        let has_globals = self
            .module
            .global_variables
            .iter()
            .any(|(_, g)| g.name.as_deref().is_none_or(|n| !preamble.contains(n)));
        if has_globals {
            section_gap!(self, has_prev_section);
        }
        for (h, g) in self.module.global_variables.iter() {
            if !preamble.is_empty()
                && let Some(name) = g.name.as_deref()
                && preamble.contains(name)
            {
                continue;
            }
            if let Some(binding) = g.binding {
                self.out.push_str("@group(");
                self.out.push_str(&binding.group.to_string());
                self.push_binding_sep();
                self.out.push_str(&binding.binding.to_string());
                self.push_attr_end();
            }
            self.out.push_str("var");
            match g.space {
                naga::AddressSpace::Handle => self.out.push(' '),
                naga::AddressSpace::Storage { access } => {
                    self.out.push('<');
                    self.out.push_str("storage");
                    // Elide the default `read`; compare the resolved name so empty
                    // / non-LOAD-only flag sets classify via `storage_access`.
                    let acc_str = storage_access(access);
                    if acc_str != "read" {
                        self.push_separator();
                        self.out.push_str(acc_str);
                    }
                    self.push_angle_end();
                }
                _ => {
                    self.out.push('<');
                    self.out.push_str(address_space(g.space));
                    self.push_angle_end();
                }
            }
            self.out.push_str(&self.global_names[h.index()]);
            self.push_colon();
            self.out.push_str(&self.type_ref(g.ty)?);
            if let Some(init) = g.init {
                self.push_assign();
                let text = self.emit_global_expr_in(init, false, &mut decl_ctx)?;
                self.out.push_str(&text);
            }
            self.out.push(';');
            self.push_newline();
        }
        if has_globals {
            has_prev_section = true;
        }

        // Pre-rendered so the borrow of `extracted_literals` ends before
        // `self.out` is written.
        let extracted_lines: Vec<String> = {
            let assign_tok = self.assign_sep();
            let mut pairs: Vec<(&super::syntax::LiteralExtractKey, &String)> =
                self.extracted_literals.iter().collect();
            pairs.sort_unstable_by_key(|&(_, name)| name.as_str());
            pairs
                .iter()
                .map(|(key, name)| {
                    let mut line = String::from("const ");
                    line.push_str(name);
                    line.push_str(assign_tok);
                    line.push_str(&key.decl_text);
                    line.push(';');
                    line
                })
                .collect()
        };
        if !extracted_lines.is_empty() {
            section_gap!(self, has_prev_section);
            for line in &extracted_lines {
                self.out.push_str(line);
                self.push_newline();
            }
            has_prev_section = true;
        }

        // Preserved names are in scope because a `let` shadowing one is legal
        // but would leave the name map unable to account for the extra
        // occurrences.  Function-locals are not: `local_used_names` tracks the
        // ones actually in scope.
        let mut module_used_names: std::collections::HashSet<String> =
            self.emitted_module_names().map(str::to_owned).collect();
        module_used_names.extend(self.extracted_literals.values().cloned());

        let num_functions = self.module.functions.len();

        for (h, f) in self.module.functions.iter() {
            if !preamble.is_empty()
                && let Some(name) = f.name.as_deref()
                && preamble.contains(name)
            {
                continue;
            }
            section_gap!(self, has_prev_section);
            let fn_name = self.function_names[h.index()].clone();
            self.generate_function(
                &fn_name,
                f,
                &self.info[h],
                None,
                &module_used_names,
                h.index(),
            )?;
            self.push_newline();
            has_prev_section = true;
        }

        for (i, ep) in self.module.entry_points.iter().enumerate() {
            if !preamble.is_empty() && preamble.contains(&ep.name) {
                continue;
            }
            section_gap!(self, has_prev_section);

            self.emit_diagnostic_attrs(ep.function.diagnostic_filter_leaf);

            // This entry point's own recorded payload: scanning for the first
            // `IncomingRayPayload` global would wire every hit / miss entry point
            // to the same payload.
            let incoming_payload_name = ep
                .incoming_ray_payload
                .map(|h| self.global_names[h.index()].clone());

            match ep.stage {
                naga::ShaderStage::Vertex => self.out.push_str("@vertex "),
                naga::ShaderStage::Fragment => self.out.push_str("@fragment "),
                naga::ShaderStage::Compute => {
                    self.out.push_str(if self.options.beautify {
                        "@compute @workgroup_size("
                    } else {
                        "@compute@workgroup_size("
                    });

                    let overrides = ep.workgroup_size_overrides.as_ref();

                    if let Some(handle) = overrides.and_then(|o| o[0]) {
                        let text = self.emit_global_expr_in(handle, false, &mut decl_ctx)?;
                        self.out.push_str(&text);
                    } else {
                        self.out.push_str(&ep.workgroup_size[0].to_string());
                    }

                    let has_dim1 =
                        overrides.and_then(|o| o[1]).is_some() || ep.workgroup_size[1] != 1;
                    let has_dim2 =
                        overrides.and_then(|o| o[2]).is_some() || ep.workgroup_size[2] != 1;

                    if has_dim1 || has_dim2 {
                        self.push_separator();
                        if let Some(handle) = overrides.and_then(|o| o[1]) {
                            let text = self.emit_global_expr_in(handle, false, &mut decl_ctx)?;
                            self.out.push_str(&text);
                        } else {
                            self.out.push_str(&ep.workgroup_size[1].to_string());
                        }
                        if has_dim2 {
                            self.push_separator();
                            if let Some(handle) = overrides.and_then(|o| o[2]) {
                                let text =
                                    self.emit_global_expr_in(handle, false, &mut decl_ctx)?;
                                self.out.push_str(&text);
                            } else {
                                self.out.push_str(&ep.workgroup_size[2].to_string());
                            }
                        }
                    }
                    self.out
                        .push_str(if self.options.beautify { ") " } else { ")" });
                }
                naga::ShaderStage::RayGeneration => self.out.push_str("@ray_generation "),
                naga::ShaderStage::AnyHit => {
                    if let Some(name) = &incoming_payload_name {
                        self.out.push_str(if self.options.beautify {
                            "@any_hit @incoming_payload("
                        } else {
                            "@any_hit@incoming_payload("
                        });
                        self.out.push_str(name);
                        self.out
                            .push_str(if self.options.beautify { ") " } else { ")" });
                    } else {
                        self.out.push_str("@any_hit ");
                    }
                }
                naga::ShaderStage::ClosestHit => {
                    if let Some(name) = &incoming_payload_name {
                        self.out.push_str(if self.options.beautify {
                            "@closest_hit @incoming_payload("
                        } else {
                            "@closest_hit@incoming_payload("
                        });
                        self.out.push_str(name);
                        self.out
                            .push_str(if self.options.beautify { ") " } else { ")" });
                    } else {
                        self.out.push_str("@closest_hit ");
                    }
                }
                naga::ShaderStage::Miss => {
                    if let Some(name) = &incoming_payload_name {
                        self.out.push_str(if self.options.beautify {
                            "@miss @incoming_payload("
                        } else {
                            "@miss@incoming_payload("
                        });
                        self.out.push_str(name);
                        self.out
                            .push_str(if self.options.beautify { ") " } else { ")" });
                    } else {
                        self.out.push_str("@miss ");
                    }
                }
                _ => {
                    return Err(Error::Emit(format!(
                        "unsupported entry point stage for '{}': {:?}",
                        ep.name, ep.stage,
                    )));
                }
            }
            self.generate_function(
                &ep.name,
                &ep.function,
                self.info.get_entry_point(i),
                Some(ep.name.as_str()),
                &module_used_names,
                num_functions + i,
            )?;
            self.push_newline();
            has_prev_section = true;
        }

        let trimmed_len = self.out.trim_end().len();
        self.out.truncate(trimmed_len);
        self.push_newline();

        Ok(())
    }

    fn generate_struct(
        &mut self,
        ty_handle: naga::Handle<naga::Type>,
        members: &[naga::StructMember],
        struct_span: u32,
    ) -> Result<(), Error> {
        // A failed `Layouter::update` leaves every later type unpopulated and
        // `self.layouter[h]` panics on it; refuse so the fallback emitter
        // handles the struct.
        if !self.layouter_complete {
            return Err(Error::Emit(format!(
                "layouter is incomplete (some module types could not be \
                 laid out); cannot safely emit struct '{}'",
                self.type_names[&ty_handle]
            )));
        }
        self.out.push_str("struct ");
        self.out.push_str(&self.type_names[&ty_handle]);
        self.open_brace();

        // Simulate the default WGSL layout and emit `@size` / `@align` only
        // where naga's actual offsets diverge from it; `default_offset` is where
        // the default algorithm would place the next member.
        let member_count = members.len();

        let mut default_offset: u32 = 0;

        for (idx, member) in members.iter().enumerate() {
            self.push_indent();

            let natural = &self.layouter[member.ty];
            let natural_align = natural.alignment;
            let natural_size = natural.size;

            let expected_offset = natural_align.round_up(default_offset);

            // `need_align` is provably never true: naga lays every member at
            // `round_up(running_offset, natural_align)`, exactly
            // `expected_offset`, and a wider explicit `@align` gap is reproduced
            // by `@size` on the prior member, which advances `default_offset` to
            // the forced offset.  Kept as a defensive net that `Err`s to the
            // alternate emitter: a wrong offset is a silent layout miscompile.
            let need_align = member.offset != expected_offset && member.offset > expected_offset;

            if need_align {
                // `trailing_zeros()` is 32 for offset 0, which `need_align`
                // already rules out; `checked_shl` keeps that inspection-proof.
                let a = 1u32
                    .checked_shl(member.offset.trailing_zeros())
                    .unwrap_or(0);
                let align_obj = naga::proc::Alignment::new(a);
                let works = align_obj
                    .map(|ao| ao.round_up(default_offset) == member.offset)
                    .unwrap_or(false);
                if works {
                    self.out.push_str(&format!("@align({a})"));
                    if self.options.beautify {
                        self.out.push(' ');
                    }
                } else {
                    // The gap is not a power-of-two alignment expressible here;
                    // it needs `@size` on the previous member, whose text is
                    // already committed, so refuse and let the fallback emitter
                    // handle it.
                    return Err(Error::Emit(format!(
                        "struct member '{}'[{}] requires padding that can be expressed \
                         only with @size on the previous member; cannot emit safely \
                         (member.offset={}, default_offset={}, computed_align={})",
                        self.type_names[&ty_handle], idx, member.offset, default_offset, a,
                    )));
                }
            }

            let effective_size = if idx + 1 < member_count {
                members[idx + 1].offset - member.offset
            } else {
                struct_span - member.offset
            };

            let actual_start = member.offset;

            // Default-layout footprint: `natural_size` rounded up to the next
            // member's alignment (the struct's for the last member).
            let default_effective = if idx + 1 < member_count {
                let next_align = self.layouter[members[idx + 1].ty].alignment;
                next_align.round_up(actual_start + natural_size) - actual_start
            } else {
                let struct_align = members
                    .iter()
                    .map(|m| self.layouter[m.ty].alignment)
                    .max()
                    .unwrap_or(naga::proc::Alignment::ONE);
                struct_align.round_up(actual_start + natural_size) - actual_start
            };

            // WGSL forbids `@size` on a runtime-sized array (always the last
            // member) and tint rejects it; span rounding (an `@align` on an
            // earlier member) can make the two sizes diverge here, so suppress
            // it - the array's footprint is fixed at binding time.
            let is_runtime_array = matches!(
                self.module.types[member.ty].inner,
                naga::TypeInner::Array {
                    size: naga::ArraySize::Dynamic,
                    ..
                }
            );
            let need_size = !is_runtime_array && effective_size != default_effective;
            if need_size {
                self.out.push_str(&format!("@size({})", effective_size));
                if self.options.beautify {
                    self.out.push(' ');
                }
            }

            default_offset = actual_start
                + if need_size {
                    effective_size
                } else {
                    natural_size
                };

            if let Some(binding) = &member.binding {
                self.out
                    .push_str(&binding_attrs(binding, !self.options.beautify)?);
            }
            if let Some(mangled) = self.member_names.get(&(ty_handle, idx as u32)) {
                self.out.push_str(mangled);
            } else if let Some(name) = &member.name {
                self.out.push_str(name);
            } else {
                self.out.push_str(&format!("m{}", idx));
            }
            self.push_colon();
            self.out.push_str(&self.type_ref(member.ty)?);
            // The last comma is optional; beautify keeps it, compact drops it.
            if idx + 1 < member_count || self.options.beautify {
                self.out.push(',');
            }
            self.push_newline();
        }
        self.close_brace();
        Ok(())
    }

    /// `@diagnostic(...)` attributes owned by the function itself: the filter
    /// chain from `leaf` up to, excluding, the module-level chain.
    fn emit_diagnostic_attrs(
        &mut self,
        leaf: Option<naga::Handle<naga::diagnostic_filter::DiagnosticFilterNode>>,
    ) {
        let module_leaf = self.module.diagnostic_filter_leaf;
        let mut filters = Vec::new();
        let mut next = leaf;
        while let Some(handle) = next {
            if module_leaf == Some(handle) {
                break;
            }
            let node = &self.module.diagnostic_filters[handle];
            filters.push(&node.inner);
            next = node.parent;
        }
        // Definition order is parent -> child, the reverse of the chain walk.
        for filter in filters.iter().rev() {
            self.out.push_str("@diagnostic(");
            self.out.push_str(severity_name(filter.new_severity));
            self.out.push(',');
            self.out
                .push_str(&triggering_rule_name(&filter.triggering_rule));
            self.out
                .push_str(if self.options.beautify { ") " } else { ")" });
        }
    }

    fn generate_function(
        &mut self,
        displayed_name: &str,
        func: &'a naga::Function,
        finfo: &'a naga::valid::FunctionInfo,
        entry_name: Option<&str>,
        module_used_names: &std::collections::HashSet<String>,
        cache_idx: usize,
    ) -> Result<(), Error> {
        let mut ref_counts = std::mem::take(&mut self.ref_count_cache[cache_idx].ref_counts);
        let (deferred_vars, dead_vars) = std::mem::take(&mut self.defer_cache[cache_idx]);
        // Must-bind loads first: `find_for_loop_vars` consults them so its
        // counter-var suppression stays in lockstep with the for-conversion
        // decision (both reject a loop whose update would inline such a load).
        let must_bind_loads = compute_must_bind_loads(func, self.module);
        let for_loop_vars = find_for_loop_vars(func, &must_bind_loads);
        discount_initializer_refs(func, &for_loop_vars, &mut ref_counts);
        let inlineable_calls = find_inlineable_calls(
            &func.body,
            &ref_counts,
            &func.expressions,
            &self.pure_functions,
        );
        let mut ctx = FunctionCtx {
            func,
            exprs: &func.expressions,
            types: ExprTypes::Function(finfo),
            elide_array_ctor: true,
            pinned_root: None,
            argument_names: Vec::with_capacity(func.arguments.len()),
            local_names: Default::default(),
            expr_names: Default::default(),
            ref_counts,
            deferred_vars,
            dead_vars,
            for_loop_vars,
            expr_name_counter: 0,
            module_names: module_used_names,
            local_used_names: std::collections::HashSet::new(),
            inlineable_calls,
            must_bind_loads,
            render_depth_memo: vec![0; func.expressions.len()],
            stashed_call_depth: Default::default(),
            const_hazard_bindings: Vec::new(),
            display_name: displayed_name.to_string(),
        };

        for (i, arg) in func.arguments.iter().enumerate() {
            let name = arg.name.clone().unwrap_or_else(|| format!("a{}", i));
            ctx.local_used_names.insert(name.clone());
            ctx.argument_names.push(name);
        }
        for (h, local) in func.local_variables.iter() {
            let name = local
                .name
                .clone()
                .unwrap_or_else(|| format!("l{}", h.index()));
            if !ctx.dead_vars[h.index()] {
                ctx.local_used_names.insert(name.clone());
            }
            ctx.local_names.insert(h, name);
        }

        let fn_name = entry_name.unwrap_or(displayed_name);

        // Entry points emit theirs before the stage attribute.
        if entry_name.is_none() {
            self.emit_diagnostic_attrs(func.diagnostic_filter_leaf);
        }

        self.out.push_str("fn ");
        self.out.push_str(fn_name);
        self.out.push('(');
        for (i, arg) in func.arguments.iter().enumerate() {
            if i > 0 {
                self.push_separator();
            }
            if let Some(binding) = &arg.binding {
                self.out
                    .push_str(&binding_attrs(binding, !self.options.beautify)?);
            }
            self.out.push_str(&ctx.argument_names[i]);
            self.push_colon();
            self.out.push_str(&self.type_ref(arg.ty)?);
        }
        self.out.push(')');

        if let Some(result) = &func.result {
            self.push_arrow();
            if let Some(binding) = &result.binding {
                self.out
                    .push_str(&binding_attrs(binding, !self.options.beautify)?);
            }
            self.out.push_str(&self.type_ref(result.ty)?);
        }

        self.open_brace();

        for (h, local) in func.local_variables.iter() {
            if ctx.deferred_vars[h.index()]
                || ctx.dead_vars[h.index()]
                || ctx.for_loop_vars[h.index()]
            {
                continue;
            }
            self.push_indent();
            self.out.push_str("var ");
            self.out.push_str(&ctx.local_names[&h]);
            if let Some(init) = local.init {
                let init_expr = &func.expressions[init];
                // `:type` is redundant when the init text carries a concrete type.
                let can_elide_type = match init_expr {
                    naga::Expression::Compose { .. }
                    | naga::Expression::ZeroValue(_)
                    | naga::Expression::Splat { .. } => true,
                    naga::Expression::Literal(lit) => !matches!(
                        lit,
                        naga::Literal::AbstractInt(_) | naga::Literal::AbstractFloat(_)
                    ),
                    _ => false,
                };
                if !can_elide_type {
                    self.push_colon();
                    self.out.push_str(&self.type_ref(local.ty)?);
                }
                self.push_assign();
                // A concrete literal keeps its suffix so the elided type still infers.
                if let (true, naga::Expression::Literal(lit)) = (can_elide_type, init_expr) {
                    self.out.push_str(&super::syntax::literal_to_wgsl(
                        *lit,
                        &self.options.float_precision,
                    ));
                } else {
                    self.out.push_str(&self.emit_expr(init, &mut ctx)?);
                }
            } else {
                // Zero-initialised by WGSL: the shorter of `:type` / `=0i`.
                self.emit_zero_init_tail(local.ty)?;
            }
            self.out.push(';');
            self.push_newline();
        }

        self.generate_block_elide_trailing_return(&func.body, &mut ctx)?;

        // On re-parse naga's `ensure_block_returns` appends an implicit `return;`
        // after a tail `loop` (it never proves a loop non-falling-through), an
        // `InvalidReturnType` in a non-void function; dead-branch / DCE legally
        // produce that shape by stripping the unreachable return after an
        // always-returning loop.  naga inspects only a block's LAST statement,
        // so one synthesised trailing return suppresses the whole injection.
        // Append only when the body provably never falls through (the same
        // judgement that authorised stripping the original), so the return is
        // dead; a body that CAN reach its end (a live return wrongly dropped by
        // some pass) must keep failing validation rather than silently return
        // zero.
        if let Some(result) = &func.result
            && !block_naga_terminates(&func.body)
            && crate::passes::dead_branch::block_definitely_terminates(&func.body)
        {
            let zero = self.zero_value(result.ty)?;
            self.push_indent();
            self.out.push_str("return ");
            self.out.push_str(&zero);
            self.out.push(';');
            self.push_newline();
        }

        self.close_brace();
        Ok(())
    }
}

/// Mirror of naga's `proc::ensure_block_returns`: `true` when naga's front-end
/// would not inject an implicit `return;` at this block's tail.  naga appends
/// one whenever the tail is not a returning terminator, a `loop` included
/// (never proven non-falling-through, even when its body always returns), so
/// unlike `dead_branch::tail_terminates` the `Loop` arm is `false`.
fn block_naga_terminates(block: &naga::Block) -> bool {
    match block.last() {
        Some(
            naga::Statement::Return { .. }
            | naga::Statement::Break
            | naga::Statement::Continue
            | naga::Statement::Kill,
        ) => true,
        Some(naga::Statement::Block(inner)) => block_naga_terminates(inner),
        Some(naga::Statement::If { accept, reject, .. }) => {
            block_naga_terminates(accept) && block_naga_terminates(reject)
        }
        // naga recurses only into non-fall-through cases.
        Some(naga::Statement::Switch { cases, .. }) => cases
            .iter()
            .all(|c| c.fall_through || block_naga_terminates(&c.body)),
        // `Loop`, `Emit`, `Store`, `Call`, `Atomic`, ... and the empty block
        // (`None`) are exactly naga's "append `Return { None }`" arms.
        _ => false,
    }
}

// MARK: Diagnostic directive rendering

fn severity_name(severity: naga::diagnostic_filter::Severity) -> &'static str {
    use naga::diagnostic_filter::Severity as S;
    match severity {
        S::Off => "off",
        S::Info => "info",
        S::Warning => "warning",
        S::Error => "error",
    }
}

fn triggering_rule_name(rule: &naga::diagnostic_filter::FilterableTriggeringRule) -> String {
    use naga::diagnostic_filter::FilterableTriggeringRule as R;
    match rule {
        R::Standard(std_rule) => match std_rule {
            naga::diagnostic_filter::StandardFilterableTriggeringRule::DerivativeUniformity => {
                "derivative_uniformity".to_string()
            }
        },
        R::Unknown(name) => name.to_string(),
        R::User(parts) => format!("{}.{}", parts[0], parts[1]),
    }
}
