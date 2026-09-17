//! Module-level emission driver: [`Generator::generate_module`] emits every
//! module-scope declaration in the order WGSL requires (directives, aliases,
//! structs, constants, overrides, globals, functions, entry points), each
//! section gated on liveness and on the alias plan computed up front.

use crate::error::Error;

use super::core::{ExprTypes, FunctionAnalyses, FunctionCtx, FunctionExprInfo, Generator};
use super::syntax::{address_space, binding_attrs, storage_access};

mod call_inline;
mod defer_vars;
mod local_resolve;
mod must_bind;
mod ref_counts;
mod twins;

pub(super) use defer_vars::find_deferrable_vars;
pub(super) use local_resolve::local_var_in_stmts;
pub(super) use twins::Twins;

use crate::analysis::compute_fn_effects;
use crate::handle_set::HandleSet;
/// Renders of one function before its binding decisions are taken as they
/// stand: the first prices by the census, each next by the counts of the
/// one before; two settle every function seen, and the cap holds a bistable
/// pair to the state it was found in.
const RENDER_ROUNDS: usize = 3;

use call_inline::find_inlineable_calls;
use defer_vars::find_for_loop_vars;
use must_bind::compute_must_bind;
use ref_counts::{
    compute_expression_ref_counts, discount_compose_folds, discount_initializer_refs,
};
use twins::structural_twins;

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
    per_vertex: bool,
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
            naga::Binding::Location {
                interpolation: Some(naga::Interpolation::PerVertex),
                ..
            } => self.per_vertex = true,
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
            (self.per_vertex, "enable wgpu_per_vertex;"),
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
        .chain(crate::ir::visit::all_functions(module).flat_map(|f| f.expressions.iter()))
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

/// A declaration the `--preamble` text already carries under `name`: the
/// consumer splices that text ahead of the output, so emitting it again would
/// declare it twice.
fn preamble_owns(preamble: &std::collections::HashSet<String>, name: Option<&str>) -> bool {
    name.is_some_and(|n| preamble.contains(n))
}

impl<'a> Generator<'a> {
    /// The module-wide analyses every function's rendering reads: ref
    /// counts, callee purity, and the literal extraction (a literal renders
    /// as its `const` name from here on).  Once, before any text.  The
    /// census serves the literal count and the bodies' analyses, both of
    /// which a render of these arenas may have left already.
    pub(super) fn prepare(&mut self) {
        if self.analyses.literal_counts.is_none() {
            self.ref_count_cache = crate::ir::visit::all_functions(self.module)
                .map(compute_expression_ref_counts)
                .collect();
        }

        self.fn_effects = compute_fn_effects(self.module);
        self.name_weights = crate::passes::rename::Weights::declared(self.module);

        self.scan_and_extract_literals();
    }

    /// The names a function body's `let`s must dodge.  Preserved names are
    /// in scope because a `let` shadowing one is legal but would leave the
    /// name map unable to account for the extra occurrences.
    /// Function-locals are not: `local_used_names` tracks the ones actually
    /// in scope.
    pub(super) fn module_used_names(&self) -> std::collections::HashSet<String> {
        let mut names: std::collections::HashSet<String> =
            self.emitted_module_names().map(str::to_owned).collect();
        names.extend(self.extracted_literals.values().cloned());
        names
    }

    /// Emit the module: the per-function analyses and literal extraction, then
    /// every declaration section in spec order.
    pub(super) fn generate_module(&mut self) -> Result<(), Error> {
        let mut has_prev_section = false;

        self.prepare();

        // One context for every declaration section: rebuilding it per
        // declaration clones every constant name, quadratic on a module that
        // is mostly constants.  Each constant is named in it as it is emitted,
        // exactly as `expr_to_const` names it for the function sections.
        let mut decl_ctx = self.module_ctx();

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
                self.section_gap(has_prev_section);
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
            self.section_gap(has_prev_section);
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
        self.emit_struct_decls(&preamble, &mut has_prev_section)?;
        self.emit_constant_decls(&preamble, &mut decl_ctx, &mut has_prev_section)?;
        self.emit_override_decls(&preamble, &mut decl_ctx, &mut has_prev_section)?;
        self.emit_global_decls(&preamble, &mut decl_ctx, &mut has_prev_section)?;

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
            self.section_gap(has_prev_section);
            for line in &extracted_lines {
                self.out.push_str(line);
                self.push_newline();
            }
            has_prev_section = true;
        }

        let module_used_names = self.module_used_names();

        for (h, f) in self.module.functions.iter() {
            if preamble_owns(&preamble, f.name.as_deref()) {
                continue;
            }
            self.section_gap(has_prev_section);
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
            if preamble_owns(&preamble, Some(&ep.name)) {
                continue;
            }
            self.section_gap(has_prev_section);

            self.emit_diagnostic_attrs(ep.function.diagnostic_filter_leaf);

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
                naga::ShaderStage::AnyHit
                | naga::ShaderStage::ClosestHit
                | naga::ShaderStage::Miss => {
                    self.out.push_str(match ep.stage {
                        naga::ShaderStage::AnyHit => "@any_hit",
                        naga::ShaderStage::ClosestHit => "@closest_hit",
                        _ => "@miss",
                    });
                    // This entry point's own recorded payload: scanning for the
                    // first `IncomingRayPayload` global would wire every hit /
                    // miss entry point to the same payload.
                    if let Some(h) = ep.incoming_ray_payload {
                        self.out.push_str(if self.options.beautify {
                            " @incoming_payload("
                        } else {
                            "@incoming_payload("
                        });
                        self.out.push_str(&self.global_names[h.index()]);
                        self.out
                            .push_str(if self.options.beautify { ") " } else { ")" });
                    } else {
                        self.out.push(' ');
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
                self.module.functions.len() + i,
            )?;
            self.push_newline();
            has_prev_section = true;
        }

        let trimmed_len = self.out.trim_end().len();
        self.out.truncate(trimmed_len);
        self.push_newline();
        self.type_uses.absorb(&decl_ctx.type_uses);

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
            if let Some(mangled) = self
                .member_names
                .get(ty_handle)
                .and_then(|names| names.get(idx))
            {
                self.out.push_str(mangled);
            } else if let Some(name) = &member.name {
                self.out.push_str(name);
            } else {
                self.out.push_str(&format!("m{}", idx));
            }
            self.push_colon();
            let spelled = self.declare_type(member.ty)?;
            self.out.push_str(&spelled);
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

    /// [`Generator::push_newline`] pushes nothing in compact output, so this
    /// is the beautified layout's only blank-line rule: one ahead of each
    /// section, struct, function and entry point when anything precedes.
    fn section_gap(&mut self, has_prev: bool) {
        if has_prev {
            self.push_newline();
        }
    }

    /// The gap ahead of a section's first item, and the section on record:
    /// called before every item, it acts once.
    fn section_start(&mut self, has_prev_section: &mut bool, first: &mut bool) {
        if std::mem::take(first) {
            self.section_gap(*has_prev_section);
            *has_prev_section = true;
        }
    }

    /// Emit the `struct` declarations: every live, non-predeclared struct the
    /// preamble does not already own.  Each is recorded host-addressable
    /// before the preamble filter, so a preamble-owned struct still reaches
    /// the name map.
    fn emit_struct_decls(
        &mut self,
        preamble: &std::collections::HashSet<String>,
        has_prev_section: &mut bool,
    ) -> Result<(), Error> {
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
                self.map_visible_structs.insert(h);
                if preamble_owns(preamble, ty.name.as_deref()) {
                    continue;
                }
                self.section_gap(*has_prev_section);
                self.generate_struct(h, members, *span)?;
                self.push_newline();
                *has_prev_section = true;
            }
        }

        Ok(())
    }

    /// Emit the named `const` declarations, recording each in
    /// [`Generator::expr_to_const`] and `decl_ctx` as it goes.
    fn emit_constant_decls(
        &mut self,
        preamble: &std::collections::HashSet<String>,
        decl_ctx: &mut FunctionCtx<'a, 'a>,
        has_prev_section: &mut bool,
    ) -> Result<(), Error> {
        let mut first = true;
        for (h, c) in self.module.constants.iter() {
            if c.name.is_none()
                || !self.live_constants.contains(h)
                || preamble_owns(preamble, c.name.as_deref())
            {
                continue;
            }
            self.section_start(has_prev_section, &mut first);
            self.out.push_str("const ");
            self.out.push_str(&self.constant_names[h.index()]);
            let init_expr = &self.module.global_expressions[c.init];
            let self_typed = const_init_has_explicit_type(init_expr);
            if !self_typed {
                self.push_colon();
                let spelled = self.declare_type(c.ty)?;
                self.out.push_str(&spelled);
            }
            self.push_assign();
            let expr = self.emit_global_expr_in(c.init, self_typed, decl_ctx)?;
            self.out.push_str(&expr);
            self.out.push(';');
            self.push_newline();
            self.expr_to_const.insert(c.init, h);
            decl_ctx
                .expr_names
                .insert(c.init, self.constant_names[h.index()].clone());
        }

        Ok(())
    }

    fn emit_override_decls(
        &mut self,
        preamble: &std::collections::HashSet<String>,
        decl_ctx: &mut FunctionCtx<'a, 'a>,
        has_prev_section: &mut bool,
    ) -> Result<(), Error> {
        let mut first = true;
        for (h, ov) in self.module.overrides.iter() {
            if preamble_owns(preamble, ov.name.as_deref()) {
                continue;
            }
            self.section_start(has_prev_section, &mut first);
            if let Some(id) = ov.id {
                self.out.push_str("@id(");
                self.out.push_str(&id.to_string());
                self.out
                    .push_str(if self.options.beautify { ") " } else { ")" });
            }
            self.out.push_str("override ");
            self.out.push_str(&self.override_names[h.index()]);
            self.push_colon();
            let spelled = self.declare_type(ov.ty)?;
            self.out.push_str(&spelled);
            if let Some(init) = ov.init {
                self.push_assign();
                let text = self.emit_global_expr_in(init, false, decl_ctx)?;
                self.out.push_str(&text);
            }
            self.out.push(';');
            self.push_newline();
        }

        Ok(())
    }

    fn emit_global_decls(
        &mut self,
        preamble: &std::collections::HashSet<String>,
        decl_ctx: &mut FunctionCtx<'a, 'a>,
        has_prev_section: &mut bool,
    ) -> Result<(), Error> {
        let mut first = true;
        for (h, g) in self.module.global_variables.iter() {
            if preamble_owns(preamble, g.name.as_deref()) {
                continue;
            }
            self.section_start(has_prev_section, &mut first);
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
            let spelled = self.declare_type(g.ty)?;
            self.out.push_str(&spelled);
            if let Some(init) = g.init {
                self.push_assign();
                let text = self.emit_global_expr_in(init, false, decl_ctx)?;
                self.out.push_str(&text);
            }
            self.out.push(';');
            self.push_newline();
        }
        Ok(())
    }

    /// The analyses `func`'s context is built from: the ones a render of
    /// these arenas left ([`super::core::Analyses`]), else built over the
    /// prepared caches (taken, so each function is built once).
    fn function_analyses(
        &mut self,
        func: &'a naga::Function,
        finfo: &'a naga::valid::FunctionInfo,
        cache_idx: usize,
    ) -> FunctionAnalyses {
        if let Some(built) = self.take_function_analyses(cache_idx) {
            return built;
        }
        // A body the prepared census does not cover counts its own.
        let FunctionExprInfo {
            mut ref_counts,
            live,
            mut paren_uses,
        } = self
            .ref_count_cache
            .get_mut(cache_idx)
            .map(std::mem::take)
            .unwrap_or_else(|| compute_expression_ref_counts(func));
        // A value is priced by the uses of all its spellings, so the loop
        // model below sees it bound where the byte rule will bind it.
        let twins = structural_twins(
            func,
            &self.module.types,
            finfo,
            &mut ref_counts,
            &mut paren_uses,
        );
        let must_bind = compute_must_bind(func, self.module, &self.fn_effects, &ref_counts);
        discount_compose_folds(
            func,
            finfo,
            &self.module.types,
            &must_bind,
            &live,
            &twins,
            &mut ref_counts,
        );
        let for_loop_vars = find_for_loop_vars(func, &must_bind, &self.analyses.defer[cache_idx].0);
        discount_initializer_refs(func, &for_loop_vars, &mut ref_counts);
        let inlineable_calls =
            find_inlineable_calls(&func.body, &ref_counts, &func.expressions, &self.fn_effects);
        FunctionAnalyses {
            ref_counts,
            paren_uses,
            twins,
            must_bind,
            for_loop_vars,
            inlineable_calls,
        }
    }

    /// The context one function body renders in: its analyses
    /// ([`Self::function_analyses`]) and its argument and local names
    /// claimed.  `cache_idx` indexes `ref_count_cache` / `Analyses::defer`:
    /// `module.functions` order, then the entry points.
    pub(super) fn function_ctx<'m>(
        &mut self,
        displayed_name: &str,
        func: &'a naga::Function,
        finfo: &'a naga::valid::FunctionInfo,
        module_used_names: &'m std::collections::HashSet<String>,
        cache_idx: usize,
    ) -> FunctionCtx<'a, 'm> {
        let FunctionAnalyses {
            ref_counts,
            paren_uses,
            twins,
            must_bind,
            for_loop_vars,
            inlineable_calls,
        } = self.function_analyses(func, finfo, cache_idx);
        let (deferred_vars, dead_vars) = self.analyses.defer[cache_idx].clone();
        let typed_pointer_arg_locals =
            typed_pointer_arg_locals(func, &self.module.types, &self.type_names);
        let mut ctx = FunctionCtx {
            func,
            exprs: &func.expressions,
            types: ExprTypes::Function(finfo),
            elide_array_ctor: true,
            pinned_root: None,
            argument_names: Vec::with_capacity(func.arguments.len()),
            local_names: Default::default(),
            expr_names: Default::default(),
            twins,
            ref_counts,
            paren_uses,
            deferred_vars,
            dead_vars,
            typed_pointer_arg_locals,
            for_loop_vars,
            expr_name_counter: 0,
            drawn_expr_name: None,
            module_names: module_used_names,
            local_used_names: std::collections::HashSet::new(),
            inlineable_calls,
            must_bind,
            render_depth_memo: vec![0; func.expressions.len()],
            stashed_call_depth: Default::default(),
            render_counts: vec![0; func.expressions.len()],
            paren_counts: vec![0; func.expressions.len()],
            count_journal: Vec::new(),
            type_uses: super::core::TypeUses::sized(self.module.types.len()),
            measured: None,
            decisions: Vec::new(),
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
        ctx
    }

    /// `body` indexes the per-function caches and the name weights.
    fn generate_function(
        &mut self,
        displayed_name: &str,
        func: &'a naga::Function,
        finfo: &'a naga::valid::FunctionInfo,
        entry_name: Option<&str>,
        module_used_names: &std::collections::HashSet<String>,
        body: crate::passes::rename::Body,
    ) -> Result<(), Error> {
        let mut ctx = self.function_ctx(displayed_name, func, finfo, module_used_names, body);
        let fn_name = entry_name.unwrap_or(displayed_name);
        let built = self.generate_function_in(fn_name, func, entry_name.is_some(), &mut ctx)?;
        self.keep_function_analyses(body, built);
        for (h, expr) in func.expressions.iter() {
            let count = usize::try_from(ctx.render_counts[h.index()]).unwrap_or(0);
            self.name_weights.reference(body, expr, count);
        }
        self.type_uses.absorb(&ctx.type_uses);
        Ok(())
    }

    /// The declaration of `func` as `fn_name`, from a context in the state
    /// [`Self::function_ctx`] built it: nothing bound yet.  The text then
    /// judges its own binding decisions: work the bytes inlined into a loop
    /// (`loop_sunk_work` over the names given) is pinned, and a `let` the
    /// text's own counts would decide the other way is repriced by them (a
    /// consumer the rule inlines renders its operand at each of its uses,
    /// which the census counted once; a twin's counts are its first
    /// spelling's, `counts_by_value`), either rendering the function again
    /// from the context as built; [`RENDER_ROUNDS`] caps a bistable pair at
    /// a text that is valid but a byte or two off.  Hands back the context
    /// as passed, the rendered one in `ctx`.
    pub(super) fn generate_function_in<'m>(
        &mut self,
        fn_name: &str,
        func: &'a naga::Function,
        is_entry_point: bool,
        ctx: &mut FunctionCtx<'a, 'm>,
    ) -> Result<FunctionCtx<'a, 'm>, Error> {
        let start = self.out.len();
        let has_loop = must_bind::has_loop(func);
        // What a further round overwrites in `ctx`, as passed: `must_bind`
        // and `measured`.
        let mut passed = None;
        for round in 1.. {
            let mut attempt = ctx.clone();
            self.render_function(fn_name, func, is_entry_point, &mut attempt)?;
            let mut sunk = HandleSet::default();
            if has_loop {
                must_bind::loop_sunk_work(
                    func,
                    self.module,
                    &must_bind::Bindings::Rendered {
                        names: &attempt.expr_names,
                        stashed: &attempt.inlineable_calls,
                        twins: &attempt.twins,
                    },
                    &mut sunk,
                );
            }
            let beautify = self.options.beautify;
            let (render_counts, paren_counts) = attempt.counts_by_value();
            let settled = sunk.is_empty()
                && attempt.decisions.iter().all(|d| {
                    let i = d.handle.index();
                    let refs = usize::try_from(render_counts[i]).unwrap_or(0);
                    let parens = usize::try_from(paren_counts[i]).unwrap_or(0);
                    super::stmt_emit::binding_pays(refs, parens, d.len, d.name, beautify) == d.bound
                });
            if settled || round == RENDER_ROUNDS {
                let mut as_passed = std::mem::replace(ctx, attempt);
                if let Some((must_bind, measured)) = passed {
                    as_passed.must_bind = must_bind;
                    as_passed.measured = measured;
                }
                return Ok(as_passed);
            }
            self.out.truncate(start);
            passed.get_or_insert_with(|| (ctx.must_bind.clone(), ctx.measured.take()));
            ctx.must_bind.extend(sunk.iter().copied());
            ctx.measured = Some((render_counts, paren_counts));
        }
        unreachable!("the round cap returns")
    }

    fn render_function(
        &mut self,
        fn_name: &str,
        func: &'a naga::Function,
        is_entry_point: bool,
        ctx: &mut FunctionCtx<'a, '_>,
    ) -> Result<(), Error> {
        // Entry points emit theirs before the stage attribute.
        if !is_entry_point {
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
            let spelled = self.spell_type(arg.ty, ctx)?;
            self.out.push_str(&spelled);
        }
        self.out.push(')');

        if let Some(result) = &func.result {
            self.push_arrow();
            if let Some(binding) = &result.binding {
                self.out
                    .push_str(&binding_attrs(binding, !self.options.beautify)?);
            }
            let spelled = self.spell_type(result.ty, ctx)?;
            self.out.push_str(&spelled);
        }

        self.open_brace();

        for (h, local) in func.local_variables.iter() {
            if ctx.deferred_vars[h.index()]
                || ctx.dead_vars[h.index()]
                || ctx.for_loop_vars[h.index()].is_some()
            {
                continue;
            }
            self.push_indent();
            self.out.push_str("var ");
            self.out.push_str(&ctx.local_names[&h]);
            if let Some(init) = local.init {
                let init_expr = &func.expressions[init];
                // `:type` is redundant when the init text carries a concrete
                // type: a constructor, a suffixed literal, or the name of a
                // constant declared with one (an abstract-typed `const K=5;`
                // would leave the `var` abstract).
                let can_elide_type = match init_expr {
                    naga::Expression::Compose { .. }
                    | naga::Expression::ZeroValue(_)
                    | naga::Expression::Splat { .. } => true,
                    naga::Expression::Literal(lit) => !matches!(
                        lit,
                        naga::Literal::AbstractInt(_) | naga::Literal::AbstractFloat(_)
                    ),
                    naga::Expression::Constant(c) => {
                        let constant = &self.module.constants[*c];
                        constant.name.is_some()
                            && !self.module.types[constant.ty]
                                .inner
                                .scalar()
                                .is_some_and(|s| s.is_abstract())
                    }
                    _ => false,
                };
                if !can_elide_type || ctx.needs_declared_type(h, init) {
                    self.push_colon();
                    let spelled = self.spell_type(local.ty, ctx)?;
                    self.out.push_str(&spelled);
                }
                self.push_assign();
                // A concrete literal keeps its suffix so the elided type still infers.
                if let (true, naga::Expression::Literal(lit)) = (can_elide_type, init_expr) {
                    self.out.push_str(&super::syntax::literal_to_wgsl(
                        *lit,
                        &self.options.float_precision,
                    ));
                } else {
                    self.push_expr(init, ctx)?;
                }
            } else {
                // Zero-initialised by WGSL: the shorter of `:type` / `=0i`.
                self.emit_zero_init_tail(local.ty, ctx)?;
            }
            self.out.push(';');
            self.push_newline();
        }

        self.generate_block_elide_trailing_return(&func.body, ctx)?;

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
            let zero = self.zero_value(result.ty, ctx)?;
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

/// `FunctionCtx::typed_pointer_arg_locals`; `type_names` holds every type
/// spelled by a name (struct or minted alias).
fn typed_pointer_arg_locals(
    func: &naga::Function,
    types: &naga::UniqueArena<naga::Type>,
    type_names: &crate::handle_set::HandleMap<naga::Type, String>,
) -> Vec<bool> {
    let mut out = vec![false; func.local_variables.len()];
    crate::ir::visit::for_each_statement(&func.body, &mut |stmt| {
        if let naga::Statement::Call { arguments, .. } = stmt {
            for &arg in arguments {
                let naga::Expression::LocalVariable(lh) = func.expressions[arg] else {
                    continue;
                };
                let ty = func.local_variables[lh].ty;
                if type_names.contains_key(ty)
                    && matches!(
                        types[ty].inner,
                        naga::TypeInner::Matrix { .. } | naga::TypeInner::Array { .. }
                    )
                {
                    out[lh.index()] = true;
                }
            }
        }
    });
    out
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
