//! Module-level emission driver.
//!
//! Top-level entry point [`Generator::generate_module`] orchestrates
//! every module-scope declaration in the order WGSL requires:
//! directives (`enable`, `requires`, `diagnostic`), type aliases,
//! struct declarations, constants, overrides, globals, functions,
//! entry points.  Each section is gated on liveness and on the alias
//! plan computed up front in [`super::core::Generator::new`].

use std::collections::HashMap;

use crate::error::Error;

use super::core::{FunctionCtx, Generator};
use super::syntax::{address_space, binding_attrs, storage_access};

mod call_inline;
mod defer_vars;
mod local_resolve;
mod must_bind;
mod ref_counts;

pub(super) use defer_vars::find_deferrable_vars;
pub(super) use local_resolve::local_var_in_stmts;

use call_inline::{compute_pure_functions, find_inlineable_calls};
use defer_vars::find_for_loop_vars;
use must_bind::compute_must_bind_loads;
use ref_counts::compute_expression_ref_counts;

/// `true` when a constant's init expression already renders its own
/// concrete WGSL type, making a `const NAME: T = ...` annotation
/// redundant: `Compose` / `ZeroValue` / `Splat` constructors spell their
/// type, and a concrete (non-abstract) `Literal` carries a typed suffix.
/// Single source of truth: `generate_constants` omits the annotation in
/// exactly these cases, and `count_type_handle_refs` counts a constant's
/// declared type only when it is NOT one of these, so both must read the
/// same predicate or the alias-savings estimate diverges from the output.
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
    /// Run the entire module emission pipeline end to end.  Populates
    /// the `ref_count_cache`, extracts shared literals, decides which
    /// `enable` directives the module requires, and then emits every
    /// declaration section in spec-mandated order.
    pub(super) fn generate_module(&mut self) -> Result<(), Error> {
        let mut has_prev_section = false;

        // Pre-compute expression ref counts for all functions (regular + entry points)
        // once, so both literal extraction and code generation can reuse them.
        self.ref_count_cache = self
            .module
            .functions
            .iter()
            .map(|(_, f)| compute_expression_ref_counts(f))
            .chain(
                self.module
                    .entry_points
                    .iter()
                    .map(|ep| compute_expression_ref_counts(&ep.function)),
            )
            .collect();

        // Per-function purity, computed once: `find_inlineable_calls` inlines a
        // single-use call result only when its callee is pure (no observable
        // side effect), so an impure call is never relocated past a read of the
        // memory it writes.
        self.pure_functions = compute_pure_functions(self.module);

        // Pre-scan for repeated literals to extract as shared consts.
        self.scan_and_extract_literals();

        // Helper: in beautify mode, insert a blank line between declaration groups.
        macro_rules! section_gap {
            ($self:expr, $has_prev:expr) => {
                if $has_prev {
                    $self.push_newline();
                }
            };
        }

        // Emit `enable` directives for features the module actually uses.
        // Mirrors naga's own `enable`-directive detection.
        let mut needs_f16 = false;
        let mut needs_int16 = false;
        let mut needs_dual_source_blending = false;
        let mut needs_clip_distances = false;
        let mut needs_primitive_index = false;
        let mut needs_draw_index = false;
        let mut needs_mesh_shaders = self.module.uses_mesh_shaders();
        let mut needs_cooperative_matrix = false;
        let mut needs_ray_tracing = false;
        let mut needs_binding_array = false;
        let mut needs_ray_query = false;
        let mut needs_ray_query_vertex_return = false;
        let mut has_acceleration_structure = false;

        let check_binding = |binding: &naga::Binding,
                             needs_dual: &mut bool,
                             needs_clip: &mut bool,
                             needs_prim: &mut bool,
                             needs_draw: &mut bool,
                             needs_mesh: &mut bool,
                             needs_rt: &mut bool| {
            match *binding {
                naga::Binding::Location {
                    blend_src: Some(_), ..
                } => *needs_dual = true,
                naga::Binding::BuiltIn(naga::BuiltIn::ClipDistances) => *needs_clip = true,
                naga::Binding::BuiltIn(naga::BuiltIn::PrimitiveIndex) => *needs_prim = true,
                naga::Binding::BuiltIn(naga::BuiltIn::DrawIndex) => *needs_draw = true,
                naga::Binding::Location {
                    per_primitive: true,
                    ..
                } => *needs_mesh = true,
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
                ) => *needs_rt = true,
                _ => {}
            }
        };

        // Scan types: f16, cooperative matrix, acceleration structure, struct
        // member bindings (clip_distances, mesh, dual_source_blending, etc.).
        for (h, ty) in self.module.types.iter() {
            match ty.inner {
                // Gated on liveness, unlike naga's writer scan of the raw
                // arena: naga's compactor roots `special_types` (e.g. a dead
                // `__frexp_result_f16` and its f16/i16 member scalars survive
                // every DCE), so a raw scan emits `enable f16;` for a shader
                // whose emitted text has no f16 token at all - the enable
                // then drops on re-minify (non-idempotent) and retains a
                // device-feature requirement the output no longer needs.
                // Under-detection stays fail-safe, not silent-invalid:
                // output using f16 without the enable fails the run()
                // re-parse self-check and ships via the naga fallback.
                naga::TypeInner::Scalar(s)
                | naga::TypeInner::Vector { scalar: s, .. }
                | naga::TypeInner::Matrix { scalar: s, .. }
                    if self.live_types.contains(&h) =>
                {
                    needs_f16 |= s == naga::Scalar::F16;
                    needs_int16 |= s == naga::Scalar::I16 || s == naga::Scalar::U16;
                }
                naga::TypeInner::Struct { ref members, .. } => {
                    for binding in members.iter().filter_map(|m| m.binding.as_ref()) {
                        check_binding(
                            binding,
                            &mut needs_dual_source_blending,
                            &mut needs_clip_distances,
                            &mut needs_primitive_index,
                            &mut needs_draw_index,
                            &mut needs_mesh_shaders,
                            &mut needs_ray_tracing,
                        );
                    }
                }
                naga::TypeInner::CooperativeMatrix { .. } => {
                    needs_cooperative_matrix = true;
                }
                // `acceleration_structure` / `ray_query` parse under EITHER
                // `wgpu_ray_query` OR `wgpu_ray_tracing_pipeline` (naga's
                // front lists both as satisfying), so an acceleration
                // structure resolves to an enable only after every pipeline
                // signal (stage, payload, builtin) has been scanned - see the
                // resolution below the stage scan.  The `vertex_return` type
                // flag additionally requires `enable
                // wgpu_ray_query_vertex_return;` to spell.
                naga::TypeInner::AccelerationStructure { vertex_return } => {
                    has_acceleration_structure = true;
                    needs_ray_query_vertex_return |= vertex_return;
                }
                // naga's own writer has no `ray_query` detection (it cannot
                // write `Statement::RayQuery` at all - `unreachable!()` in
                // its statement arm), so this goes beyond mirroring naga:
                // without the enable, emitted `ray_query` locals and
                // `rayQuery*` builtins fail the naga re-parse self-check.
                naga::TypeInner::RayQuery { vertex_return } => {
                    needs_ray_query = true;
                    needs_ray_query_vertex_return |= vertex_return;
                }
                // naga 30 requires `enable wgpu_binding_array;` for a
                // `binding_array<...>` type; mirror its backend's detection.
                naga::TypeInner::BindingArray { .. } => {
                    needs_binding_array = true;
                }
                _ => {}
            }
        }

        // The type scan above (mirroring naga's own writer) misses an f16
        // value that registers no standalone f16 `TypeInner`: a bare `F16`
        // literal or a value-changing cast to f16 that survives folding
        // because it has a runtime operand (e.g. `f32(f16(x) + 2h)` keeps a
        // `2h` literal and an `f16(..)` cast).  Both still emit text that
        // requires `enable f16;`, so also scan every expression arena -
        // const-init, per-function, and entry-point - for them; omitting
        // the directive there yields invalid, naga-rejected output.
        if !needs_f16 {
            let scan = |arena: &naga::Arena<naga::Expression>| {
                arena.iter().any(|(_, e)| {
                    matches!(
                        e,
                        naga::Expression::Literal(naga::Literal::F16(_))
                            | naga::Expression::As {
                                kind: naga::ScalarKind::Float,
                                convert: Some(2),
                                ..
                            }
                    )
                })
            };
            needs_f16 = scan(&self.module.global_expressions)
                || self
                    .module
                    .functions
                    .iter()
                    .any(|(_, f)| scan(&f.expressions))
                || self
                    .module
                    .entry_points
                    .iter()
                    .any(|ep| scan(&ep.function.expressions));
        }

        // Same edge case for `wgpu_int16`: a bare `i16`/`u16` literal (emitted
        // as `i16(N)` / `u16(N)`) or a cast to a 16-bit integer may not register
        // a standalone `TypeInner`, yet the emitted text still needs the enable.
        if !needs_int16 {
            let scan = |arena: &naga::Arena<naga::Expression>| {
                arena.iter().any(|(_, e)| {
                    matches!(
                        e,
                        naga::Expression::Literal(naga::Literal::I16(_) | naga::Literal::U16(_))
                            | naga::Expression::As {
                                kind: naga::ScalarKind::Sint | naga::ScalarKind::Uint,
                                convert: Some(2),
                                ..
                            }
                    )
                })
            };
            needs_int16 = scan(&self.module.global_expressions)
                || self
                    .module
                    .functions
                    .iter()
                    .any(|(_, f)| scan(&f.expressions))
                || self
                    .module
                    .entry_points
                    .iter()
                    .any(|ep| scan(&ep.function.expressions));
        }

        // Scan entry point bindings (arguments and result).
        for ep in &self.module.entry_points {
            if let Some(res) = ep.function.result.as_ref().and_then(|r| r.binding.as_ref()) {
                check_binding(
                    res,
                    &mut needs_dual_source_blending,
                    &mut needs_clip_distances,
                    &mut needs_primitive_index,
                    &mut needs_draw_index,
                    &mut needs_mesh_shaders,
                    &mut needs_ray_tracing,
                );
            }
            for arg_binding in ep
                .function
                .arguments
                .iter()
                .filter_map(|a| a.binding.as_ref())
            {
                check_binding(
                    arg_binding,
                    &mut needs_dual_source_blending,
                    &mut needs_clip_distances,
                    &mut needs_primitive_index,
                    &mut needs_draw_index,
                    &mut needs_mesh_shaders,
                    &mut needs_ray_tracing,
                );
            }
        }

        // RayPayload / IncomingRayPayload address spaces -> ray tracing.
        if self.module.global_variables.iter().any(|gv| {
            matches!(
                gv.1.space,
                naga::AddressSpace::RayPayload | naga::AddressSpace::IncomingRayPayload
            )
        }) {
            needs_ray_tracing = true;
        }

        // Ray tracing shader stages -> ray tracing.
        if self.module.entry_points.iter().any(|ep| {
            matches!(
                ep.stage,
                naga::ShaderStage::RayGeneration
                    | naga::ShaderStage::AnyHit
                    | naga::ShaderStage::ClosestHit
                    | naga::ShaderStage::Miss
            )
        }) {
            needs_ray_tracing = true;
        }

        // Resolve acceleration structures now that every pipeline signal has
        // been scanned: with a pipeline signal present, the pipeline enable
        // already covers the type; otherwise emit `enable wgpu_ray_query;`
        // (naga parses the type under either directive and the IR carries no
        // record of which enable admitted it, so a pipeline-enabled but
        // signal-free input is deliberately rewritten to the query enable).
        if has_acceleration_structure && !needs_ray_tracing {
            needs_ray_query = true;
        }

        // Note: we intentionally do NOT synthesize `enable subgroups;`.
        // This avoids known naga subgroup-directive text-parse limitations
        // and prevents non-profitable growth for tiny subgroup-only shaders.

        // Emit.  Ordering follows naga's own WGSL backend (f16, int16, ...,
        // binding_array, ...) so the directive block matches the naga baseline.
        let any_enable = needs_f16
            || needs_int16
            || needs_dual_source_blending
            || needs_clip_distances
            || needs_mesh_shaders
            || needs_binding_array
            || needs_draw_index
            || needs_primitive_index
            || needs_cooperative_matrix
            || needs_ray_tracing
            || needs_ray_query
            || needs_ray_query_vertex_return;
        if any_enable {
            if needs_f16 {
                self.out.push_str("enable f16;");
                self.push_newline();
            }
            if needs_int16 {
                self.out.push_str("enable wgpu_int16;");
                self.push_newline();
            }
            if needs_dual_source_blending {
                self.out.push_str("enable dual_source_blending;");
                self.push_newline();
            }
            if needs_clip_distances {
                self.out.push_str("enable clip_distances;");
                self.push_newline();
            }
            if needs_mesh_shaders {
                self.out.push_str("enable wgpu_mesh_shader;");
                self.push_newline();
            }
            // naga 30 requires this to parse a `binding_array<...>`; the naga
            // self-check and re-parse depend on it.  It is a naga-only directive
            // tint rejects, so `run` strips it from the final tint-facing output
            // (tint supports binding arrays natively without an enable).
            if needs_binding_array {
                self.out.push_str("enable wgpu_binding_array;");
                self.push_newline();
            }
            if needs_draw_index {
                self.out.push_str("enable draw_index;");
                self.push_newline();
            }
            if needs_primitive_index {
                self.out.push_str("enable primitive_index;");
                self.push_newline();
            }
            if needs_cooperative_matrix {
                self.out.push_str("enable wgpu_cooperative_matrix;");
                self.push_newline();
            }
            if needs_ray_tracing {
                self.out.push_str("enable wgpu_ray_tracing_pipeline;");
                self.push_newline();
            }
            if needs_ray_query {
                self.out.push_str("enable wgpu_ray_query;");
                self.push_newline();
            }
            if needs_ray_query_vertex_return {
                self.out.push_str("enable wgpu_ray_query_vertex_return;");
                self.push_newline();
            }
            has_prev_section = true;
        }

        // Emit module-level `diagnostic(severity, rule);` directives
        // by walking the parent chain from Module::diagnostic_filter_leaf.
        {
            let mut filters = Vec::new();
            let mut next = self.module.diagnostic_filter_leaf;
            while let Some(handle) = next {
                let node = &self.module.diagnostic_filters[handle];
                filters.push(&node.inner);
                next = node.parent;
            }
            // Emit in definition order (parent -> child, i.e. reversed).
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

        // Emit type alias declarations.
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
        // Collect handles of naga's special/predeclared struct types so we
        // don't re-emit them as user-defined struct declarations.  In WGSL
        // these are predeclared types (e.g. `RayDesc`, `RayIntersection`,
        // `__modf_result_f32`, `__atomic_compare_exchange_result`, ...) and
        // declaring them again as a struct makes the validator reject
        // constructor expressions because the user struct and the predeclared
        // type end up with different type-arena handles.
        //
        // Mirrors naga's own `is_builtin_wgsl_struct` test exactly.
        let special_type_handles: std::collections::HashSet<naga::Handle<naga::Type>> = {
            let st = &self.module.special_types;
            // Named singleton special types.
            let mut set: std::collections::HashSet<_> = [
                st.ray_desc,
                st.ray_intersection,
                st.ray_vertex_return,
                st.external_texture_params,
                st.external_texture_transfer_function,
            ]
            .iter()
            .filter_map(|h| *h)
            .collect();
            // All predeclared result types: AtomicCompareExchangeWeakResult,
            // ModfResult, FrexpResult (and any future variants naga adds).
            set.extend(st.predeclared_types.values().copied());
            set
        };
        for (h, ty) in self.module.types.iter() {
            if let naga::TypeInner::Struct { members, span } = &ty.inner {
                if !self.live_types.contains(&h) {
                    continue;
                }
                // Skip predeclared/special struct types - they must not be
                // declared explicitly in WGSL output.
                if special_type_handles.contains(&h) {
                    continue;
                }
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
            self.live_constants.contains(&h)
                && c.name.as_deref().is_some_and(|n| !preamble.contains(n))
        });
        if has_constants {
            section_gap!(self, has_prev_section);
        }
        for (h, c) in self.module.constants.iter() {
            if c.name.is_none() {
                continue;
            }
            if !self.live_constants.contains(&h) {
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
            // Keep the `: T` annotation only when the init text does not
            // already spell its own type (abstract literals, `Constant`
            // refs, arithmetic); see `const_init_has_explicit_type`.
            let init_expr = &self.module.global_expressions[c.init];
            if !const_init_has_explicit_type(init_expr) {
                self.push_colon();
                self.out.push_str(&self.type_ref(c.ty)?);
            }
            self.push_assign();
            let expr = self.emit_global_expr(c.init)?;
            self.out.push_str(&expr);
            self.out.push(';');
            self.push_newline();
            // Register this constant's init handle so that later constants
            // whose sub-expressions share the same handle can emit the short
            // name instead of re-inlining the full expression tree.
            self.expr_to_const.insert(c.init, h);
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
                self.out.push_str(&self.emit_global_expr(init)?);
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
                    // Elide the default `read` access.  Compare against
                    // the resolved name (not the raw bitflag) so empty
                    // / non-LOAD-only flag combinations classify
                    // correctly via `storage_access`'s mapping.
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
                self.out.push_str(&self.emit_global_expr(init)?);
            }
            self.out.push(';');
            self.push_newline();
        }
        if has_globals {
            has_prev_section = true;
        }

        // Emit extracted shared literal consts.
        // Pre-render lines so we release the immutable borrow on extracted_literals
        // before mutably borrowing self for output.
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

        // Pre-build module-scope used_names once, shared across all functions.
        let mut module_used_names = std::collections::HashSet::new();
        module_used_names.extend(self.type_names.values().cloned());
        module_used_names.extend(self.constant_names.iter().cloned());
        module_used_names.extend(self.override_names.iter().cloned());
        module_used_names.extend(self.global_names.iter().cloned());
        module_used_names.extend(self.function_names.iter().cloned());
        module_used_names.extend(self.module.entry_points.iter().map(|ep| ep.name.clone()));
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

            // Emit per-function @diagnostic(...) attributes.
            self.emit_diagnostic_attrs(ep.function.diagnostic_filter_leaf);

            // Use THIS entry point's recorded payload handle, not a scan for the
            // first IncomingRayPayload global: a module with two hit/miss entry
            // points using different payloads would otherwise bind every one to
            // the first payload, silently wiring the wrong variable at runtime.
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

                    // Emit dimension 0.
                    if let Some(handle) = overrides.and_then(|o| o[0]) {
                        self.out.push_str(&self.emit_global_expr(handle)?);
                    } else {
                        self.out.push_str(&ep.workgroup_size[0].to_string());
                    }

                    // Determine whether dimensions 1 and 2 are non-trivial
                    // (override-expression present OR literal != 1).
                    let has_dim1 =
                        overrides.and_then(|o| o[1]).is_some() || ep.workgroup_size[1] != 1;
                    let has_dim2 =
                        overrides.and_then(|o| o[2]).is_some() || ep.workgroup_size[2] != 1;

                    if has_dim1 || has_dim2 {
                        self.push_separator();
                        if let Some(handle) = overrides.and_then(|o| o[1]) {
                            self.out.push_str(&self.emit_global_expr(handle)?);
                        } else {
                            self.out.push_str(&ep.workgroup_size[1].to_string());
                        }
                        if has_dim2 {
                            self.push_separator();
                            if let Some(handle) = overrides.and_then(|o| o[2]) {
                                self.out.push_str(&self.emit_global_expr(handle)?);
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

        // Trim trailing whitespace.
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
        // `self.layouter[h]` panics on the first missing handle, and a
        // failed `Layouter::update` leaves every type after the error
        // unpopulated.  Refuse to emit so the pipeline's fallback path
        // handles the struct instead of crashing here.
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

        // Walk the members while simulating the default WGSL layout
        // algorithm.  Emit @size / @align only when the actual naga
        // offsets diverge from what the default layout would produce.
        let member_count = members.len();

        // `default_offset` tracks where the *default* layout algorithm
        // (no explicit attributes) would place the next member.
        let mut default_offset: u32 = 0;

        for (idx, member) in members.iter().enumerate() {
            self.push_indent();

            let natural = &self.layouter[member.ty];
            let natural_align = natural.alignment;
            let natural_size = natural.size;

            // Where default layout would place this member.
            let expected_offset = natural_align.round_up(default_offset);

            // Does the actual offset require a different alignment?
            //
            // `need_align` is in practice never true: `expected_offset ==
            // member.offset` for every member, because naga lays each one at
            // `round_up(running_offset, natural_align)` - exactly the
            // `expected_offset` computed here - and any wider gap from an
            // explicit `@align` is reproduced by `@size` on the prior member
            // (below), which advances `default_offset` to that forced offset.
            // Kept as a defensive net (it `Err`s to the alternate emitter)
            // rather than deleted: a wrong offset is a silent layout
            // miscompile, so provably-dead code beats an unproven deletion.
            let need_align = member.offset != expected_offset && member.offset > expected_offset;

            if need_align {
                // Find the smallest power-of-2 alignment that, applied to
                // default_offset, yields at least member.offset.
                //
                // Defensive: `member.offset.trailing_zeros()` returns
                // 32 when offset == 0 (panic on `1u32 << 32` in debug,
                // UB on release pre-1.x).  `need_align` above already
                // rules out offset == 0 (`expected_offset == 0` for
                // the first member and `member.offset > expected_offset`
                // demands a strictly larger offset, so we never reach
                // here with offset 0), but `checked_shl` makes the
                // invariant inspection-proof.
                let a = 1u32
                    .checked_shl(member.offset.trailing_zeros())
                    .unwrap_or(0);
                // Verify: round_up(default_offset, a) == member.offset
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
                    // The gap from `default_offset` to `member.offset`
                    // is not a power-of-two alignment expressible on
                    // this member - it would require `@size` on the
                    // previous member, whose text is already committed
                    // to `self.out`.  Refuse rather than emit the wrong
                    // layout; the fallback emitter handles it.
                    return Err(Error::Emit(format!(
                        "struct member '{}'[{}] requires padding that can be expressed \
                         only with @size on the previous member; cannot emit safely \
                         (member.offset={}, default_offset={}, computed_align={})",
                        self.type_names[&ty_handle], idx, member.offset, default_offset, a,
                    )));
                }
            }

            // Effective size occupied by this member (from offset to
            // the next member's offset or struct span).
            let effective_size = if idx + 1 < member_count {
                members[idx + 1].offset - member.offset
            } else {
                struct_span - member.offset
            };

            // Advance the default-layout cursor as if we just placed
            // this member.  Start from the actual offset (which may
            // have been forced by @align above) so subsequent members
            // track correctly.
            let actual_start = member.offset;

            // What the default layout would give as effective member
            // footprint: natural_size, rounded up to the next member's
            // alignment (or struct alignment for last member).
            let default_effective = if idx + 1 < member_count {
                let next_align = self.layouter[members[idx + 1].ty].alignment;
                next_align.round_up(actual_start + natural_size) - actual_start
            } else {
                // Last member: struct span from actual_start using the
                // struct alignment (max of all member alignments).
                let struct_align = members
                    .iter()
                    .map(|m| self.layouter[m.ty].alignment)
                    .max()
                    .unwrap_or(naga::proc::Alignment::ONE);
                struct_align.round_up(actual_start + natural_size) - actual_start
            };

            // A runtime-sized array member (only ever the last member) must not
            // carry an `@size` attribute: WGSL forbids sizing a runtime-sized
            // array and tint rejects it.  The layouter's span rounding can
            // otherwise make `effective_size` diverge from `default_effective`
            // here (e.g. an `@align` on an earlier member rounds the struct
            // span up), so suppress the attribute for such members - a runtime
            // array's footprint is determined at binding time, never here.
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

            // Update the default-offset cursor.  If we emitted @size
            // or @align, subsequent members must track from the actual
            // position to keep computing correct expected offsets.
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
            // The last member's comma is optional; beautify keeps the
            // conventional trailing comma, compact drops it.
            if idx + 1 < member_count || self.options.beautify {
                self.out.push(',');
            }
            self.push_newline();
        }
        self.close_brace();
        Ok(())
    }

    /// Emit `@diagnostic(severity, rule) ` attributes for a function,
    /// walking the filter tree from the given leaf up through parents.
    /// Only emits the filters that belong to the function itself (those
    /// whose parent chain starts at the function's own leaf, stopping
    /// before module-level filters).
    fn emit_diagnostic_attrs(
        &mut self,
        leaf: Option<naga::Handle<naga::diagnostic_filter::DiagnosticFilterNode>>,
    ) {
        let module_leaf = self.module.diagnostic_filter_leaf;
        let mut filters = Vec::new();
        let mut next = leaf;
        while let Some(handle) = next {
            // Stop when we reach the module-level chain (module_leaf or
            // any of its ancestors are module-scoped).
            if module_leaf == Some(handle) {
                break;
            }
            let node = &self.module.diagnostic_filters[handle];
            filters.push(&node.inner);
            next = node.parent;
        }
        // Emit in definition order (parent -> child, i.e. reversed).
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
        let ref_counts = std::mem::take(&mut self.ref_count_cache[cache_idx].ref_counts);
        let (deferred_vars, dead_vars) = find_deferrable_vars(func);
        // Compute the must-bind loads first: `find_for_loop_vars` consults them
        // so its counter-var suppression stays in lockstep with
        // `try_emit_for_loop`'s for-conversion decision (both reject a loop
        // whose update clause would inline a must-bind load).
        let must_bind_loads = compute_must_bind_loads(func, self.module);
        let for_loop_vars = find_for_loop_vars(func, &must_bind_loads);
        let inlineable_calls = find_inlineable_calls(
            &func.body,
            &ref_counts,
            &func.expressions,
            &self.pure_functions,
        );
        let mut ctx = FunctionCtx {
            func,
            info: finfo,
            argument_names: Vec::with_capacity(func.arguments.len()),
            local_names: HashMap::new(),
            expr_names: HashMap::new(),
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
            stashed_call_depth: HashMap::new(),
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

        // For regular (non-entry-point) functions, emit @diagnostic
        // attributes here (entry points handle them before the stage
        // attribute).
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
                // Elide `:type` when the init text carries a concrete type:
                // typed-suffix Literal, Compose, ZeroValue, or Splat.
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
                // When type is elided and init is a concrete Literal, emit
                // with typed suffix so WGSL infers the right type.
                if let (true, naga::Expression::Literal(lit)) = (can_elide_type, init_expr) {
                    self.out.push_str(&super::syntax::literal_to_wgsl(
                        *lit,
                        &self.options.float_precision,
                    ));
                } else {
                    self.out.push_str(&self.emit_expr(init, &mut ctx)?);
                }
            } else {
                // No initializer: WGSL zero-initialises the local.  Pick the
                // shorter of `:type` / `=0i` (a bare scalar counter keeps the
                // cheaper `=0i` form; composites and `bool` keep `:type`).
                self.emit_zero_init_tail(local.ty)?;
            }
            self.out.push(';');
            self.push_newline();
        }

        self.generate_block_elide_trailing_return(&func.body, &mut ctx)?;

        // On re-parse, naga's front-end (`proc::ensure_block_returns`) appends an
        // implicit `return;` (value `None`) after a tail `loop` - it never proves
        // a loop non-falling-through, even one whose body always returns - which
        // is an `InvalidReturnType` in a non-void function. Dead-branch/DCE
        // legally strip the unreachable trailing return after an always-returning
        // loop, producing exactly that shape. naga inspects only a block's LAST
        // statement, so one synthesised trailing return makes the tail a `Return`
        // and suppresses the whole recursive injection (loops nested in if/switch
        // arms included).
        //
        // Append ONLY when the body provably never falls through
        // (`block_definitely_terminates`, the same judgement that authorised
        // stripping the original return) so the synthesised return is provably
        // dead. The guard keeps a body that CAN reach its end - a
        // `Store`/`Call`/breakable-loop tail, reachable only if some pass wrongly
        // dropped a *live* return - failing validation rather than silently
        // returning zero: a fail-safe crash beats a masked miscompile.
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

/// Mirror of naga's `proc::ensure_block_returns`: `true` when naga's WGSL
/// front-end would NOT inject an implicit `return;` at this block's tail.
///
/// naga appends `Statement::Return { value: None }` whenever a block tail is not
/// a returning terminator - notably a `loop` (never proven non-falling-through,
/// even when its body always returns), `Emit`, `Store`, `Call`, ... or an empty
/// block - which is invalid in a non-void function, hence this guard on whether
/// a trailing return must be synthesised.
///
/// Deliberately distinct from `dead_branch::definitely_terminates`, which asks
/// whether control diverts past the enclosing scope and so counts an always-
/// returning loop as terminating; naga's front-end never does, so the `Loop`
/// arm is always `false` here.
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
        // naga recurses only into non-fall-through cases; a fall-through case
        // never triggers an append on its own.
        Some(naga::Statement::Switch { cases, .. }) => cases
            .iter()
            .all(|c| c.fall_through || block_naga_terminates(&c.body)),
        // `Loop`, `Emit`, `Store`, `Call`, `Atomic`, ... and the empty block
        // (`None`) are exactly naga's "append `Return { None }`" arms.
        _ => false,
    }
}

// MARK: Diagnostic directive rendering

/// Map a [`naga::diagnostic_filter::Severity`] to its WGSL keyword.
fn severity_name(severity: naga::diagnostic_filter::Severity) -> &'static str {
    use naga::diagnostic_filter::Severity as S;
    match severity {
        S::Off => "off",
        S::Info => "info",
        S::Warning => "warning",
        S::Error => "error",
    }
}

/// Map a filterable triggering rule to its WGSL token.
/// Render a diagnostic rule to its dotted WGSL form
/// (`derivative_uniformity`, `warn.foo`, etc.) for the
/// `diagnostic(...)` directive.
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
