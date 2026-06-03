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

use super::core::{FunctionCtx, FunctionExprInfo, Generator};
use super::syntax::{address_space, binding_attrs, storage_access};

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
        let mut needs_dual_source_blending = false;
        let mut needs_clip_distances = false;
        let mut needs_primitive_index = false;
        let mut needs_draw_index = false;
        let mut needs_mesh_shaders = self.module.uses_mesh_shaders();
        let mut needs_cooperative_matrix = false;
        let mut needs_ray_tracing = false;

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
                naga::Binding::BuiltIn(naga::BuiltIn::ClipDistance) => *needs_clip = true,
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
        for (_, ty) in self.module.types.iter() {
            match ty.inner {
                naga::TypeInner::Scalar(s)
                | naga::TypeInner::Vector { scalar: s, .. }
                | naga::TypeInner::Matrix { scalar: s, .. } => {
                    needs_f16 |= s == naga::Scalar::F16;
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
                naga::TypeInner::AccelerationStructure { .. } => {
                    needs_ray_tracing = true;
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

        // Note: we intentionally do NOT synthesize `enable subgroups;`.
        // This avoids known naga subgroup-directive text-parse limitations
        // and prevents non-profitable growth for tiny subgroup-only shaders.

        // Emit.
        let any_enable = needs_f16
            || needs_dual_source_blending
            || needs_clip_distances
            || needs_mesh_shaders
            || needs_draw_index
            || needs_primitive_index
            || needs_cooperative_matrix
            || needs_ray_tracing;
        if any_enable {
            if needs_f16 {
                self.out.push_str("enable f16;");
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
                self.out.push_str(") ");
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

            let incoming_payload_name = self.module.global_variables.iter().find_map(|(h, g)| {
                if matches!(g.space, naga::AddressSpace::IncomingRayPayload) {
                    Some(self.global_names[h.index()].clone())
                } else {
                    None
                }
            });

            match ep.stage {
                naga::ShaderStage::Vertex => self.out.push_str("@vertex "),
                naga::ShaderStage::Fragment => self.out.push_str("@fragment "),
                naga::ShaderStage::Compute => {
                    self.out.push_str("@compute @workgroup_size(");

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
                    self.out.push_str(") ");
                }
                naga::ShaderStage::RayGeneration => self.out.push_str("@ray_generation "),
                naga::ShaderStage::AnyHit => {
                    self.out.push_str("@any_hit ");
                    if let Some(name) = &incoming_payload_name {
                        self.out.push_str("@incoming_payload(");
                        self.out.push_str(name);
                        self.out.push_str(") ");
                    }
                }
                naga::ShaderStage::ClosestHit => {
                    self.out.push_str("@closest_hit ");
                    if let Some(name) = &incoming_payload_name {
                        self.out.push_str("@incoming_payload(");
                        self.out.push_str(name);
                        self.out.push_str(") ");
                    }
                }
                naga::ShaderStage::Miss => {
                    self.out.push_str("@miss ");
                    if let Some(name) = &incoming_payload_name {
                        self.out.push_str("@incoming_payload(");
                        self.out.push_str(name);
                        self.out.push_str(") ");
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
                    self.out.push_str(&format!("@align({a}) "));
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

            let need_size = effective_size != default_effective;
            if need_size {
                self.out.push_str(&format!("@size({}) ", effective_size));
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
                self.out.push_str(&binding_attrs(binding)?);
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
            self.out.push(',');
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
            self.out.push_str(") ");
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
                self.out.push_str(&binding_attrs(binding)?);
            }
            self.out.push_str(&ctx.argument_names[i]);
            self.push_colon();
            self.out.push_str(&self.type_ref(arg.ty)?);
        }
        self.out.push(')');

        if let Some(result) = &func.result {
            self.push_arrow();
            if let Some(binding) = &result.binding {
                self.out.push_str(&binding_attrs(binding)?);
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
                self.push_colon();
                self.out.push_str(&self.type_ref(local.ty)?);
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

// MARK: Expression ref-count analysis

/// Count how many times each expression handle is referenced by
/// other *live* expressions (those in `Emit` ranges) and by
/// statements in the function.  Dead expressions are excluded so
/// they never inflate reference counts.
///
/// Returns both the per-handle reference count vector and the
/// `live` bitmap in a single [`FunctionExprInfo`].  Both are
/// consumed downstream (`generate_function` reads `ref_counts`,
/// `literal_extract` reads both); producing them in one walk
/// avoids a second body traversal.
pub(super) fn compute_expression_ref_counts(func: &naga::Function) -> FunctionExprInfo {
    let len = func.expressions.len();
    let mut counts: Vec<usize> = vec![0; len];

    // Collect handles that appear in Emit ranges (live emitted expressions).
    let mut live = vec![false; len];
    collect_emitted_handles(&func.body, &mut live);

    // Count references only from live expressions.
    for (h, expr) in func.expressions.iter() {
        if live[h.index()] {
            count_expr_children(expr, &mut counts);
        }
    }

    // Count references from statements.
    count_block_refs(&func.body, &mut counts);

    FunctionExprInfo {
        ref_counts: counts,
        live,
    }
}

/// Mark every handle that appears inside an `Emit` range of `block`
/// (recursively across control flow).  Emission-range membership is
/// the authoritative liveness signal for literal extraction and
/// expression ref counting.
pub(super) fn collect_emitted_handles(block: &naga::Block, live: &mut Vec<bool>) {
    for stmt in block {
        match stmt {
            naga::Statement::Emit(range) => {
                for h in range.clone() {
                    live[h.index()] = true;
                }
            }
            naga::Statement::Block(inner) => collect_emitted_handles(inner, live),
            naga::Statement::If { accept, reject, .. } => {
                collect_emitted_handles(accept, live);
                collect_emitted_handles(reject, live);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    collect_emitted_handles(&case.body, live);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                collect_emitted_handles(body, live);
                collect_emitted_handles(continuing, live);
            }
            _ => {}
        }
    }
}

/// Increment `counts[h]` by one.  The maximum ref count is bounded by
/// the total expression-reference count in the function, which fits
/// comfortably in `usize` for any realistic shader, so a checked add
/// is unwarranted - on overflow the program is already pathological
/// and `usize::MAX` writes would have been the least of our worries.
fn bump(counts: &mut [usize], h: naga::Handle<naga::Expression>) {
    counts[h.index()] += 1;
}

/// Generator-local alias for the shared exhaustive child walker in
/// [`crate::passes::expr_util::visit_expression_children`].  Kept as a
/// thin `pub(super)` indirection so the generator's call sites
/// don't take a direct dependency on the passes layer.
pub(super) fn visit_expr_children(
    expr: &naga::Expression,
    f: impl FnMut(naga::Handle<naga::Expression>),
) {
    crate::passes::expr_util::visit_expression_children(expr, f);
}

/// Shortcut helper that bumps `counts` for every child handle of
/// `expr`.  Used by [`compute_expression_ref_counts`] in the
/// arena-traversal loop.
fn count_expr_children(expr: &naga::Expression, counts: &mut [usize]) {
    visit_expr_children(expr, |h| bump(counts, h));
}

/// Walk `block` and increment `counts[h]` for every expression
/// handle referenced from a statement operand, recursing into nested
/// blocks.  Ensures statement-level uses contribute to liveness in
/// the same way expression-level uses do.
fn count_block_refs(block: &naga::Block, counts: &mut Vec<usize>) {
    for stmt in block {
        match stmt {
            naga::Statement::Emit(_) => {}
            naga::Statement::Block(inner) => count_block_refs(inner, counts),
            naga::Statement::If {
                condition,
                accept,
                reject,
            } => {
                bump(counts, *condition);
                count_block_refs(accept, counts);
                count_block_refs(reject, counts);
            }
            naga::Statement::Switch { selector, cases } => {
                bump(counts, *selector);
                for case in cases {
                    count_block_refs(&case.body, counts);
                }
            }
            naga::Statement::Loop {
                body,
                continuing,
                break_if,
            } => {
                count_block_refs(body, counts);
                count_block_refs(continuing, counts);
                if let Some(h) = break_if {
                    bump(counts, *h);
                }
            }
            naga::Statement::Return { value: Some(h) } => {
                bump(counts, *h);
            }
            naga::Statement::Store { pointer, value } => {
                bump(counts, *pointer);
                bump(counts, *value);
            }
            naga::Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                bump(counts, *image);
                bump(counts, *coordinate);
                if let Some(i) = array_index {
                    bump(counts, *i);
                }
                bump(counts, *value);
            }
            naga::Statement::Atomic {
                pointer,
                fun,
                value,
                ..
            } => {
                bump(counts, *pointer);
                bump(counts, *value);
                crate::passes::expr_util::visit_atomic_function_handles(fun, &mut |h| {
                    bump(counts, h)
                });
            }
            naga::Statement::ImageAtomic {
                image,
                coordinate,
                array_index,
                fun,
                value,
            } => {
                bump(counts, *image);
                bump(counts, *coordinate);
                if let Some(i) = array_index {
                    bump(counts, *i);
                }
                bump(counts, *value);
                crate::passes::expr_util::visit_atomic_function_handles(fun, &mut |h| {
                    bump(counts, h)
                });
            }
            naga::Statement::WorkGroupUniformLoad { pointer, .. } => {
                bump(counts, *pointer);
            }
            naga::Statement::Call { arguments, .. } => {
                for a in arguments {
                    bump(counts, *a);
                }
            }
            naga::Statement::RayQuery { query, fun } => {
                bump(counts, *query);
                match fun {
                    naga::RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } => {
                        bump(counts, *acceleration_structure);
                        bump(counts, *descriptor);
                    }
                    naga::RayQueryFunction::Proceed { .. } => {}
                    naga::RayQueryFunction::GenerateIntersection { hit_t } => {
                        bump(counts, *hit_t);
                    }
                    naga::RayQueryFunction::ConfirmIntersection
                    | naga::RayQueryFunction::Terminate => {}
                }
            }
            naga::Statement::SubgroupBallot {
                predicate: Some(h), ..
            } => {
                bump(counts, *h);
            }
            naga::Statement::SubgroupGather { mode, argument, .. } => {
                bump(counts, *argument);
                match mode {
                    naga::GatherMode::BroadcastFirst | naga::GatherMode::QuadSwap(_) => {}
                    naga::GatherMode::Broadcast(h)
                    | naga::GatherMode::Shuffle(h)
                    | naga::GatherMode::ShuffleDown(h)
                    | naga::GatherMode::ShuffleUp(h)
                    | naga::GatherMode::ShuffleXor(h)
                    | naga::GatherMode::QuadBroadcast(h) => {
                        bump(counts, *h);
                    }
                }
            }
            naga::Statement::SubgroupCollectiveOperation { argument, .. } => {
                bump(counts, *argument);
            }
            naga::Statement::RayPipelineFunction(fun) => match fun {
                naga::RayPipelineFunction::TraceRay {
                    acceleration_structure,
                    descriptor,
                    payload,
                } => {
                    bump(counts, *acceleration_structure);
                    bump(counts, *descriptor);
                    bump(counts, *payload);
                }
            },
            naga::Statement::CooperativeStore { target, data } => {
                bump(counts, *target);
                bump(counts, data.pointer);
                bump(counts, data.stride);
            }
            _ => {}
        }
    }
}

// MARK: Local reference resolution

/// Walk a chain of `AccessIndex` / `Access` / `LocalVariable`
/// expressions and return the root local when the chain ultimately
/// resolves to one, or `None` otherwise.
fn resolve_local_var(
    expr: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> Option<naga::Handle<naga::LocalVariable>> {
    match &expressions[expr] {
        naga::Expression::LocalVariable(lh) => Some(*lh),
        naga::Expression::AccessIndex { base, .. } | naga::Expression::Access { base, .. } => {
            resolve_local_var(*base, expressions)
        }
        _ => None,
    }
}

/// Check if `local` is referenced (read *or* written) in any of `stmts`.
/// Scans recursively into nested blocks.  Used in the deferred-var look-ahead
/// safety check: a deferred-var Store + Loop may only be absorbed into a
/// `for(var x=init;...)` when `x` is not referenced after the Loop,
/// because the for-init scopes `x` inside the loop body.
/// `true` when any statement in `stmts` references the local `lh`,
/// either directly, through an access chain, or inside a nested block.
pub(super) fn local_var_in_stmts(
    stmts: &[&naga::Statement],
    local: naga::Handle<naga::LocalVariable>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    stmts
        .iter()
        .any(|s| local_var_in_stmt(s, local, expressions))
}

/// Block-scoped variant of [`local_var_in_stmts`].
fn local_var_in_block(
    block: &naga::Block,
    local: naga::Handle<naga::LocalVariable>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    block
        .iter()
        .any(|s| local_var_in_stmt(s, local, expressions))
}

/// Single-statement variant of [`local_var_in_stmts`].  Recurses
/// into nested control-flow blocks and into expression operands via
/// [`resolve_local_var`].
fn local_var_in_stmt(
    stmt: &naga::Statement,
    local: naga::Handle<naga::LocalVariable>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    use naga::Expression as E;
    use naga::Statement as S;
    match stmt {
        S::Emit(range) => range.clone().any(|h| {
            matches!(&expressions[h], E::Load { pointer }
                    if resolve_local_var(*pointer, expressions) == Some(local))
        }),
        // These statement-level arms resolve only the bare POINTER/place operand
        // (a write target or taken address).  A local used BY VALUE - an atomic
        // `value`/`compare`, a call/image/ray operand - is a `Load(LocalVariable)`
        // expression, which `resolve_local_var` does not see through (it returns
        // `None` for a `Load`); such reads are already caught by the `Emit` arm
        // above.  So intentionally omitting value/compare operands here cannot
        // miss a reference - the same place/value split the sibling
        // `collect_block_local_refs` relies on.
        S::Store { pointer, .. }
        | S::WorkGroupUniformLoad { pointer, .. }
        | S::Atomic { pointer, .. } => resolve_local_var(*pointer, expressions) == Some(local),
        S::ImageStore {
            image,
            coordinate,
            array_index,
            value,
        } => [Some(*image), Some(*coordinate), *array_index, Some(*value)]
            .into_iter()
            .flatten()
            .any(|e| resolve_local_var(e, expressions) == Some(local)),
        S::Call { arguments, .. } => arguments
            .iter()
            .any(|&a| resolve_local_var(a, expressions) == Some(local)),
        S::Return { value: Some(v) } => resolve_local_var(*v, expressions) == Some(local),
        S::If { accept, reject, .. } => {
            local_var_in_block(accept, local, expressions)
                || local_var_in_block(reject, local, expressions)
        }
        S::Switch { cases, .. } => cases
            .iter()
            .any(|c| local_var_in_block(&c.body, local, expressions)),
        S::Loop {
            body, continuing, ..
        } => {
            local_var_in_block(body, local, expressions)
                || local_var_in_block(continuing, local, expressions)
        }
        S::Block(inner) => local_var_in_block(inner, local, expressions),
        // Pointer/operand-bearing statements that the sibling
        // `collect_block_local_refs` also handles.  Without these arms a
        // post-loop reference to the absorbed induction local through one of
        // them would be missed, leaving the for-init-scoped local referenced
        // out of scope (invalid WGSL).  The genuinely-reachable case for a
        // function-local is `CooperativeStore`'s `data.pointer`.
        S::ImageAtomic {
            image,
            coordinate,
            array_index,
            value,
            ..
        } => [Some(*image), Some(*coordinate), *array_index, Some(*value)]
            .into_iter()
            .flatten()
            .any(|e| resolve_local_var(e, expressions) == Some(local)),
        S::SubgroupBallot {
            predicate: Some(p), ..
        } => resolve_local_var(*p, expressions) == Some(local),
        S::SubgroupGather { mode, argument, .. } => {
            let hits = |e| resolve_local_var(e, expressions) == Some(local);
            hits(*argument)
                || match mode {
                    naga::GatherMode::Broadcast(h)
                    | naga::GatherMode::Shuffle(h)
                    | naga::GatherMode::ShuffleDown(h)
                    | naga::GatherMode::ShuffleUp(h)
                    | naga::GatherMode::ShuffleXor(h)
                    | naga::GatherMode::QuadBroadcast(h) => hits(*h),
                    naga::GatherMode::BroadcastFirst | naga::GatherMode::QuadSwap(_) => false,
                }
        }
        S::SubgroupCollectiveOperation { argument, .. } => {
            resolve_local_var(*argument, expressions) == Some(local)
        }
        S::RayPipelineFunction(naga::RayPipelineFunction::TraceRay {
            acceleration_structure,
            descriptor,
            payload,
        }) => [*acceleration_structure, *descriptor, *payload]
            .into_iter()
            .any(|e| resolve_local_var(e, expressions) == Some(local)),
        S::CooperativeStore { target, data } => [*target, data.pointer, data.stride]
            .into_iter()
            .any(|e| resolve_local_var(e, expressions) == Some(local)),
        S::RayQuery { query, fun } => {
            let hits = |e| resolve_local_var(e, expressions) == Some(local);
            hits(*query)
                || match fun {
                    naga::RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } => hits(*acceleration_structure) || hits(*descriptor),
                    naga::RayQueryFunction::GenerateIntersection { hit_t } => hits(*hit_t),
                    naga::RayQueryFunction::Proceed { .. }
                    | naga::RayQueryFunction::ConfirmIntersection
                    | naga::RayQueryFunction::Terminate => false,
                }
        }
        _ => false,
    }
}

// MARK: For-loop reconstruction analysis

/// Record every statement-index that references a local variable,
/// keyed by the local's handle.  Feeds the for-loop reconstruction
/// heuristic, which needs to see whether a candidate induction local
/// is used outside the loop that would absorb it.
fn collect_block_local_refs(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    expr_reads: &[Option<naga::Handle<naga::LocalVariable>>],
    seen: &mut Vec<bool>,
) {
    for stmt in block {
        match stmt {
            naga::Statement::Emit(range) => {
                for h in range.clone() {
                    if let Some(lh) = expr_reads[h.index()] {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::Store { pointer, .. }
            | naga::Statement::WorkGroupUniformLoad { pointer, .. } => {
                if let Some(lh) = resolve_local_var(*pointer, expressions) {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::Atomic { pointer, .. } => {
                if let Some(lh) = resolve_local_var(*pointer, expressions) {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                for e in [Some(*image), Some(*coordinate), *array_index, Some(*value)]
                    .into_iter()
                    .flatten()
                {
                    if let Some(lh) = resolve_local_var(e, expressions) {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::ImageAtomic {
                image,
                coordinate,
                array_index,
                value,
                ..
            } => {
                for e in [Some(*image), Some(*coordinate), *array_index, Some(*value)]
                    .into_iter()
                    .flatten()
                {
                    if let Some(lh) = resolve_local_var(e, expressions) {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::Return { value: Some(v) } => {
                if let Some(lh) = resolve_local_var(*v, expressions) {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::SubgroupBallot {
                predicate: Some(p), ..
            } => {
                if let Some(lh) = resolve_local_var(*p, expressions) {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::SubgroupGather { mode, argument, .. } => {
                if let Some(lh) = resolve_local_var(*argument, expressions) {
                    seen[lh.index()] = true;
                }
                let index = match mode {
                    naga::GatherMode::Broadcast(h)
                    | naga::GatherMode::Shuffle(h)
                    | naga::GatherMode::ShuffleDown(h)
                    | naga::GatherMode::ShuffleUp(h)
                    | naga::GatherMode::ShuffleXor(h)
                    | naga::GatherMode::QuadBroadcast(h) => Some(*h),
                    _ => None,
                };
                if let Some(idx) = index
                    && let Some(lh) = resolve_local_var(idx, expressions)
                {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::SubgroupCollectiveOperation { argument, .. } => {
                if let Some(lh) = resolve_local_var(*argument, expressions) {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::Call { arguments, .. } => {
                for &arg in arguments {
                    if let Some(lh) = resolve_local_var(arg, expressions) {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::RayPipelineFunction(fun) => match fun {
                naga::RayPipelineFunction::TraceRay {
                    acceleration_structure,
                    descriptor,
                    payload,
                } => {
                    for e in [*acceleration_structure, *descriptor, *payload] {
                        if let Some(lh) = resolve_local_var(e, expressions) {
                            seen[lh.index()] = true;
                        }
                    }
                }
            },
            naga::Statement::CooperativeStore { target, data } => {
                for e in [*target, data.pointer, data.stride] {
                    if let Some(lh) = resolve_local_var(e, expressions) {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::RayQuery { query, fun } => {
                if let Some(lh) = resolve_local_var(*query, expressions) {
                    seen[lh.index()] = true;
                }
                match fun {
                    naga::RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } => {
                        for e in [*acceleration_structure, *descriptor] {
                            if let Some(lh) = resolve_local_var(e, expressions) {
                                seen[lh.index()] = true;
                            }
                        }
                    }
                    naga::RayQueryFunction::GenerateIntersection { hit_t } => {
                        if let Some(lh) = resolve_local_var(*hit_t, expressions) {
                            seen[lh.index()] = true;
                        }
                    }
                    naga::RayQueryFunction::Proceed { .. }
                    | naga::RayQueryFunction::ConfirmIntersection
                    | naga::RayQueryFunction::Terminate => {}
                }
            }
            naga::Statement::Block(inner) => {
                collect_block_local_refs(inner, expressions, expr_reads, seen);
            }
            naga::Statement::If { accept, reject, .. } => {
                collect_block_local_refs(accept, expressions, expr_reads, seen);
                collect_block_local_refs(reject, expressions, expr_reads, seen);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    collect_block_local_refs(&case.body, expressions, expr_reads, seen);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                collect_block_local_refs(body, expressions, expr_reads, seen);
                collect_block_local_refs(continuing, expressions, expr_reads, seen);
            }
            _ => {}
        }
    }
}

/// A single-use `Call` result that may still be inlined into a later use
/// site, paired with the set of function-locals its arguments load.  The
/// locals set is what lets a Store to a local invalidate only the pending
/// calls that actually read that local (see the `Store` arm).
struct PendingCall {
    result: naga::Handle<naga::Expression>,
    reads_locals: std::collections::HashSet<naga::Handle<naga::LocalVariable>>,
}

/// Collect every function-local whose VALUE a call argument's evaluation
/// depends on, so a Store to that local cannot be reordered before the
/// pending call's (re-)evaluation at a later use site.  Two dependency kinds:
///
/// * a `Load` rooted at a local reads the local's value directly; and
/// * a POINTER argument rooted at a local (`&d`, `&d.f`, `&arr[i]` -
///   i.e. a bare `LocalVariable` / `Access` / `AccessIndex` reference) lets
///   the callee read the pointee at call time, so the result still depends on
///   the local's value when the call is evaluated.
///
/// Missing the pointer case lets a single-use `let c = g(&d); d = ...;` call
/// be inlined past the store, so the callee derefs the post-store value -
/// a silent reorder miscompilation.
fn collect_loaded_locals(
    expr: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    out: &mut std::collections::HashSet<naga::Handle<naga::LocalVariable>>,
    visited: &mut std::collections::HashSet<naga::Handle<naga::Expression>>,
) {
    // A common subexpression shared across argument positions forms a diamond
    // in the expression DAG; without a visited set it would be re-walked once
    // per incoming edge (super-linear).  `out` is a set so re-inserting is
    // harmless - only the traversal cost is saved.
    if !visited.insert(expr) {
        return;
    }
    match &expressions[expr] {
        naga::Expression::Load { pointer } => {
            if let Some(local) = resolve_local_var(*pointer, expressions) {
                out.insert(local);
            }
        }
        // A reference/pointer to a local (or a component of one).  Whether it
        // is passed by pointer to a callee or loaded later, the local's value
        // is observed at evaluation time.
        naga::Expression::LocalVariable(_)
        | naga::Expression::Access { .. }
        | naga::Expression::AccessIndex { .. } => {
            if let Some(local) = resolve_local_var(expr, expressions) {
                out.insert(local);
            }
        }
        _ => {}
    }
    visit_expr_children(&expressions[expr], |child| {
        collect_loaded_locals(child, expressions, out, visited)
    });
}

/// The root a pointer expression resolves to, for the write-effect analysis.
enum PointerRoot {
    /// A function-local variable (a write here is contained in the function).
    Local,
    /// A module-scope global (a write here escapes to every caller).
    Global,
    /// The function's own pointer PARAMETER `idx` (a write through it lands in
    /// whatever the caller passed - escapes to the caller).
    Param(u32),
    /// An exotic pointer expression - conservatively treated as escaping.
    Other,
}

fn resolve_pointer_root(
    ptr: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> PointerRoot {
    match &expressions[ptr] {
        naga::Expression::LocalVariable(_) => PointerRoot::Local,
        naga::Expression::GlobalVariable(_) => PointerRoot::Global,
        naga::Expression::FunctionArgument(i) => PointerRoot::Param(*i),
        naga::Expression::Access { base, .. } | naga::Expression::AccessIndex { base, .. } => {
            resolve_pointer_root(*base, expressions)
        }
        _ => PointerRoot::Other,
    }
}

/// The memory effects of a function that are observable OUTSIDE a call to it.
///
/// Writes to the function's own locals never escape; the two ways an effect
/// reaches the caller are tracked separately so a caller that supplies its OWN
/// local to a param-writing helper stays pure (the helper's write lands in the
/// caller's local, not the caller's caller):
/// * `escapes` - a write to a global, an atomic / image store / barrier / ray /
///   subgroup / cooperative op / `discard`, or such an effect transitively via
///   a callee.  Always observable, regardless of how the function is called.
/// * `written_params` - the function writes through these of its OWN pointer
///   parameters (directly, or by forwarding them to a param-writing callee).
///   Whether THAT escapes depends on what each caller passes.
#[derive(Clone)]
struct FnEffects {
    escapes: bool,
    written_params: std::collections::HashSet<u32>,
}

/// Fold the effect of one statement into `eff`.  Nested control-flow blocks are
/// walked by [`accumulate_block_effects`].
fn accumulate_statement_effects(
    stmt: &naga::Statement,
    expressions: &naga::Arena<naga::Expression>,
    module: &naga::Module,
    memo: &mut [Option<FnEffects>],
    eff: &mut FnEffects,
) {
    use naga::Statement as S;
    match stmt {
        S::Store { pointer, .. } | S::Atomic { pointer, .. } => {
            match resolve_pointer_root(*pointer, expressions) {
                PointerRoot::Local => {}
                PointerRoot::Param(i) => {
                    eff.written_params.insert(i);
                }
                PointerRoot::Global | PointerRoot::Other => eff.escapes = true,
            }
        }
        S::ImageStore { .. } | S::ImageAtomic { .. } | S::CooperativeStore { .. } => {
            eff.escapes = true
        }
        S::ControlBarrier(_) | S::MemoryBarrier(_) | S::WorkGroupUniformLoad { .. } => {
            eff.escapes = true
        }
        S::RayQuery { .. } | S::RayPipelineFunction(_) => eff.escapes = true,
        S::SubgroupBallot { .. }
        | S::SubgroupGather { .. }
        | S::SubgroupCollectiveOperation { .. } => eff.escapes = true,
        S::Kill => eff.escapes = true,
        S::Call {
            function,
            arguments,
            ..
        } => {
            let callee = function_effects(*function, module, memo);
            if callee.escapes {
                eff.escapes = true;
            }
            // Each global/local write the callee performs THROUGH a pointer
            // parameter lands in whatever WE passed for that parameter: our own
            // local stays contained, our own param forwards the escape outward,
            // a global (or an exotic pointer) escapes here and now.
            for &p in &callee.written_params {
                match arguments.get(p as usize) {
                    Some(&arg) => match resolve_pointer_root(arg, expressions) {
                        PointerRoot::Local => {}
                        PointerRoot::Param(i) => {
                            eff.written_params.insert(i);
                        }
                        PointerRoot::Global | PointerRoot::Other => eff.escapes = true,
                    },
                    None => eff.escapes = true, // arity mismatch - stay conservative
                }
            }
        }
        S::Block(inner) => accumulate_block_effects(inner, expressions, module, memo, eff),
        S::If { accept, reject, .. } => {
            accumulate_block_effects(accept, expressions, module, memo, eff);
            accumulate_block_effects(reject, expressions, module, memo, eff);
        }
        S::Switch { cases, .. } => {
            for case in cases {
                accumulate_block_effects(&case.body, expressions, module, memo, eff);
            }
        }
        S::Loop {
            body, continuing, ..
        } => {
            accumulate_block_effects(body, expressions, module, memo, eff);
            accumulate_block_effects(continuing, expressions, module, memo, eff);
        }
        S::Emit(_) | S::Return { .. } | S::Break | S::Continue => {}
    }
}

fn accumulate_block_effects(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    module: &naga::Module,
    memo: &mut [Option<FnEffects>],
    eff: &mut FnEffects,
) {
    for stmt in block.iter() {
        accumulate_statement_effects(stmt, expressions, module, memo, eff);
    }
}

/// Memoised [`FnEffects`] of `module.functions[h]`.  The call graph is acyclic
/// (naga forbids recursion); the in-progress marker (`escapes = true`) both
/// memoises and makes any unexpected cycle resolve to the conservative
/// "escapes everything", so the recursion always terminates.
fn function_effects(
    h: naga::Handle<naga::Function>,
    module: &naga::Module,
    memo: &mut [Option<FnEffects>],
) -> FnEffects {
    if let Some(known) = &memo[h.index()] {
        return known.clone();
    }
    memo[h.index()] = Some(FnEffects {
        escapes: true,
        written_params: std::collections::HashSet::new(),
    });
    let func = &module.functions[h];
    let mut eff = FnEffects {
        escapes: false,
        written_params: std::collections::HashSet::new(),
    };
    accumulate_block_effects(&func.body, &func.expressions, module, memo, &mut eff);
    memo[h.index()] = Some(eff.clone());
    eff
}

/// Per-`module.functions` inline-purity bitmap, computed once per module and
/// shared by every `find_inlineable_calls` invocation: a single-use `Call` is
/// only ever relocated to an arbitrary use site when its callee is inline-pure,
/// i.e. its only effect observable by the caller is its return value.  That is
/// exactly `!escapes && written_params.is_empty()` - a function writing through
/// one of its OWN params is NOT inline-pure (the write reaches the caller), but
/// a function that merely calls such a helper with its OWN local is.
pub(super) fn compute_pure_functions(module: &naga::Module) -> Vec<bool> {
    let mut memo: Vec<Option<FnEffects>> = vec![None; module.functions.len()];
    for (h, _) in module.functions.iter() {
        function_effects(h, module, &mut memo);
    }
    memo.into_iter()
        .map(|e| match e {
            Some(eff) => !eff.escapes && eff.written_params.is_empty(),
            None => false,
        })
        .collect()
}

/// Identify `Call` results that can be safely inlined at their single use
/// site instead of being bound to a `let`.  A result is inlineable when:
///
/// 1. its `ref_count` is exactly 1 (used once);
/// 2. its callee is PURE (`pure_functions[callee]`), OR it is an impure call
///    whose consuming statement evaluates NO other memory access (see
///    `last_impure` / `impure_call_inlines_safely`): an impure call writes
///    memory, and relocating it to an arbitrary use site would re-order that
///    write against an intervening read OR an operand-evaluation-order sibling
///    read of the same memory - a silent miscompile.  Operand order is
///    invisible here, so purity is the gate for the general case; the exception
///    is a consuming statement whose every other operand is memory-free, where
///    the call is the sole memory access and nothing can be reordered; and
/// 3. the consuming statement is reached from the `Call` without crossing a
///    potentially-interfering side-effecting statement (another `Call`,
///    `Atomic` / `ImageStore` / `RayQuery` and the like, a non-local `Store`,
///    or any control-flow boundary: `If` / `Loop` / `Switch` / `Block`).
///
/// A pure call still depends on its arguments' VALUES at evaluation time, so a
/// `Store` to a function-local does NOT unconditionally clear the pending set;
/// it invalidates only the pending calls whose arguments read that local
/// (tracked per-local via [`PendingCall`]'s `reads_locals`; see the `Store`
/// arm).  Consumption is recorded the moment a statement references a pending
/// result - the handle moves to the inlineable set immediately, so a later
/// clearing event cannot undo a use already made.  Evaluation order is thus
/// preserved: the use site stays in program order and the (pure) call's
/// re-evaluation reads the same argument values it would have at the call site.
fn find_inlineable_calls(
    block: &naga::Block,
    ref_counts: &[usize],
    expressions: &naga::Arena<naga::Expression>,
    pure_functions: &[bool],
) -> std::collections::HashSet<naga::Handle<naga::Expression>> {
    let mut result = std::collections::HashSet::new();
    let mut pending: Vec<PendingCall> = Vec::new();
    // The result of an IMPURE call from an earlier statement, still eligible to
    // be inlined into the statement that consumes it.  An impure call writes
    // memory, so it may be inlined only where its side effect cannot be
    // reordered against any other memory access - decided by
    // `impure_call_inlines_safely` (the consuming statement's every OTHER
    // operand must be memory-free).  `Emit` statements between the call and its
    // consumer only build the consuming expression and write nothing, so the
    // candidate survives them; the first non-`Emit` statement decides.
    let mut last_impure: Option<naga::Handle<naga::Expression>> = None;

    for stmt in block.iter() {
        if let Some(h) = last_impure {
            if matches!(stmt, naga::Statement::Emit(_)) {
                // Still assembling the consuming expression; keep `h` pending.
            } else {
                last_impure = None;
                if impure_call_inlines_safely(stmt, h, expressions) {
                    result.insert(h);
                }
            }
        }

        // Phase 1: record any pending handles consumed by this statement.
        // Order matters: we must detect consumption BEFORE applying the
        // statement's clearing rules, so a later control-flow statement
        // cannot retroactively drop a result whose use already happened.
        if !pending.is_empty() {
            consume_pending_for_statement(stmt, expressions, &mut pending, &mut result);
        }

        // Phase 2: apply the statement's effect on the pending set and
        // recurse into nested blocks.
        match stmt {
            naga::Statement::Call {
                result: Some(h),
                function,
                arguments,
                ..
            } if ref_counts[h.index()] == 1 => {
                // A new Call is itself a side-effecting statement.  Any
                // *prior* pending handle that was NOT consumed by this
                // Call's arguments (already handled by Phase 1) must be
                // dropped: inlining it at a later use site would reorder
                // its evaluation past this Call's side effects.
                pending.clear();
                // Only a PURE callee may be relocated to an arbitrary later use
                // site: an impure call's write would be re-ordered against an
                // intervening read - or, since operand evaluation order is not
                // visible here, against a sibling read in the use expression
                // itself.  A pure call still depends on its argument VALUES at
                // evaluation time, so track the locals its args read for the
                // `Store`-interference check below.
                if pure_functions[function.index()] {
                    let mut reads_locals = std::collections::HashSet::new();
                    let mut visited = std::collections::HashSet::new();
                    for &arg in arguments {
                        collect_loaded_locals(arg, expressions, &mut reads_locals, &mut visited);
                    }
                    pending.push(PendingCall {
                        result: *h,
                        reads_locals,
                    });
                } else {
                    // Impure: eligible to be inlined into its consuming
                    // statement only where that statement evaluates no other
                    // memory access (see `last_impure` /
                    // `impure_call_inlines_safely`).
                    last_impure = Some(*h);
                }
            }
            naga::Statement::Emit(_) | naga::Statement::Return { .. } => {
                // Non-side-effecting: keep pending calls
            }
            naga::Statement::Store { pointer, .. } => {
                // A Store to local `L` only interferes with a pending call
                // whose ARGUMENTS load `L`: inlining the call to a later use
                // site re-evaluates its argument text after the store, so a
                // `Load(L)` argument would read the post-store value.  Drop
                // exactly those pending calls; calls whose args don't read
                // `L` stay inlineable.  A non-local Store (storage / workgroup
                // global) can be observed by any callee, so clear everything.
                if let Some(stored) = resolve_local_var(*pointer, expressions) {
                    pending.retain(|p| !p.reads_locals.contains(&stored));
                } else {
                    pending.clear();
                }
            }
            naga::Statement::If { accept, reject, .. } => {
                pending.clear();
                result.extend(find_inlineable_calls(
                    accept,
                    ref_counts,
                    expressions,
                    pure_functions,
                ));
                result.extend(find_inlineable_calls(
                    reject,
                    ref_counts,
                    expressions,
                    pure_functions,
                ));
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                pending.clear();
                result.extend(find_inlineable_calls(
                    body,
                    ref_counts,
                    expressions,
                    pure_functions,
                ));
                result.extend(find_inlineable_calls(
                    continuing,
                    ref_counts,
                    expressions,
                    pure_functions,
                ));
            }
            naga::Statement::Switch { cases, .. } => {
                pending.clear();
                for case in cases {
                    result.extend(find_inlineable_calls(
                        &case.body,
                        ref_counts,
                        expressions,
                        pure_functions,
                    ));
                }
            }
            naga::Statement::Block(inner) => {
                pending.clear();
                result.extend(find_inlineable_calls(
                    inner,
                    ref_counts,
                    expressions,
                    pure_functions,
                ));
            }
            _ => {
                pending.clear();
            }
        }
    }

    result.extend(pending.into_iter().map(|p| p.result));
    result
}

/// Whether an impure single-use call producing `call_result` can be inlined
/// into `stmt`, the first non-`Emit` statement after the call.
///
/// Safe exactly when every expression `stmt` evaluates APART FROM the call is
/// memory-free (a literal / constant / by-value parameter, or arithmetic over
/// such).  Then the inlined call is the statement's ONLY memory access besides
/// its own terminal store, so its side effect cannot be reordered against a
/// sibling read, an intervening read, or a hoisted `let`-bound load - whatever
/// the call writes, nothing else in the statement observes it.  This makes the
/// operand evaluation order (and which sub-expressions the generator chooses to
/// `let`-bind) irrelevant, which is what keeps the analysis sound without
/// modelling either.
///
/// Subsumes the bare `out = call()` / `return call()` direct-value forms and
/// additionally recovers `out = (call() - .5) * k`, `if call() == k`,
/// `switch call()`, and `arr[const] = call()`.  An expression that reads memory
/// anywhere outside the call (e.g. `out = g + call()` with `g` a load) makes
/// the statement ineligible, so the call stays `let`-bound.  Only value-bearing
/// statement kinds are handled; any other consuming statement keeps the call
/// bound.
fn impure_call_inlines_safely(
    stmt: &naga::Statement,
    call_result: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> bool {
    use naga::Statement as S;
    let mut found = false;
    let mut memo = std::collections::HashMap::new();
    let mut memfree =
        |root| expr_is_memory_free(root, call_result, expressions, &mut found, &mut memo);
    let all_memfree = match stmt {
        // Evaluate BOTH operands (no short-circuit) so `found` is set whichever
        // side carries the call: an `arr[const] = call()` places it in `value`,
        // while a `bare = (call()..)` also keeps `pointer` memory-free.
        S::Store { pointer, value } => {
            let p = memfree(*pointer);
            let v = memfree(*value);
            p && v
        }
        S::Return { value: Some(v) } => memfree(*v),
        // Only the condition / selector is the consuming expression; the branch
        // bodies are later statements that legitimately access memory.
        S::If { condition, .. } => memfree(*condition),
        S::Switch { selector, .. } => memfree(*selector),
        _ => return false,
    };
    found && all_memfree
}

/// `true` when evaluating the expression tree rooted at `root` reads no memory
/// and observes no side effect, treating `call_result` as a transparent hole
/// (it is the one call we intend to inline) and setting `*found` when that hole
/// is reached.  A `Load`, any effect-result expression (call / atomic / image /
/// ray / subgroup / `arrayLength`), and any future variant default to NOT
/// memory-free, so the predicate is conservative by construction.
fn expr_is_memory_free(
    root: naga::Handle<naga::Expression>,
    call_result: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    found: &mut bool,
    memo: &mut std::collections::HashMap<naga::Handle<naga::Expression>, bool>,
) -> bool {
    if root == call_result {
        // The inlined call is reached on a unique path (`ref_count == 1`), so
        // this never collides with a memoised entry.
        *found = true;
        return true;
    }
    if let Some(&m) = memo.get(&root) {
        return m;
    }
    use naga::Expression as E;
    let memory_free = match &expressions[root] {
        E::Literal(_)
        | E::Constant(_)
        | E::Override(_)
        | E::ZeroValue(_)
        | E::FunctionArgument(_)
        | E::GlobalVariable(_)
        | E::LocalVariable(_) => true,
        E::Access { .. }
        | E::AccessIndex { .. }
        | E::Splat { .. }
        | E::Swizzle { .. }
        | E::Unary { .. }
        | E::Binary { .. }
        | E::Select { .. }
        | E::Relational { .. }
        | E::Math { .. }
        | E::As { .. }
        | E::Compose { .. }
        | E::Derivative { .. } => {
            let mut ok = true;
            visit_expr_children(&expressions[root], |child| {
                if !expr_is_memory_free(child, call_result, expressions, found, memo) {
                    ok = false;
                }
            });
            ok
        }
        _ => false,
    };
    memo.insert(root, memory_free);
    memory_free
}

/// Visit every direct expression handle referenced by `stmt` (for Emit
/// statements, this includes the direct children of every expression in the
/// emit range).  Any handle found in `pending` is moved to `result`.
/// Consume the pending call-result list into the `inlineable_calls`
/// set when the next statement is a single consumer, otherwise drop
/// the pending entries so they emit as explicit `let` bindings.
fn consume_pending_for_statement(
    stmt: &naga::Statement,
    expressions: &naga::Arena<naga::Expression>,
    pending: &mut Vec<PendingCall>,
    result: &mut std::collections::HashSet<naga::Handle<naga::Expression>>,
) {
    let check =
        |h: naga::Handle<naga::Expression>,
         pending: &mut Vec<PendingCall>,
         result: &mut std::collections::HashSet<naga::Handle<naga::Expression>>| {
            if let Some(pos) = pending.iter().position(|p| p.result == h) {
                result.insert(pending.swap_remove(pos).result);
            }
        };

    match stmt {
        naga::Statement::Emit(range) => {
            for h in range.clone() {
                let mut uses: Vec<naga::Handle<naga::Expression>> = Vec::new();
                visit_expr_children(&expressions[h], |child| uses.push(child));
                for u in uses {
                    check(u, pending, result);
                    if pending.is_empty() {
                        return;
                    }
                }
            }
        }
        naga::Statement::If { condition, .. } => check(*condition, pending, result),
        naga::Statement::Switch { selector, .. } => check(*selector, pending, result),
        // A loop's `break_if` is re-evaluated each iteration, but a pending
        // call from the ENCLOSING block was emitted once before the loop;
        // consuming it into `break_if` would recompute it per iteration - a
        // silent miscompile when a call argument reads a local the loop
        // mutates (the intra-block Store-interference guard cannot see those
        // nested writes).  Left unconsumed it falls to `pending.clear()` in
        // `find_inlineable_calls`'s Loop arm and emits as a pre-loop `let`;
        // calls emitted INSIDE the loop still inline via that function's
        // recursion into body/continuing with a fresh pending set.
        naga::Statement::Loop { .. } => {}
        naga::Statement::Return { value: Some(h) } => check(*h, pending, result),
        naga::Statement::Store { pointer, value } => {
            check(*pointer, pending, result);
            check(*value, pending, result);
        }
        naga::Statement::Call { arguments, .. } => {
            for a in arguments {
                check(*a, pending, result);
            }
        }
        naga::Statement::ImageStore {
            image,
            coordinate,
            array_index,
            value,
        } => {
            check(*image, pending, result);
            check(*coordinate, pending, result);
            if let Some(i) = array_index {
                check(*i, pending, result);
            }
            check(*value, pending, result);
        }
        naga::Statement::Atomic {
            pointer,
            value,
            fun,
            ..
        } => {
            check(*pointer, pending, result);
            check(*value, pending, result);
            crate::passes::expr_util::visit_atomic_function_handles(fun, &mut |h| {
                check(h, pending, result)
            });
        }
        naga::Statement::ImageAtomic {
            image,
            coordinate,
            array_index,
            value,
            fun,
        } => {
            check(*image, pending, result);
            check(*coordinate, pending, result);
            if let Some(i) = array_index {
                check(*i, pending, result);
            }
            check(*value, pending, result);
            crate::passes::expr_util::visit_atomic_function_handles(fun, &mut |h| {
                check(h, pending, result)
            });
        }
        naga::Statement::WorkGroupUniformLoad { pointer, .. } => {
            check(*pointer, pending, result);
        }
        naga::Statement::SubgroupBallot {
            predicate: Some(p), ..
        } => check(*p, pending, result),
        naga::Statement::SubgroupGather { argument, mode, .. } => {
            check(*argument, pending, result);
            match mode {
                naga::GatherMode::BroadcastFirst | naga::GatherMode::QuadSwap(_) => {}
                naga::GatherMode::Broadcast(h)
                | naga::GatherMode::Shuffle(h)
                | naga::GatherMode::ShuffleDown(h)
                | naga::GatherMode::ShuffleUp(h)
                | naga::GatherMode::ShuffleXor(h)
                | naga::GatherMode::QuadBroadcast(h) => check(*h, pending, result),
            }
        }
        naga::Statement::SubgroupCollectiveOperation { argument, .. } => {
            check(*argument, pending, result);
        }
        naga::Statement::RayQuery { query, fun } => {
            check(*query, pending, result);
            match fun {
                naga::RayQueryFunction::Initialize {
                    acceleration_structure,
                    descriptor,
                } => {
                    check(*acceleration_structure, pending, result);
                    check(*descriptor, pending, result);
                }
                naga::RayQueryFunction::GenerateIntersection { hit_t } => {
                    check(*hit_t, pending, result);
                }
                _ => {}
            }
        }
        naga::Statement::RayPipelineFunction(naga::RayPipelineFunction::TraceRay {
            acceleration_structure,
            descriptor,
            payload,
        }) => {
            check(*acceleration_structure, pending, result);
            check(*descriptor, pending, result);
            check(*payload, pending, result);
        }
        naga::Statement::CooperativeStore { target, data } => {
            check(*target, pending, result);
            check(data.pointer, pending, result);
            check(data.stride, pending, result);
        }
        // No direct expression references (or only nested blocks handled by recursion):
        naga::Statement::Block(_)
        | naga::Statement::Break
        | naga::Statement::Continue
        | naga::Statement::Kill
        | naga::Statement::ControlBarrier(_)
        | naga::Statement::MemoryBarrier(_)
        | naga::Statement::Return { value: None }
        | naga::Statement::SubgroupBallot {
            predicate: None, ..
        } => {}
    }
}

// MARK: Mutated-load binding analysis

/// The memory location a pointer refers to, resolved to a root variable
/// plus one level of refinement off that root.  Two places that share a
/// root but carry distinct *constant* first-level indices are provably
/// disjoint; anything coarser (`Whole` or a dynamic `Opaque` index)
/// conservatively aliases everything in the root.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PlaceRoot {
    Local(naga::Handle<naga::LocalVariable>),
    Global(naga::Handle<naga::GlobalVariable>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Refine {
    /// The whole variable (no access applied off the root).
    Whole,
    /// A single statically-known field / element index off the root.
    Field(u32),
    /// A dynamically-indexed (or otherwise unknown) location in the root.
    Opaque,
}

#[derive(Clone, Copy)]
struct Place {
    root: PlaceRoot,
    refine: Refine,
}

/// Conservative may-alias test.  Sound direction: returns `true` whenever
/// the two places *might* name overlapping memory.  Only proven-disjoint
/// pairs (same root, distinct constant first-level indices) return `false`.
fn places_may_alias(a: Place, b: Place) -> bool {
    if a.root != b.root {
        return false;
    }
    match (a.refine, b.refine) {
        (Refine::Field(x), Refine::Field(y)) => x == y,
        // `Whole` contains every field; `Opaque` could be any field.
        _ => true,
    }
}

/// Lower a pointer expression to its [`Place`], walking `Access` /
/// `AccessIndex` chains down to the root variable.  The refinement is the
/// access applied DIRECTLY to the root (the first level); deeper accesses
/// keep that first-level refinement (conservative - sub-locations of the
/// same first-level component are treated as aliasing).
///
/// Returns `None` when the root is not a concrete variable (a
/// function-argument pointer, or an exotic pointer expression).  Such a
/// place is treated as "Unknown" by the caller and aliases everything.
fn resolve_place(
    pointer: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> Option<Place> {
    match &expressions[pointer] {
        naga::Expression::LocalVariable(l) => Some(Place {
            root: PlaceRoot::Local(*l),
            refine: Refine::Whole,
        }),
        naga::Expression::GlobalVariable(g) => Some(Place {
            root: PlaceRoot::Global(*g),
            refine: Refine::Whole,
        }),
        naga::Expression::AccessIndex { base, index } => {
            let base_place = resolve_place(*base, expressions)?;
            if matches!(base_place.refine, Refine::Whole) {
                Some(Place {
                    root: base_place.root,
                    refine: Refine::Field(*index),
                })
            } else {
                Some(base_place)
            }
        }
        naga::Expression::Access { base, index } => {
            let base_place = resolve_place(*base, expressions)?;
            if matches!(base_place.refine, Refine::Whole) {
                let refine = const_index_value(*index, expressions)
                    .map(Refine::Field)
                    .unwrap_or(Refine::Opaque);
                Some(Place {
                    root: base_place.root,
                    refine,
                })
            } else {
                Some(base_place)
            }
        }
        _ => None,
    }
}

/// The constant value of an index expression, if it is a non-negative
/// integer `Literal`; otherwise `None` (a dynamic or non-integer index).
fn const_index_value(
    index: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
) -> Option<u32> {
    let naga::Expression::Literal(lit) = &expressions[index] else {
        return None;
    };
    match lit {
        naga::Literal::U32(v) => Some(*v),
        naga::Literal::I32(v) => u32::try_from(*v).ok(),
        naga::Literal::U64(v) => u32::try_from(*v).ok(),
        naga::Literal::I64(v) => u32::try_from(*v).ok(),
        naga::Literal::AbstractInt(v) => u32::try_from(*v).ok(),
        _ => None,
    }
}

/// `true` when a callee or another invocation could write this global
/// (so a load of it can become stale).  Immutable address spaces
/// (`uniform`, resource handles, immediate/push-constant, read-only
/// storage) can never change and never produce a hazard.
fn global_is_writable(module: &naga::Module, g: naga::Handle<naga::GlobalVariable>) -> bool {
    match module.global_variables[g].space {
        naga::AddressSpace::Uniform
        | naga::AddressSpace::Handle
        | naga::AddressSpace::Immediate => false,
        naga::AddressSpace::Storage { access } => access.contains(naga::StorageAccess::STORE),
        // Function (locals), Private, WorkGroup, ray/task payloads: writable.
        _ => true,
    }
}

/// `true` when texture global `g` is a STORE-access storage texture, i.e.
/// a `textureStore`/`textureAtomic` (here or in a callee) can mutate it, so
/// a prior `textureLoad` of it can go stale.  Sampled textures and
/// read-only storage textures are immutable resources and never produce a
/// hazard.  Note: textures live in `AddressSpace::Handle`, for which
/// [`global_is_writable`] returns `false` - writability for a texture is a
/// property of its storage *access*, not its address space, so the
/// `ImageLoad` hazard test MUST route through this helper, not that one.
///
/// A `binding_array<texture_storage_*<...>>` global has type
/// `TypeInner::BindingArray`, not `Image`, so peel one level to its element
/// type before classifying (WGSL/naga forbid nested binding arrays, so a
/// single peel suffices).  Without it a `textureLoad(texs[i], ..)` would be
/// dropped from the hazard set and inlined past a `textureStore` to the same
/// element - a silent miscompile.
fn image_is_writable_storage(module: &naga::Module, g: naga::Handle<naga::GlobalVariable>) -> bool {
    let mut inner = &module.types[module.global_variables[g].ty].inner;
    if let naga::TypeInner::BindingArray { base, .. } = inner {
        inner = &module.types[*base].inner;
    }
    matches!(
        inner,
        naga::TypeInner::Image {
            class: naga::ImageClass::Storage { access, .. },
            ..
        } if access.contains(naga::StorageAccess::STORE)
    )
}

/// A write a statement performs, as seen by the load-hazard analysis.
///
/// Every `naga::Statement` variant is classified exhaustively in
/// [`statement_write_effects`] (no wildcard arm), so a future statement
/// kind that can write memory forces a compile error there rather than a
/// silent miss - this enum needs no catch-all "writes everything" case.
enum WriteEffect {
    /// Writes a specific resolved place.
    Place(Place),
    /// May write some writable global (a callee, barrier, or
    /// param-pointer store): invalidates loads rooted at a global, plus
    /// any Unknown-place load (a param pointer may itself target a global).
    Globals,
}

impl WriteEffect {
    /// Whether this write could invalidate a tracked load whose place is
    /// `load` (`None` = an Unknown place that aliases everything).
    fn invalidates(&self, load: &Option<Place>) -> bool {
        match self {
            WriteEffect::Globals => match load {
                None => true,
                Some(p) => matches!(p.root, PlaceRoot::Global(_)),
            },
            WriteEffect::Place(w) => match load {
                // A `None` (function-argument pointer) load reads caller memory
                // or a global - NEVER a named local of THIS function (the caller
                // cannot hold a pointer to a local that does not exist in its
                // scope).  So a store to a resolved LOCAL place can never alias
                // it; a store to a GLOBAL still might (the param could point
                // there), so stay conservative for globals.
                None => !matches!(w.root, PlaceRoot::Local(_)),
                Some(p) => places_may_alias(*w, *p),
            },
        }
    }
}

/// Record the pointer-to-LOCAL write place a call argument exposes: a callee
/// taking `ptr<function, T>` may write the pointee.  A naga pointer argument is
/// a root variable (`LocalVariable` / `GlobalVariable` / `FunctionArgument`)
/// with optional `Access`/`AccessIndex` refinement, which `resolve_place` walks
/// to the root - so the pointee is exactly the argument's own place.  Pointers
/// are non-storable (no loadable pointer values, no pointer aggregates), so no
/// sub-expression can carry a second writable pointee, and an index
/// sub-expression is a value the callee reads, never writes; hence NO recursion
/// into children.  Only a LOCAL root is recorded - pointers to globals, and
/// param-pointer roots (which resolve to `None`), are already covered by the
/// blanket [`WriteEffect::Globals`] every `Call` records.
fn collect_ptr_local_writes(
    arg: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    out: &mut Vec<WriteEffect>,
) {
    if matches!(
        expressions[arg],
        naga::Expression::LocalVariable(_)
            | naga::Expression::Access { .. }
            | naga::Expression::AccessIndex { .. }
    ) && let Some(p) = resolve_place(arg, expressions)
        && matches!(p.root, PlaceRoot::Local(_))
    {
        out.push(WriteEffect::Place(p));
    }
}

/// Append every [`WriteEffect`] a single statement performs to `out`.
/// Control-flow statements (`Block`/`If`/`Switch`/`Loop`) contribute
/// nothing here - their nested blocks are walked separately.
fn statement_write_effects(
    stmt: &naga::Statement,
    expressions: &naga::Arena<naga::Expression>,
    out: &mut Vec<WriteEffect>,
) {
    use naga::Statement as S;
    match stmt {
        S::Store { pointer, .. } | S::Atomic { pointer, .. } => {
            match resolve_place(*pointer, expressions) {
                Some(p) => out.push(WriteEffect::Place(p)),
                // A store through an unresolved (function-argument) pointer
                // could land in any global; it cannot reach our own locals.
                None => out.push(WriteEffect::Globals),
            }
        }
        // The write is through `data.pointer` (the destination); `target` is the
        // matrix VALUE being stored, a read - not a written place.  (Latent today:
        // the generator cannot yet emit CooperativeStore; kept correct so the
        // exhaustive match holds no silent miss.)
        S::CooperativeStore { data, .. } => match resolve_place(data.pointer, expressions) {
            Some(p) => out.push(WriteEffect::Place(p)),
            None => out.push(WriteEffect::Globals),
        },
        S::Call { arguments, .. } => {
            // A callee may write any global, plus any local it receives by pointer.
            out.push(WriteEffect::Globals);
            for &arg in arguments {
                collect_ptr_local_writes(arg, expressions, out);
            }
        }
        // Memory-synchronisation points make other invocations' prior stores
        // to shared globals observable, so a pre-barrier load of such a global
        // can differ from a post-barrier re-read.  (Conservatively `Globals`;
        // this also over-invalidates private-space loads - harmless over-binding,
        // since a barrier cannot change a per-invocation private value.)
        S::ControlBarrier(_) | S::MemoryBarrier(_) | S::WorkGroupUniformLoad { .. } => {
            out.push(WriteEffect::Globals)
        }
        S::RayQuery { query, .. } => {
            out.push(WriteEffect::Globals);
            if let Some(p) = resolve_place(*query, expressions) {
                out.push(WriteEffect::Place(p));
            }
        }
        S::RayPipelineFunction(fun) => {
            out.push(WriteEffect::Globals);
            let naga::RayPipelineFunction::TraceRay { payload, .. } = fun;
            if let Some(p) = resolve_place(*payload, expressions) {
                out.push(WriteEffect::Place(p));
            }
        }
        // Image stores/atomics mutate a storage texture.  A buffer `Load`
        // never reaches a texture, but an `ImageLoad` (registered as a
        // pending load below) does, so the write must invalidate it.  The
        // destination is `image`; the stored value / atomic operand is a
        // read handled by the use-detection path.
        S::ImageStore { image, .. } | S::ImageAtomic { image, .. } => {
            match resolve_place(*image, expressions) {
                Some(p) => out.push(WriteEffect::Place(p)),
                // An image reached through an unresolved (function-argument)
                // value could be any texture global; stay conservative.
                None => out.push(WriteEffect::Globals),
            }
        }
        // Subgroup operations exchange already-computed values across lanes via
        // registers; they perform NO memory access and impose no memory
        // ordering, so they cannot stale any load.  (Their argument operands
        // are still seen as uses via the leaf use-detection path.)
        S::SubgroupBallot { .. }
        | S::SubgroupGather { .. }
        | S::SubgroupCollectiveOperation { .. } => {}
        // No memory write of their own (control flow handled by recursion).
        S::Emit(_)
        | S::Block(_)
        | S::If { .. }
        | S::Switch { .. }
        | S::Loop { .. }
        | S::Return { .. }
        | S::Break
        | S::Continue
        | S::Kill => {}
    }
}

/// Accumulate every [`WriteEffect`] performed anywhere inside `block`,
/// recursing through nested control flow.  Used to pre-mark loads that
/// outlive a loop's back-edge.
fn collect_block_write_effects(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    out: &mut Vec<WriteEffect>,
) {
    for stmt in block.iter() {
        statement_write_effects(stmt, expressions, out);
        match stmt {
            naga::Statement::Block(inner) => collect_block_write_effects(inner, expressions, out),
            naga::Statement::If { accept, reject, .. } => {
                collect_block_write_effects(accept, expressions, out);
                collect_block_write_effects(reject, expressions, out);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    collect_block_write_effects(&case.body, expressions, out);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                collect_block_write_effects(body, expressions, out);
                collect_block_write_effects(continuing, expressions, out);
            }
            _ => {}
        }
    }
}

/// A `Load` that has been emitted and is still in flight: its place plus
/// whether a write to that place has been observed since its `Emit`.
#[derive(Clone)]
struct PendingLoad {
    /// `None` = an Unknown place (function-argument pointer) - aliases all.
    place: Option<Place>,
    written: bool,
}

type Pending = std::collections::HashMap<naga::Handle<naga::Expression>, PendingLoad>;

/// Merge two control-flow successor states: a load is "written" after the
/// join if it was written on EITHER path (conservative).  Keys from both
/// sides are kept so a branch-local load that (legally) outlives its
/// branch is still tracked downstream.
fn merge_pending(mut a: Pending, b: Pending) -> Pending {
    for (h, pl) in b {
        a.entry(h)
            .and_modify(|e| e.written |= pl.written)
            .or_insert(pl);
    }
    a
}

/// Walk the operand cone of `root`, flagging every in-flight load that is
/// (a) reachable from `root` and (b) already marked written.  Such a load
/// is read AFTER its place was overwritten, so it must be bound.  The
/// `visited` set keeps the walk linear over shared sub-DAGs.
fn flag_used_loads(
    root: naga::Handle<naga::Expression>,
    expressions: &naga::Arena<naga::Expression>,
    pending: &Pending,
    must_bind: &mut std::collections::HashSet<naga::Handle<naga::Expression>>,
    visited: &mut std::collections::HashSet<naga::Handle<naga::Expression>>,
) {
    if !visited.insert(root) {
        return;
    }
    if let Some(pl) = pending.get(&root)
        && pl.written
    {
        must_bind.insert(root);
        // Stop here.  A written `root` is added to `must_bind`, so it emits as
        // a `let` at its own Emit site, freezing its WHOLE operand cone
        // (unbound children inlined, bound children naming their own earlier
        // `let`s) lexically before the write that marked it `written` -
        // including any nested written load reachable only via `root`.
        // Re-pinning a child would therefore change nothing.  A child also used
        // OUTSIDE this parent is still pinned at that other use: the early
        // return records only `root` in `visited` (children stay walkable), and
        // each statement walk starts a fresh `visited`.
        return;
    }
    visit_expr_children(&expressions[root], |child| {
        flag_used_loads(child, expressions, pending, must_bind, visited);
    });
}

/// Forward dataflow over one block, threading `pending` (in-flight loads)
/// and accumulating into `must_bind`.  See [`compute_must_bind_loads`].
fn analyze_block(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    module: &naga::Module,
    pending: &mut Pending,
    must_bind: &mut std::collections::HashSet<naga::Handle<naga::Expression>>,
) {
    for stmt in block.iter() {
        analyze_statement(stmt, expressions, module, pending, must_bind);
    }
}

/// Mark every in-flight load `written` whose place a write effect of `stmt` may
/// alias.  Monotone (only flips unwritten -> written) and skips already-written
/// loads, so it is idempotent across the loop pre-mark and the linear pass.
fn apply_writes(
    stmt: &naga::Statement,
    expressions: &naga::Arena<naga::Expression>,
    pending: &mut Pending,
) {
    let mut effects = Vec::new();
    statement_write_effects(stmt, expressions, &mut effects);
    if effects.is_empty() {
        return;
    }
    for pl in pending.values_mut() {
        if !pl.written && effects.iter().any(|e| e.invalidates(&pl.place)) {
            pl.written = true;
        }
    }
}

fn analyze_statement(
    stmt: &naga::Statement,
    expressions: &naga::Arena<naga::Expression>,
    module: &naga::Module,
    pending: &mut Pending,
    must_bind: &mut std::collections::HashSet<naga::Handle<naga::Expression>>,
) {
    use naga::Statement as S;
    match stmt {
        S::Emit(range) => {
            // Uses first (a write never occurs within an Emit): a load defined
            // in this same range is not yet pending, so a sibling consuming it
            // is correctly not flagged.
            let mut visited = std::collections::HashSet::new();
            for h in range.clone() {
                flag_used_loads(h, expressions, pending, must_bind, &mut visited);
            }
            // Then register the loads this Emit introduces.
            for h in range.clone() {
                match &expressions[h] {
                    naga::Expression::Load { pointer } => {
                        let place = resolve_place(*pointer, expressions);
                        let track = match &place {
                            Some(p) => match p.root {
                                PlaceRoot::Global(g) => global_is_writable(module, g),
                                PlaceRoot::Local(_) => true,
                            },
                            // Unknown place (function-argument pointer): track it -
                            // any later write may alias the pointee.
                            None => true,
                        };
                        if track {
                            pending.insert(
                                h,
                                PendingLoad {
                                    place,
                                    written: false,
                                },
                            );
                        }
                    }
                    // `textureLoad` reads a texel; a later `textureStore` /
                    // `textureAtomic` (or a callee) to the same storage texture
                    // can stale it, exactly like a buffer `Load`.  Track it so a
                    // single-use `textureLoad` is bound rather than inlined past
                    // the write.  Gate on storage-texture writability via
                    // `image_is_writable_storage` - NOT `global_is_writable`,
                    // which reports textures (Handle space) as non-writable and
                    // would silently drop the hazard.
                    naga::Expression::ImageLoad { image, .. } => {
                        let place = resolve_place(*image, expressions);
                        let track = match &place {
                            Some(p) => match p.root {
                                PlaceRoot::Global(g) => image_is_writable_storage(module, g),
                                // A texture is always a Handle-space global,
                                // never a local; treat a malformed Local root
                                // conservatively.
                                PlaceRoot::Local(_) => true,
                            },
                            // Texture passed as a value parameter: a callee may
                            // hold and store to it - track conservatively.
                            None => true,
                        };
                        if track {
                            pending.insert(
                                h,
                                PendingLoad {
                                    place,
                                    written: false,
                                },
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
        S::Block(inner) => analyze_block(inner, expressions, module, pending, must_bind),
        S::If {
            condition,
            accept,
            reject,
        } => {
            let mut visited = std::collections::HashSet::new();
            flag_used_loads(*condition, expressions, pending, must_bind, &mut visited);
            let mut accept_state = pending.clone();
            analyze_block(accept, expressions, module, &mut accept_state, must_bind);
            let mut reject_state = pending.clone();
            analyze_block(reject, expressions, module, &mut reject_state, must_bind);
            *pending = merge_pending(accept_state, reject_state);
        }
        S::Switch { selector, cases } => {
            let mut visited = std::collections::HashSet::new();
            flag_used_loads(*selector, expressions, pending, must_bind, &mut visited);
            if cases.iter().any(|c| c.fall_through) {
                // A fall-through case chains into the next, so a write in one
                // case can reach a use in a later one.  WGSL source never
                // produces fall-through, but handle it soundly by threading the
                // SAME state sequentially through the cases (a conservative
                // over-approximation: it also assumes a directly-entered case
                // ran after its predecessors, which only ever over-binds).
                for case in cases {
                    analyze_block(&case.body, expressions, module, pending, must_bind);
                }
            } else {
                // Cases are mutually exclusive: analyse each from the pre-switch
                // state and union (OR) their post-states.  Every case is
                // reachable (WGSL switches are exhaustive).
                let mut merged: Option<Pending> = None;
                for case in cases {
                    let mut case_state = pending.clone();
                    analyze_block(&case.body, expressions, module, &mut case_state, must_bind);
                    merged = Some(match merged {
                        None => case_state,
                        Some(m) => merge_pending(m, case_state),
                    });
                }
                if let Some(m) = merged {
                    *pending = m;
                }
            }
        }
        S::Loop {
            body,
            continuing,
            break_if,
        } => {
            // Back-edge: a write anywhere in the loop can execute before an
            // earlier-or-later use (next iteration) and after a load emitted
            // before the loop.  Pre-mark every outer load the loop may write.
            let mut loop_writes = Vec::new();
            collect_block_write_effects(body, expressions, &mut loop_writes);
            collect_block_write_effects(continuing, expressions, &mut loop_writes);
            if !loop_writes.is_empty() {
                for pl in pending.values_mut() {
                    if !pl.written && loop_writes.iter().any(|e| e.invalidates(&pl.place)) {
                        pl.written = true;
                    }
                }
            }
            // Loads emitted INSIDE the loop are re-evaluated each iteration
            // (expression values never cross the back-edge - only memory does),
            // so a single linear pass over body+continuing is exact for them.
            analyze_block(body, expressions, module, pending, must_bind);
            analyze_block(continuing, expressions, module, pending, must_bind);
            if let Some(h) = break_if {
                let mut visited = std::collections::HashSet::new();
                flag_used_loads(*h, expressions, pending, must_bind, &mut visited);
            }
        }
        // Leaf statements: their operands are uses, then their writes apply.
        _ => {
            let mut visited = std::collections::HashSet::new();
            crate::passes::expr_util::visit_statement_expression_handles(stmt, false, &mut |h| {
                flag_used_loads(h, expressions, pending, must_bind, &mut visited);
            });
            apply_writes(stmt, expressions, pending);
        }
    }
}

/// Identify every `Load` expression in `func` that must be bound to a
/// `let` rather than inlined, because the place it reads is written
/// between the `Load`'s `Emit` and a use of its value.  Inlining such a
/// load relocates the memory read past the write and yields the
/// post-write value - a silent miscompile.
///
/// The analysis is a single forward pass over the structured statement
/// tree (`analyze_*`).  It deliberately OVER-approximates the hazard
/// (binding a load is always semantically safe; the only cost is a few
/// bytes), so unresolved places, branches, loops, and call/barrier write
/// effects are all handled conservatively.  Read-only globals and pure
/// read-only locals are never flagged, so the common case stays inlined.
pub(super) fn compute_must_bind_loads(
    func: &naga::Function,
    module: &naga::Module,
) -> std::collections::HashSet<naga::Handle<naga::Expression>> {
    let mut pending: Pending = std::collections::HashMap::new();
    let mut must_bind = std::collections::HashSet::new();
    analyze_block(
        &func.body,
        &func.expressions,
        module,
        &mut pending,
        &mut must_bind,
    );
    must_bind
}

// MARK: Deferred-variable analysis

/// Identify locals whose declaration can be deferred to the site of
/// their first `Store` (at any nesting depth) and locals that turn
/// out to be entirely dead.  The returned vectors are indexed by
/// local handle.
///
/// A variable is deferrable when BOTH of these hold (the analysis does
/// NOT inspect `local.init` directly):
///
/// 1. its first reference in the enclosing block (considering both
///    reads in `Emit` ranges and writes in `Store` statements, plus
///    any sub-block references) is a *direct* whole-variable `Store`
///    at that block level; and
/// 2. all of its references are confined to that block, so the
///    `var` declaration emitted at the store site stays in scope
///    for every use.
///
/// Condition (1) is exactly what makes any initialiser dead-on-arrival:
/// the first thing that happens to the variable is a full overwrite, so
/// the init value is never observed.  The deferred `var` therefore drops
/// the init and re-emits it as the store's value at the deferred site;
/// a variable whose init IS live fails condition (1) (its first
/// reference is a read) and is not deferred.
pub(super) fn find_deferrable_vars(func: &naga::Function) -> (Vec<bool>, Vec<bool>) {
    use naga::Expression as E;

    let expr_len = func.expressions.len();
    let local_len = func.local_variables.len();

    // Map expression handles that **read** a local variable (via Load whose
    // pointer chain resolves to a LocalVariable).
    let mut expr_reads: Vec<Option<naga::Handle<naga::LocalVariable>>> = vec![None; expr_len];
    for (eh, expr) in func.expressions.iter() {
        if let E::Load { pointer } = *expr
            && let Some(lh) = resolve_local_var(pointer, &func.expressions)
        {
            expr_reads[eh.index()] = Some(lh);
        }
    }

    // All locals are candidates at the top level (the function body contains
    // every possible reference).
    let candidates = vec![true; local_len];

    let mut deferrable = vec![false; local_len];
    scan_block_deferrable_vars(
        &func.body,
        &func.expressions,
        &expr_reads,
        &candidates,
        &mut deferrable,
    );

    // Variables never referenced (no stores, no loads) are dead.
    // Use collect_block_local_refs on the whole function body to build
    // the complete `seen` set.
    let mut seen = vec![false; local_len];
    collect_block_local_refs(&func.body, &func.expressions, &expr_reads, &mut seen);
    let mut dead = vec![false; local_len];
    for (h, _) in func.local_variables.iter() {
        if !seen[h.index()] {
            dead[h.index()] = true;
        }
    }

    (deferrable, dead)
}

/// Recursively scan a block looking for deferrable-var opportunities.
///
/// `candidates[i]` is `true` when local `i` has all of its references within
/// `block` and has not already been marked deferrable.
/// DFS walker used by [`find_deferrable_vars`] to classify each
/// local's first-touch kind (read vs. store) and detect locals whose
/// only uses are stores.
fn scan_block_deferrable_vars(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    expr_reads: &[Option<naga::Handle<naga::LocalVariable>>],
    candidates: &[bool],
    result: &mut Vec<bool>,
) {
    let local_len = result.len();
    if !candidates.iter().any(|&b| b) {
        return;
    }

    // We need ownership info for the recursion step (determining which
    // sub-block a candidate is confined to).
    let ref_owner = compute_block_ownership(block, expressions, expr_reads, local_len);

    // Walk statements in program order, tracking which candidates have been
    // "seen" (any reference).  A direct Store to an unseen candidate is
    // deferrable at this block level.
    let mut seen = vec![false; local_len];
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Emit(range) => {
                for h in range.clone() {
                    if let Some(lh) = expr_reads[h.index()]
                        && candidates[lh.index()]
                    {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::Store { pointer, .. } => {
                if let naga::Expression::LocalVariable(lh) = expressions[*pointer] {
                    // Direct store to the whole variable.
                    if candidates[lh.index()] && !seen[lh.index()] && !result[lh.index()] {
                        result[lh.index()] = true;
                    }
                    seen[lh.index()] = true;
                } else if let Some(lh) = resolve_local_var(*pointer, expressions) {
                    // Indirect store (e.g. field/index access) - not deferrable,
                    // but the variable is now "seen".
                    if candidates[lh.index()] {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::Call { arguments, .. } => {
                for &arg in arguments {
                    if let Some(lh) = resolve_local_var(arg, expressions)
                        && candidates[lh.index()]
                    {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::Atomic { pointer, .. }
            | naga::Statement::WorkGroupUniformLoad { pointer, .. } => {
                if let Some(lh) = resolve_local_var(*pointer, expressions)
                    && candidates[lh.index()]
                {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                for e in [Some(*image), Some(*coordinate), *array_index, Some(*value)]
                    .into_iter()
                    .flatten()
                {
                    if let Some(lh) = resolve_local_var(e, expressions)
                        && candidates[lh.index()]
                    {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::ImageAtomic {
                image,
                coordinate,
                array_index,
                value,
                ..
            } => {
                for e in [Some(*image), Some(*coordinate), *array_index, Some(*value)]
                    .into_iter()
                    .flatten()
                {
                    if let Some(lh) = resolve_local_var(e, expressions)
                        && candidates[lh.index()]
                    {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::Return { value: Some(v) } => {
                if let Some(lh) = resolve_local_var(*v, expressions)
                    && candidates[lh.index()]
                {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::SubgroupBallot {
                predicate: Some(p), ..
            } => {
                if let Some(lh) = resolve_local_var(*p, expressions)
                    && candidates[lh.index()]
                {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::SubgroupGather { mode, argument, .. } => {
                if let Some(lh) = resolve_local_var(*argument, expressions)
                    && candidates[lh.index()]
                {
                    seen[lh.index()] = true;
                }
                let index = match mode {
                    naga::GatherMode::Broadcast(h)
                    | naga::GatherMode::Shuffle(h)
                    | naga::GatherMode::ShuffleDown(h)
                    | naga::GatherMode::ShuffleUp(h)
                    | naga::GatherMode::ShuffleXor(h)
                    | naga::GatherMode::QuadBroadcast(h) => Some(*h),
                    _ => None,
                };
                if let Some(idx) = index
                    && let Some(lh) = resolve_local_var(idx, expressions)
                    && candidates[lh.index()]
                {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::SubgroupCollectiveOperation { argument, .. } => {
                if let Some(lh) = resolve_local_var(*argument, expressions)
                    && candidates[lh.index()]
                {
                    seen[lh.index()] = true;
                }
            }
            naga::Statement::RayPipelineFunction(fun) => match fun {
                naga::RayPipelineFunction::TraceRay {
                    acceleration_structure,
                    descriptor,
                    payload,
                } => {
                    for e in [*acceleration_structure, *descriptor, *payload] {
                        if let Some(lh) = resolve_local_var(e, expressions)
                            && candidates[lh.index()]
                        {
                            seen[lh.index()] = true;
                        }
                    }
                }
            },
            naga::Statement::CooperativeStore { target, data } => {
                for e in [*target, data.pointer, data.stride] {
                    if let Some(lh) = resolve_local_var(e, expressions)
                        && candidates[lh.index()]
                    {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::RayQuery { query, fun } => {
                if let Some(lh) = resolve_local_var(*query, expressions)
                    && candidates[lh.index()]
                {
                    seen[lh.index()] = true;
                }
                match fun {
                    naga::RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } => {
                        for e in [*acceleration_structure, *descriptor] {
                            if let Some(lh) = resolve_local_var(e, expressions)
                                && candidates[lh.index()]
                            {
                                seen[lh.index()] = true;
                            }
                        }
                    }
                    naga::RayQueryFunction::GenerateIntersection { hit_t } => {
                        if let Some(lh) = resolve_local_var(*hit_t, expressions)
                            && candidates[lh.index()]
                        {
                            seen[lh.index()] = true;
                        }
                    }
                    naga::RayQueryFunction::Proceed { .. }
                    | naga::RayQueryFunction::ConfirmIntersection
                    | naga::RayQueryFunction::Terminate => {}
                }
            }
            _ => {
                // For control-flow and other compound statements, conservatively
                // mark every candidate local referenced in sub-blocks as seen.
                match stmt {
                    naga::Statement::If { accept, reject, .. } => {
                        collect_block_local_refs(accept, expressions, expr_reads, &mut seen);
                        collect_block_local_refs(reject, expressions, expr_reads, &mut seen);
                    }
                    naga::Statement::Switch { cases, .. } => {
                        for case in cases {
                            collect_block_local_refs(
                                &case.body,
                                expressions,
                                expr_reads,
                                &mut seen,
                            );
                        }
                    }
                    naga::Statement::Loop {
                        body, continuing, ..
                    } => {
                        collect_block_local_refs(body, expressions, expr_reads, &mut seen);
                        collect_block_local_refs(continuing, expressions, expr_reads, &mut seen);
                    }
                    naga::Statement::Block(inner) => {
                        collect_block_local_refs(inner, expressions, expr_reads, &mut seen);
                    }
                    _ => {}
                }
            }
        }
    }

    // Recurse into compound statements for candidates not resolved at this
    // level.  A candidate that is owned by a single compound statement can
    // potentially be deferred inside that statement's sub-block.
    for (idx, stmt) in block.iter().enumerate() {
        let any_owned =
            (0..local_len).any(|i| candidates[i] && !result[i] && ref_owner[i] == Some(idx));
        if !any_owned {
            continue;
        }

        match stmt {
            naga::Statement::If { accept, reject, .. } => {
                let mut seen_a = vec![false; local_len];
                let mut seen_r = vec![false; local_len];
                collect_block_local_refs(accept, expressions, expr_reads, &mut seen_a);
                collect_block_local_refs(reject, expressions, expr_reads, &mut seen_r);
                let a_cands: Vec<bool> = (0..local_len)
                    .map(|i| {
                        candidates[i]
                            && !result[i]
                            && ref_owner[i] == Some(idx)
                            && seen_a[i]
                            && !seen_r[i]
                    })
                    .collect();
                let r_cands: Vec<bool> = (0..local_len)
                    .map(|i| {
                        candidates[i]
                            && !result[i]
                            && ref_owner[i] == Some(idx)
                            && seen_r[i]
                            && !seen_a[i]
                    })
                    .collect();
                scan_block_deferrable_vars(accept, expressions, expr_reads, &a_cands, result);
                scan_block_deferrable_vars(reject, expressions, expr_reads, &r_cands, result);
            }
            naga::Statement::Switch { cases, .. } => {
                let case_seen: Vec<Vec<bool>> = cases
                    .iter()
                    .map(|case| {
                        let mut s = vec![false; local_len];
                        collect_block_local_refs(&case.body, expressions, expr_reads, &mut s);
                        s
                    })
                    .collect();
                for (ci, case) in cases.iter().enumerate() {
                    let cands: Vec<bool> = (0..local_len)
                        .map(|i| {
                            candidates[i]
                                && !result[i]
                                && ref_owner[i] == Some(idx)
                                && case_seen[ci][i]
                                && !case_seen.iter().enumerate().any(|(cj, s)| cj != ci && s[i])
                        })
                        .collect();
                    scan_block_deferrable_vars(&case.body, expressions, expr_reads, &cands, result);
                }
            }
            naga::Statement::Block(inner) => {
                let cands: Vec<bool> = (0..local_len)
                    .map(|i| candidates[i] && !result[i] && ref_owner[i] == Some(idx))
                    .collect();
                scan_block_deferrable_vars(inner, expressions, expr_reads, &cands, result);
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                // Recurse into body for candidates confined to body only.
                let mut seen_b = vec![false; local_len];
                let mut seen_c = vec![false; local_len];
                collect_block_local_refs(body, expressions, expr_reads, &mut seen_b);
                collect_block_local_refs(continuing, expressions, expr_reads, &mut seen_c);
                let b_cands: Vec<bool> = (0..local_len)
                    .map(|i| {
                        candidates[i]
                            && !result[i]
                            && ref_owner[i] == Some(idx)
                            && seen_b[i]
                            && !seen_c[i]
                    })
                    .collect();
                scan_block_deferrable_vars(body, expressions, expr_reads, &b_cands, result);
            }
            _ => {}
        }
    }
}

/// Identify locals whose references are confined to exactly one `Loop`
/// statement (at any nesting depth), so their declaration can be absorbed
/// into `for(var x=init;...)` or `for(var x:type;...)` (no init).
/// Return the per-local bitmap of init-once locals whose uses stay
/// confined to a single `Loop` and can therefore be absorbed into
/// that loop's `for (var ...; ...; ...)` header.
fn find_for_loop_vars(
    func: &naga::Function,
    must_bind_loads: &std::collections::HashSet<naga::Handle<naga::Expression>>,
) -> Vec<bool> {
    use naga::Expression as E;

    let expr_len = func.expressions.len();
    let local_len = func.local_variables.len();

    // Build Load->LocalVariable map, same pattern as find_deferrable_vars.
    let mut expr_reads: Vec<Option<naga::Handle<naga::LocalVariable>>> = vec![None; expr_len];
    for (eh, expr) in func.expressions.iter() {
        if let E::Load { pointer } = *expr
            && let Some(lh) = resolve_local_var(pointer, &func.expressions)
        {
            expr_reads[eh.index()] = Some(lh);
        }
    }

    // All locals are candidates at the top level.
    // Variables without an explicit init are zero-initialised in WGSL and can
    // still serve as for-loop counters (e.g. after a dead-init removal pass).
    let mut candidates = vec![false; local_len];
    for (h, _local) in func.local_variables.iter() {
        candidates[h.index()] = true;
    }

    let mut result = vec![false; local_len];
    scan_block_for_loop_vars(
        &func.body,
        &func.expressions,
        &func.local_variables,
        &expr_reads,
        &candidates,
        &mut result,
        must_bind_loads,
    );
    result
}

/// Sentinel value indicating a local is referenced by multiple statements.
const MULTI_OWNER: usize = usize::MAX;

/// Mark a local as referenced by statement `idx` in a block.  If it was
/// already referenced by a different statement, set it to `MULTI_OWNER`.
/// Tag each local's `ref_owner` slot with the loop index that owns
/// it.  `None` means no owner yet, `Some(idx)` pins the local to one
/// loop, and `Some(sentinel)` marks it as conflicted (used outside
/// any candidate loop).
fn mark_owner(ref_owner: &mut [Option<usize>], lh_idx: usize, idx: usize) {
    match ref_owner[lh_idx] {
        None => ref_owner[lh_idx] = Some(idx),
        Some(prev) if prev == idx => {} // same stmt, no change
        _ => ref_owner[lh_idx] = Some(MULTI_OWNER),
    }
}

/// For each local variable, compute which statement index in `block` "owns"
/// all of its references.  Returns `None` if the local is not referenced in
/// this block, `Some(idx)` if all references are within statement `idx`, or
/// `Some(MULTI_OWNER)` if referenced by multiple statements.
/// Traverse `block`, assigning loop ownership to every local
/// reference and flagging cross-loop escapes.  Cooperating helper
/// for [`find_for_loop_vars`].
fn compute_block_ownership(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    expr_reads: &[Option<naga::Handle<naga::LocalVariable>>],
    local_len: usize,
) -> Vec<Option<usize>> {
    let mut ref_owner: Vec<Option<usize>> = vec![None; local_len];
    let mut tmp_seen = vec![false; local_len];

    for (idx, stmt) in block.iter().enumerate() {
        match stmt {
            naga::Statement::Emit(range) => {
                for h in range.clone() {
                    if let Some(lh) = expr_reads[h.index()] {
                        mark_owner(&mut ref_owner, lh.index(), idx);
                    }
                }
            }
            naga::Statement::Store { pointer, .. } => {
                if let Some(lh) = resolve_local_var(*pointer, expressions) {
                    mark_owner(&mut ref_owner, lh.index(), idx);
                }
            }
            naga::Statement::Call { arguments, .. } => {
                for &arg in arguments {
                    if let Some(lh) = resolve_local_var(arg, expressions) {
                        mark_owner(&mut ref_owner, lh.index(), idx);
                    }
                }
            }
            naga::Statement::Return { value: Some(v) } => {
                if let Some(lh) = resolve_local_var(*v, expressions) {
                    mark_owner(&mut ref_owner, lh.index(), idx);
                }
            }
            naga::Statement::Atomic { pointer, .. }
            | naga::Statement::WorkGroupUniformLoad { pointer, .. } => {
                if let Some(lh) = resolve_local_var(*pointer, expressions) {
                    mark_owner(&mut ref_owner, lh.index(), idx);
                }
            }
            naga::Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                for e in [Some(*image), Some(*coordinate), *array_index, Some(*value)]
                    .into_iter()
                    .flatten()
                {
                    if let Some(lh) = resolve_local_var(e, expressions) {
                        mark_owner(&mut ref_owner, lh.index(), idx);
                    }
                }
            }
            naga::Statement::ImageAtomic {
                image,
                coordinate,
                array_index,
                value,
                ..
            } => {
                for e in [Some(*image), Some(*coordinate), *array_index, Some(*value)]
                    .into_iter()
                    .flatten()
                {
                    if let Some(lh) = resolve_local_var(e, expressions) {
                        mark_owner(&mut ref_owner, lh.index(), idx);
                    }
                }
            }
            naga::Statement::SubgroupBallot {
                predicate: Some(p), ..
            } => {
                if let Some(lh) = resolve_local_var(*p, expressions) {
                    mark_owner(&mut ref_owner, lh.index(), idx);
                }
            }
            naga::Statement::SubgroupGather { mode, argument, .. } => {
                if let Some(lh) = resolve_local_var(*argument, expressions) {
                    mark_owner(&mut ref_owner, lh.index(), idx);
                }
                let index = match mode {
                    naga::GatherMode::Broadcast(h)
                    | naga::GatherMode::Shuffle(h)
                    | naga::GatherMode::ShuffleDown(h)
                    | naga::GatherMode::ShuffleUp(h)
                    | naga::GatherMode::ShuffleXor(h)
                    | naga::GatherMode::QuadBroadcast(h) => Some(*h),
                    _ => None,
                };
                if let Some(idx_h) = index
                    && let Some(lh) = resolve_local_var(idx_h, expressions)
                {
                    mark_owner(&mut ref_owner, lh.index(), idx);
                }
            }
            naga::Statement::SubgroupCollectiveOperation { argument, .. } => {
                if let Some(lh) = resolve_local_var(*argument, expressions) {
                    mark_owner(&mut ref_owner, lh.index(), idx);
                }
            }
            naga::Statement::RayPipelineFunction(fun) => match fun {
                naga::RayPipelineFunction::TraceRay {
                    acceleration_structure,
                    descriptor,
                    payload,
                } => {
                    for e in [*acceleration_structure, *descriptor, *payload] {
                        if let Some(lh) = resolve_local_var(e, expressions) {
                            mark_owner(&mut ref_owner, lh.index(), idx);
                        }
                    }
                }
            },
            naga::Statement::CooperativeStore { target, data } => {
                for e in [*target, data.pointer, data.stride] {
                    if let Some(lh) = resolve_local_var(e, expressions) {
                        mark_owner(&mut ref_owner, lh.index(), idx);
                    }
                }
            }
            naga::Statement::RayQuery { query, fun } => {
                if let Some(lh) = resolve_local_var(*query, expressions) {
                    mark_owner(&mut ref_owner, lh.index(), idx);
                }
                match fun {
                    naga::RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } => {
                        for e in [*acceleration_structure, *descriptor] {
                            if let Some(lh) = resolve_local_var(e, expressions) {
                                mark_owner(&mut ref_owner, lh.index(), idx);
                            }
                        }
                    }
                    naga::RayQueryFunction::GenerateIntersection { hit_t } => {
                        if let Some(lh) = resolve_local_var(*hit_t, expressions) {
                            mark_owner(&mut ref_owner, lh.index(), idx);
                        }
                    }
                    naga::RayQueryFunction::Proceed { .. }
                    | naga::RayQueryFunction::ConfirmIntersection
                    | naga::RayQueryFunction::Terminate => {}
                }
            }
            _ => {}
        }
        // For compound statements, scan sub-blocks and attribute to `idx`.
        match stmt {
            naga::Statement::If { accept, reject, .. } => {
                tmp_seen.fill(false);
                collect_block_local_refs(accept, expressions, expr_reads, &mut tmp_seen);
                collect_block_local_refs(reject, expressions, expr_reads, &mut tmp_seen);
            }
            naga::Statement::Switch { cases, .. } => {
                tmp_seen.fill(false);
                for case in cases {
                    collect_block_local_refs(&case.body, expressions, expr_reads, &mut tmp_seen);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                tmp_seen.fill(false);
                collect_block_local_refs(body, expressions, expr_reads, &mut tmp_seen);
                collect_block_local_refs(continuing, expressions, expr_reads, &mut tmp_seen);
            }
            naga::Statement::Block(inner) => {
                tmp_seen.fill(false);
                collect_block_local_refs(inner, expressions, expr_reads, &mut tmp_seen);
            }
            _ => {
                continue; // Leaf statement - no sub-block refs to drain.
            }
        }
        for (i, &s) in tmp_seen.iter().enumerate() {
            if s {
                mark_owner(&mut ref_owner, i, idx);
            }
        }
    }

    ref_owner
}

/// Check whether a `Loop` statement matches the for-loop pattern
/// recognised by `try_emit_for_loop`:
///
/// - `break_if` is `None`;
/// - `continuing` has at most one non-`Emit` statement (the update);
/// - the update, when present, is a `Store`, `Call`, or `ImageStore`;
/// - `body` starts with an if-break guard.
fn is_for_loop_candidate(
    body: &naga::Block,
    continuing: &naga::Block,
    break_if: &Option<naga::Handle<naga::Expression>>,
    expressions: &naga::Arena<naga::Expression>,
    must_bind_loads: &std::collections::HashSet<naga::Handle<naga::Expression>>,
) -> bool {
    // Parse via the SHARED parser so this var-suppression decision and
    // `try_emit_for_loop`'s emission decision can never drift.  `None` => not
    // for-convertible (break_if present, no if-break guard, or >1 continuing
    // core update statement).
    let Some(shape) = super::stmt_emit::parse_for_loop_shape(body, continuing, break_if) else {
        return false;
    };
    // Update must be Store / Call / ImageStore, matching try_emit_for_loop's
    // pre-validation.
    if let Some(stmt) = shape.update_stmt
        && !matches!(
            stmt,
            naga::Statement::Store { .. }
                | naga::Statement::Call { .. }
                | naga::Statement::ImageStore { .. }
        )
    {
        return false;
    }
    // And the preloads must be safe to inline into the for-header, else
    // try_emit_for_loop bails to plain `loop` emission and the suppressed
    // counter `var` would be left undeclared.
    super::stmt_emit::for_loop_preload_inlining_is_safe(
        &shape,
        body,
        continuing,
        expressions,
        must_bind_loads,
    )
}

/// Recursively scan a block (and its nested sub-blocks) looking for
/// for-loop-shaped `Loop` statements that fully confine candidate locals.
///
/// `candidates[i]` is `true` when local `i` has all of its references
/// within `block` and is eligible for for-loop absorption.
/// DFS walker that identifies candidate loops and records which
/// locals are safe to absorb into each.  Complements
/// [`compute_block_ownership`], which handles the liveness half of
/// the decision.
fn scan_block_for_loop_vars(
    block: &naga::Block,
    expressions: &naga::Arena<naga::Expression>,
    local_variables: &naga::Arena<naga::LocalVariable>,
    expr_reads: &[Option<naga::Handle<naga::LocalVariable>>],
    candidates: &[bool],
    result: &mut Vec<bool>,
    must_bind_loads: &std::collections::HashSet<naga::Handle<naga::Expression>>,
) {
    let local_len = result.len();
    if !candidates.iter().any(|&b| b) {
        return;
    }

    let ref_owner = compute_block_ownership(block, expressions, expr_reads, local_len);
    let stmts: Vec<_> = block.iter().collect();

    // Check for-loop candidates: locals owned by a single Loop at this level.
    for (h, _local) in local_variables.iter() {
        let i = h.index();
        if !candidates[i] || result[i] {
            continue;
        }
        if let Some(owner) = ref_owner[i] {
            if owner == MULTI_OWNER {
                continue;
            }
            if let naga::Statement::Loop {
                body,
                continuing,
                break_if,
            } = stmts[owner]
                && is_for_loop_candidate(body, continuing, break_if, expressions, must_bind_loads)
            {
                let has_update = continuing.iter().any(|s| {
                    if let naga::Statement::Store { pointer, .. } = s {
                        matches!(
                            expressions[*pointer],
                            naga::Expression::LocalVariable(lh) if lh == h
                        )
                    } else {
                        false
                    }
                });
                if has_update {
                    result[i] = true;
                }
            }
        }
    }

    // Recurse into compound statements for remaining candidates.
    for (idx, stmt) in block.iter().enumerate() {
        // Collect candidates that are owned by this statement and not yet resolved.
        let any_owned =
            (0..local_len).any(|i| candidates[i] && !result[i] && ref_owner[i] == Some(idx));
        if !any_owned {
            continue;
        }

        match stmt {
            naga::Statement::If { accept, reject, .. } => {
                // Determine which sub-block each candidate is confined to.
                let mut seen_a = vec![false; local_len];
                let mut seen_r = vec![false; local_len];
                collect_block_local_refs(accept, expressions, expr_reads, &mut seen_a);
                collect_block_local_refs(reject, expressions, expr_reads, &mut seen_r);
                let a_cands: Vec<bool> = (0..local_len)
                    .map(|i| {
                        candidates[i]
                            && !result[i]
                            && ref_owner[i] == Some(idx)
                            && seen_a[i]
                            && !seen_r[i]
                    })
                    .collect();
                let r_cands: Vec<bool> = (0..local_len)
                    .map(|i| {
                        candidates[i]
                            && !result[i]
                            && ref_owner[i] == Some(idx)
                            && seen_r[i]
                            && !seen_a[i]
                    })
                    .collect();
                scan_block_for_loop_vars(
                    accept,
                    expressions,
                    local_variables,
                    expr_reads,
                    &a_cands,
                    result,
                    must_bind_loads,
                );
                scan_block_for_loop_vars(
                    reject,
                    expressions,
                    local_variables,
                    expr_reads,
                    &r_cands,
                    result,
                    must_bind_loads,
                );
            }
            naga::Statement::Switch { cases, .. } => {
                let case_seen: Vec<Vec<bool>> = cases
                    .iter()
                    .map(|case| {
                        let mut s = vec![false; local_len];
                        collect_block_local_refs(&case.body, expressions, expr_reads, &mut s);
                        s
                    })
                    .collect();
                for (ci, case) in cases.iter().enumerate() {
                    let cands: Vec<bool> = (0..local_len)
                        .map(|i| {
                            candidates[i]
                                && !result[i]
                                && ref_owner[i] == Some(idx)
                                && case_seen[ci][i]
                                && !case_seen.iter().enumerate().any(|(cj, s)| cj != ci && s[i])
                        })
                        .collect();
                    scan_block_for_loop_vars(
                        &case.body,
                        expressions,
                        local_variables,
                        expr_reads,
                        &cands,
                        result,
                        must_bind_loads,
                    );
                }
            }
            naga::Statement::Block(inner) => {
                let cands: Vec<bool> = (0..local_len)
                    .map(|i| candidates[i] && !result[i] && ref_owner[i] == Some(idx))
                    .collect();
                scan_block_for_loop_vars(
                    inner,
                    expressions,
                    local_variables,
                    expr_reads,
                    &cands,
                    result,
                    must_bind_loads,
                );
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                // This Loop was either already handled as a for-loop candidate
                // above, or it doesn't match the pattern.  Either way, recurse
                // into the body for locals that are confined to body only
                // (not referenced in continuing).
                let mut seen_b = vec![false; local_len];
                let mut seen_c = vec![false; local_len];
                collect_block_local_refs(body, expressions, expr_reads, &mut seen_b);
                collect_block_local_refs(continuing, expressions, expr_reads, &mut seen_c);
                let b_cands: Vec<bool> = (0..local_len)
                    .map(|i| {
                        candidates[i]
                            && !result[i]
                            && ref_owner[i] == Some(idx)
                            && seen_b[i]
                            && !seen_c[i]
                    })
                    .collect();
                scan_block_for_loop_vars(
                    body,
                    expressions,
                    local_variables,
                    expr_reads,
                    &b_cands,
                    result,
                    must_bind_loads,
                );
            }
            _ => {}
        }
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
