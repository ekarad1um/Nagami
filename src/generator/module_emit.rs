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
        // Mirrors naga's own enable detection (back/wgsl/writer.rs).
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
                if !preamble.is_empty() {
                    if let Some(name) = ty.name.as_deref() {
                        if preamble.contains(name) {
                            continue;
                        }
                    }
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
            if !preamble.is_empty() {
                if let Some(name) = c.name.as_deref() {
                    if preamble.contains(name) {
                        continue;
                    }
                }
            }
            self.out.push_str("const ");
            self.out.push_str(&self.constant_names[h.index()]);
            // Omit the type annotation when the RHS text already carries a
            // concrete type: Compose/ZeroValue have explicit type constructors,
            // and concrete Literal values now carry typed suffixes (e.g. `f`,
            // `i`, `u`).  We keep the annotation for abstract literals,
            // Constant refs, and arithmetic where the textual type may differ.
            let init_expr = &self.module.global_expressions[c.init];
            let rhs_has_explicit_type = match init_expr {
                naga::Expression::Compose { .. }
                | naga::Expression::ZeroValue(_)
                | naga::Expression::Splat { .. } => true,
                naga::Expression::Literal(lit) => !matches!(
                    lit,
                    naga::Literal::AbstractInt(_) | naga::Literal::AbstractFloat(_)
                ),
                _ => false,
            };
            if !rhs_has_explicit_type {
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
            if !preamble.is_empty() {
                if let Some(name) = ov.name.as_deref() {
                    if preamble.contains(name) {
                        continue;
                    }
                }
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
            if !preamble.is_empty() {
                if let Some(name) = g.name.as_deref() {
                    if preamble.contains(name) {
                        continue;
                    }
                }
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
                    // Elide the default `read` access mode - WGSL
                    // defaults storage bindings to read-only.
                    if access != naga::StorageAccess::LOAD {
                        self.push_separator();
                        self.out.push_str(storage_access(access));
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
            if !preamble.is_empty() {
                if let Some(name) = f.name.as_deref() {
                    if preamble.contains(name) {
                        continue;
                    }
                }
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
            let need_align = member.offset != expected_offset && member.offset > expected_offset;

            if need_align {
                // Find the smallest power-of-2 alignment that, applied to
                // default_offset, yields at least member.offset.
                // We try an @align attribute on THIS member.
                let a = 1u32 << member.offset.trailing_zeros();
                // Verify: round_up(default_offset, a) == member.offset
                let align_obj = naga::proc::Alignment::new(a);
                let works = align_obj
                    .map(|ao| ao.round_up(default_offset) == member.offset)
                    .unwrap_or(false);
                if works {
                    self.out.push_str(&format!("@align({a}) "));
                } else {
                    // Fallback: use @size on previous member to pad.
                    // This case is handled below via @size.
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
        let for_loop_vars = find_for_loop_vars(func);
        let inlineable_calls = find_inlineable_calls(&func.body, &ref_counts, &func.expressions);
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
                        self.options.max_precision,
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

        self.close_brace();
        Ok(())
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

/// Saturating increment of `counts[h]`.  Saturating arithmetic is
/// only a safety net because the real ceiling is the statement
/// count, which fits comfortably in `usize` for any realistic shader.
fn bump(counts: &mut [usize], h: naga::Handle<naga::Expression>) {
    counts[h.index()] += 1;
}

/// Visit each direct child expression handle of `expr`, calling `f` for each.
/// Invoke `f` on every child expression handle directly referenced
/// by `expr`.  Declarative and result expressions yield nothing.
/// Exhaustive per variant so a new naga expression forces the caller
/// to decide how to classify it.
pub(super) fn visit_expr_children(
    expr: &naga::Expression,
    mut f: impl FnMut(naga::Handle<naga::Expression>),
) {
    use naga::Expression as E;
    match expr {
        E::Literal(_)
        | E::Constant(_)
        | E::Override(_)
        | E::ZeroValue(_)
        | E::FunctionArgument(_)
        | E::GlobalVariable(_)
        | E::LocalVariable(_)
        | E::CallResult(_)
        | E::AtomicResult { .. }
        | E::WorkGroupUniformLoadResult { .. }
        | E::RayQueryProceedResult
        | E::SubgroupBallotResult
        | E::SubgroupOperationResult { .. } => {}
        E::Compose { components, .. } => {
            for h in components {
                f(*h);
            }
        }
        E::Access { base, index } => {
            f(*base);
            f(*index);
        }
        E::AccessIndex { base, .. } => f(*base),
        E::Splat { value, .. } => f(*value),
        E::Swizzle { vector, .. } => f(*vector),
        E::Load { pointer } => f(*pointer),
        E::ImageSample {
            image,
            sampler,
            coordinate,
            array_index,
            offset,
            level,
            depth_ref,
            ..
        } => {
            f(*image);
            f(*sampler);
            f(*coordinate);
            if let Some(i) = array_index {
                f(*i);
            }
            if let Some(o) = offset {
                f(*o);
            }
            match level {
                naga::SampleLevel::Auto | naga::SampleLevel::Zero => {}
                naga::SampleLevel::Exact(h) | naga::SampleLevel::Bias(h) => f(*h),
                naga::SampleLevel::Gradient { x, y } => {
                    f(*x);
                    f(*y);
                }
            }
            if let Some(d) = depth_ref {
                f(*d);
            }
        }
        E::ImageLoad {
            image,
            coordinate,
            array_index,
            sample,
            level,
        } => {
            f(*image);
            f(*coordinate);
            if let Some(i) = array_index {
                f(*i);
            }
            if let Some(s) = sample {
                f(*s);
            }
            if let Some(l) = level {
                f(*l);
            }
        }
        E::ImageQuery { image, query } => {
            f(*image);
            if let naga::ImageQuery::Size { level: Some(l) } = query {
                f(*l);
            }
        }
        E::Unary { expr, .. } => f(*expr),
        E::Binary { left, right, .. } => {
            f(*left);
            f(*right);
        }
        E::Select {
            condition,
            accept,
            reject,
        } => {
            f(*condition);
            f(*accept);
            f(*reject);
        }
        E::Derivative { expr, .. } => f(*expr),
        E::Relational { argument, .. } => f(*argument),
        E::Math {
            arg,
            arg1,
            arg2,
            arg3,
            ..
        } => {
            f(*arg);
            if let Some(v) = arg1 {
                f(*v);
            }
            if let Some(v) = arg2 {
                f(*v);
            }
            if let Some(v) = arg3 {
                f(*v);
            }
        }
        E::As { expr, .. } => f(*expr),
        E::ArrayLength(h) => f(*h),
        E::RayQueryVertexPositions { query, .. } => f(*query),
        E::RayQueryGetIntersection { query, .. } => f(*query),
        E::CooperativeLoad { data, .. } => {
            f(data.pointer);
            f(data.stride);
        }
        E::CooperativeMultiplyAdd { a, b, c } => {
            f(*a);
            f(*b);
            f(*c);
        }
    }
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
                if let naga::AtomicFunction::Exchange {
                    compare: Some(compare),
                } = fun
                {
                    bump(counts, *compare);
                }
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
                if let naga::AtomicFunction::Exchange {
                    compare: Some(compare),
                } = fun
                {
                    bump(counts, *compare);
                }
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
                if let Some(idx) = index {
                    if let Some(lh) = resolve_local_var(idx, expressions) {
                        seen[lh.index()] = true;
                    }
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

/// Identify Call results that can be safely inlined at their single use site
/// instead of being bound to a `let` variable.
///
/// A result is inlineable when:
///
/// 1. its `ref_count` is exactly 1 (used once); and
/// 2. the statement that consumes the result is reached from the
///    `Call` without crossing a potentially-interfering
///    side-effecting statement (another `Call`, non-local `Store`,
///    `Atomic` / `ImageStore` / `RayQuery` and so on, or any
///    control-flow boundary: `If` / `Loop` / `Switch` / `Block`).
///
/// Stores to local variables are exempt because locals are function-scoped
/// and cannot interfere with the call's execution.
///
/// Consumption detection: as we walk, each statement is inspected for direct
/// handle references (Store value, If condition, Emit-range expression
/// children, etc.).  When a statement references a pending Call result, that
/// handle is moved from `pending` to the inlineable set immediately - so any
/// later clearing event cannot undo a use that has already happened.  This is
/// the key correctness property: evaluation order is preserved because the
/// use site is fixed in program order and any side effect after it is
/// unaffected by inlining at the (pure-expression) use site.
///
/// This turns `let C = A(b.x); return C;` into `return A(b.x);`, and more
/// importantly preserves inlining across subsequent control-flow/side-effects
/// whose only previous effect was to wipe the pending set.
/// Identify `Call` results with exactly one downstream use that can
/// be safely inlined at the call site.  The heuristic requires the
/// result to be consumed before the next side-effecting statement so
/// inlining never re-orders observable effects.
fn find_inlineable_calls(
    block: &naga::Block,
    ref_counts: &[usize],
    expressions: &naga::Arena<naga::Expression>,
) -> std::collections::HashSet<naga::Handle<naga::Expression>> {
    let mut result = std::collections::HashSet::new();
    let mut pending: Vec<naga::Handle<naga::Expression>> = Vec::new();

    for stmt in block.iter() {
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
                result: Some(h), ..
            } if ref_counts[h.index()] == 1 => {
                // A new Call is itself a side-effecting statement.  Any
                // *prior* pending handle that was NOT consumed by this
                // Call's arguments (already handled by Phase 1) must be
                // dropped: inlining it at a later use site would reorder
                // its evaluation past this Call's side effects.
                pending.clear();
                pending.push(*h);
            }
            naga::Statement::Emit(_) | naga::Statement::Return { .. } => {
                // Non-side-effecting: keep pending calls
            }
            naga::Statement::Store { pointer, .. } => {
                // Stores to local variables cannot interfere with pending
                // call results - locals are function-scoped and inaccessible
                // to callees.  Only clear pending for non-local stores.
                if resolve_local_var(*pointer, expressions).is_none() {
                    pending.clear();
                }
            }
            naga::Statement::If { accept, reject, .. } => {
                pending.clear();
                result.extend(find_inlineable_calls(accept, ref_counts, expressions));
                result.extend(find_inlineable_calls(reject, ref_counts, expressions));
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                pending.clear();
                result.extend(find_inlineable_calls(body, ref_counts, expressions));
                result.extend(find_inlineable_calls(continuing, ref_counts, expressions));
            }
            naga::Statement::Switch { cases, .. } => {
                pending.clear();
                for case in cases {
                    result.extend(find_inlineable_calls(&case.body, ref_counts, expressions));
                }
            }
            naga::Statement::Block(inner) => {
                pending.clear();
                result.extend(find_inlineable_calls(inner, ref_counts, expressions));
            }
            _ => {
                pending.clear();
            }
        }
    }

    result.extend(pending);
    result
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
    pending: &mut Vec<naga::Handle<naga::Expression>>,
    result: &mut std::collections::HashSet<naga::Handle<naga::Expression>>,
) {
    let check =
        |h: naga::Handle<naga::Expression>,
         pending: &mut Vec<naga::Handle<naga::Expression>>,
         result: &mut std::collections::HashSet<naga::Handle<naga::Expression>>| {
            if let Some(pos) = pending.iter().position(|&p| p == h) {
                result.insert(pending.swap_remove(pos));
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
        naga::Statement::Loop { break_if, .. } => {
            if let Some(h) = break_if {
                check(*h, pending, result);
            }
        }
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
            if let naga::AtomicFunction::Exchange { compare: Some(cmp) } = fun {
                check(*cmp, pending, result);
            }
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
            if let naga::AtomicFunction::Exchange { compare: Some(cmp) } = fun {
                check(*cmp, pending, result);
            }
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

// MARK: Deferred-variable analysis

/// Identify locals whose declaration can be deferred to the site of
/// their first `Store` (at any nesting depth) and locals that turn
/// out to be entirely dead.  The returned vectors are indexed by
/// local handle.
///
/// A variable is deferrable when:
///
/// 1. it has no initialiser in the `LocalVariable` arena (or its
///    init is dead, i.e. overwritten before any read);
/// 2. its first reference in the enclosing block (considering both
///    reads in `Emit` ranges and writes in `Store` statements, plus
///    any sub-block references) is a *direct* `Store` at that block
///    level; and
/// 3. all of its references are confined to that block, so the
///    `var` declaration emitted at the store site stays in scope
///    for every use.
fn find_deferrable_vars(func: &naga::Function) -> (Vec<bool>, Vec<bool>) {
    use naga::Expression as E;

    let expr_len = func.expressions.len();
    let local_len = func.local_variables.len();

    // Map expression handles that **read** a local variable (via Load whose
    // pointer chain resolves to a LocalVariable).
    let mut expr_reads: Vec<Option<naga::Handle<naga::LocalVariable>>> = vec![None; expr_len];
    for (eh, expr) in func.expressions.iter() {
        if let E::Load { pointer } = *expr {
            if let Some(lh) = resolve_local_var(pointer, &func.expressions) {
                expr_reads[eh.index()] = Some(lh);
            }
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
                    if let Some(lh) = expr_reads[h.index()] {
                        if candidates[lh.index()] {
                            seen[lh.index()] = true;
                        }
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
                    if let Some(lh) = resolve_local_var(arg, expressions) {
                        if candidates[lh.index()] {
                            seen[lh.index()] = true;
                        }
                    }
                }
            }
            naga::Statement::Atomic { pointer, .. }
            | naga::Statement::WorkGroupUniformLoad { pointer, .. } => {
                if let Some(lh) = resolve_local_var(*pointer, expressions) {
                    if candidates[lh.index()] {
                        seen[lh.index()] = true;
                    }
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
                        if candidates[lh.index()] {
                            seen[lh.index()] = true;
                        }
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
                        if candidates[lh.index()] {
                            seen[lh.index()] = true;
                        }
                    }
                }
            }
            naga::Statement::Return { value: Some(v) } => {
                if let Some(lh) = resolve_local_var(*v, expressions) {
                    if candidates[lh.index()] {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::SubgroupBallot {
                predicate: Some(p), ..
            } => {
                if let Some(lh) = resolve_local_var(*p, expressions) {
                    if candidates[lh.index()] {
                        seen[lh.index()] = true;
                    }
                }
            }
            naga::Statement::SubgroupGather { mode, argument, .. } => {
                if let Some(lh) = resolve_local_var(*argument, expressions) {
                    if candidates[lh.index()] {
                        seen[lh.index()] = true;
                    }
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
                if let Some(idx) = index {
                    if let Some(lh) = resolve_local_var(idx, expressions) {
                        if candidates[lh.index()] {
                            seen[lh.index()] = true;
                        }
                    }
                }
            }
            naga::Statement::SubgroupCollectiveOperation { argument, .. } => {
                if let Some(lh) = resolve_local_var(*argument, expressions) {
                    if candidates[lh.index()] {
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
                            if candidates[lh.index()] {
                                seen[lh.index()] = true;
                            }
                        }
                    }
                }
            },
            naga::Statement::CooperativeStore { target, data } => {
                for e in [*target, data.pointer, data.stride] {
                    if let Some(lh) = resolve_local_var(e, expressions) {
                        if candidates[lh.index()] {
                            seen[lh.index()] = true;
                        }
                    }
                }
            }
            naga::Statement::RayQuery { query, fun } => {
                if let Some(lh) = resolve_local_var(*query, expressions) {
                    if candidates[lh.index()] {
                        seen[lh.index()] = true;
                    }
                }
                match fun {
                    naga::RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } => {
                        for e in [*acceleration_structure, *descriptor] {
                            if let Some(lh) = resolve_local_var(e, expressions) {
                                if candidates[lh.index()] {
                                    seen[lh.index()] = true;
                                }
                            }
                        }
                    }
                    naga::RayQueryFunction::GenerateIntersection { hit_t } => {
                        if let Some(lh) = resolve_local_var(*hit_t, expressions) {
                            if candidates[lh.index()] {
                                seen[lh.index()] = true;
                            }
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
fn find_for_loop_vars(func: &naga::Function) -> Vec<bool> {
    use naga::Expression as E;

    let expr_len = func.expressions.len();
    let local_len = func.local_variables.len();

    // Build Load->LocalVariable map, same pattern as find_deferrable_vars.
    let mut expr_reads: Vec<Option<naga::Handle<naga::LocalVariable>>> = vec![None; expr_len];
    for (eh, expr) in func.expressions.iter() {
        if let E::Load { pointer } = *expr {
            if let Some(lh) = resolve_local_var(pointer, &func.expressions) {
                expr_reads[eh.index()] = Some(lh);
            }
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
                if let Some(idx_h) = index {
                    if let Some(lh) = resolve_local_var(idx_h, expressions) {
                        mark_owner(&mut ref_owner, lh.index(), idx);
                    }
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
/// - the update, when present, is a `Store` or `Call`;
/// - `body` starts with an if-break guard.
fn is_for_loop_candidate(
    body: &naga::Block,
    continuing: &naga::Block,
    break_if: &Option<naga::Handle<naga::Expression>>,
) -> bool {
    if break_if.is_some() {
        return false;
    }
    // Continuing may start with `WorkGroupUniformLoad` preloads and then
    // have at most 1 core update statement.
    let mut non_emit_count = 0;
    let mut update_stmt = None;
    for s in continuing.iter() {
        match s {
            naga::Statement::Emit(_) => continue,
            naga::Statement::WorkGroupUniformLoad { .. } if update_stmt.is_none() => continue,
            _ => {
                non_emit_count += 1;
                if non_emit_count > 1 {
                    return false;
                }
                update_stmt = Some(s);
            }
        }
    }
    // Update must be Store or Call (matching try_emit_for_loop's pre-validation).
    if let Some(stmt) = update_stmt {
        if !matches!(
            stmt,
            naga::Statement::Store { .. }
                | naga::Statement::Call { .. }
                | naga::Statement::ImageStore { .. }
        ) {
            return false;
        }
    }
    // Body must start with an If-break guard (after leading Emits and optional
    // WorkGroupUniformLoad preloads).
    let mut body_iter = body.iter();
    let first_guard = loop {
        match body_iter.next() {
            Some(naga::Statement::Emit(_)) => continue,
            Some(naga::Statement::WorkGroupUniformLoad { .. }) => continue,
            other => break other,
        }
    };
    match first_guard {
        Some(naga::Statement::If { accept, reject, .. }) => {
            (accept.is_empty()
                && reject.len() == 1
                && matches!(reject.iter().next(), Some(naga::Statement::Break)))
                || (reject.is_empty()
                    && accept.len() == 1
                    && matches!(accept.iter().next(), Some(naga::Statement::Break)))
        }
        _ => false,
    }
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
            {
                if is_for_loop_candidate(body, continuing, break_if) {
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
                );
                scan_block_for_loop_vars(
                    reject,
                    expressions,
                    local_variables,
                    expr_reads,
                    &r_cands,
                    result,
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
