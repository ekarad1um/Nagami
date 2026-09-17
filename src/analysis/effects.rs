//! What a statement does to memory, and what a function does to memory that
//! a caller can observe.  [`statement_effects`] is the single list of writing
//! statement variants, shared by the function summary and the generator's
//! forced binding, so neither can count a write the other misses.

/// The root a pointer expression resolves to, for the write-effect analysis.
pub(crate) enum PointerRoot {
    /// A write here is contained in the function.
    Local,
    /// A write here escapes to every caller.
    Global,
    /// The function's own pointer parameter: a write through it lands in
    /// whatever the caller passed.
    Param(u32),
    /// An exotic pointer expression, treated as escaping.
    Other,
}

pub(crate) fn resolve_pointer_root(
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

/// One effect of a statement on memory (or on the invocation itself);
/// operands are reported as the statement holds them, unresolved.
pub(crate) enum Effect {
    /// A write through a pointer of this function: a `Store` / atomic
    /// pointer, a cooperative store's destination, the ray-query object,
    /// `traceRay`'s payload, or an argument the callee writes through.
    Write(naga::Handle<naga::Expression>),
    /// A `textureStore` / `textureAtomic` on `image`.
    ImageWrite(naga::Handle<naga::Expression>),
    /// Any writable global may change: a barrier makes other invocations'
    /// stores visible, `traceRay` runs other shader stages, a callee writes
    /// a global (or escapes some other way).
    Globals,
    /// Observable without touching memory: `discard`, a subgroup operation.
    Observable,
}

/// Every [`Effect`] of `stmt` itself (nested blocks are the caller's), a
/// `Call` by its callee's summary.  No wildcard arm: a new writing statement
/// variant is a compile error, not a silent miss.  `inline(never)`: LLVM
/// copies the match into every caller otherwise.
#[inline(never)]
pub(crate) fn statement_effects(
    stmt: &naga::Statement,
    callee: &mut dyn FnMut(naga::Handle<naga::Function>) -> FnEffects,
    visit: &mut dyn FnMut(Effect),
) {
    use naga::Statement as S;
    match stmt {
        S::Store { pointer, .. } | S::Atomic { pointer, .. } => visit(Effect::Write(*pointer)),
        // The destination is `data.pointer`; `target` is the stored matrix
        // value, a read.
        S::CooperativeStore { data, .. } => visit(Effect::Write(data.pointer)),
        // The stored value / atomic operand is a read.
        S::ImageStore { image, .. } | S::ImageAtomic { image, .. } => {
            visit(Effect::ImageWrite(*image))
        }
        S::ControlBarrier(_) | S::MemoryBarrier(_) | S::WorkGroupUniformLoad { .. } => {
            visit(Effect::Globals)
        }
        // Inline traversal touches only the query object; it reads the
        // acceleration structure, an immutable global.
        S::RayQuery { query, .. } => visit(Effect::Write(*query)),
        S::RayPipelineFunction(naga::RayPipelineFunction::TraceRay { payload, .. }) => {
            visit(Effect::Globals);
            visit(Effect::Write(*payload));
        }
        // Register exchange across lanes: no memory access, but the lanes
        // present at the statement are the result.
        S::SubgroupBallot { .. }
        | S::SubgroupGather { .. }
        | S::SubgroupCollectiveOperation { .. }
        | S::Kill => visit(Effect::Observable),
        S::Call {
            function,
            arguments,
            ..
        } => {
            let effects = callee(*function);
            if effects.escapes {
                visit(Effect::Globals);
            }
            let mut written = effects.written_params;
            while written != 0 {
                let p = written.trailing_zeros();
                written &= written - 1;
                match arguments.get(p as usize) {
                    Some(&arg) => visit(Effect::Write(arg)),
                    // Arity mismatch: stay conservative.
                    None => visit(Effect::Globals),
                }
            }
        }
        S::Emit(_)
        | S::Block(_)
        | S::If { .. }
        | S::Switch { .. }
        | S::Loop { .. }
        | S::Return { .. }
        | S::Break
        | S::Continue => {}
    }
}

/// A function's memory effects observable OUTSIDE a call to it.  Writes to its
/// own locals never escape; the two escape routes are tracked separately so a
/// caller that passes its OWN local to a param-writing helper stays pure:
/// `escapes` (a global write, an image store, a barrier, `traceRay`, a
/// subgroup op, `discard`, or any of these via a callee) is always
/// observable; `written_params` (writes through the function's own pointer
/// parameters, directly or via a callee) escape depending on what each
/// caller passes.  One bit per parameter; a write through a parameter past
/// the sixty-fourth counts as an escape.
#[derive(Clone, Copy)]
pub(crate) struct FnEffects {
    pub(crate) escapes: bool,
    pub(crate) written_params: u64,
}

impl FnEffects {
    const ESCAPES: Self = Self {
        escapes: true,
        written_params: 0,
    };

    /// A single-use `Call` is relocated to an arbitrary use site only when
    /// the callee's sole caller-observable effect is its return value.  A
    /// function writing through its OWN param is not inline-pure; one that
    /// merely calls such a helper with its OWN local is.
    pub(crate) fn inline_pure(self) -> bool {
        !self.escapes && self.written_params == 0
    }

    fn write_param(&mut self, index: u32) {
        match 1u64.checked_shl(index) {
            Some(bit) => self.written_params |= bit,
            None => self.escapes = true,
        }
    }
}

/// Memoised [`FnEffects`] of `module.functions[h]`.  naga forbids recursion,
/// and the in-progress marker (`escapes = true`) makes any unexpected cycle
/// resolve to "escapes everything", so the recursion always terminates.
fn function_effects(
    h: naga::Handle<naga::Function>,
    module: &naga::Module,
    memo: &mut [Option<FnEffects>],
) -> FnEffects {
    if let Some(known) = memo[h.index()] {
        return known;
    }
    memo[h.index()] = Some(FnEffects::ESCAPES);
    let func = &module.functions[h];
    let mut eff = FnEffects {
        escapes: false,
        written_params: 0,
    };
    crate::ir::visit::for_each_statement(&func.body, &mut |stmt| {
        statement_effects(
            stmt,
            &mut |callee| function_effects(callee, module, memo),
            &mut |effect| match effect {
                // A write through a pointer lands where its root says: our
                // local stays contained, our param forwards the escape to
                // the caller's choice, a global or exotic pointer escapes.
                Effect::Write(pointer) => match resolve_pointer_root(pointer, &func.expressions) {
                    PointerRoot::Local => {}
                    PointerRoot::Param(i) => eff.write_param(i),
                    PointerRoot::Global | PointerRoot::Other => eff.escapes = true,
                },
                // A texture is a global or a texture parameter, never a local.
                Effect::ImageWrite(_) | Effect::Globals | Effect::Observable => eff.escapes = true,
            },
        );
    });
    memo[h.index()] = Some(eff);
    eff
}

/// Every function's summary, indexed like `module.functions`.
pub(crate) fn compute_fn_effects(module: &naga::Module) -> Vec<FnEffects> {
    let mut memo: Vec<Option<FnEffects>> = vec![None; module.functions.len()];
    for (h, _) in module.functions.iter() {
        function_effects(h, module, &mut memo);
    }
    memo.into_iter()
        .map(|e| e.expect("every function was summarised"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summaries(src: &str) -> (naga::Module, Vec<FnEffects>) {
        let module = naga::front::wgsl::parse_str(src).expect("parse failed");
        let effects = compute_fn_effects(&module);
        (module, effects)
    }

    fn params(eff: &FnEffects) -> Vec<u32> {
        (0..64)
            .filter(|i| eff.written_params >> i & 1 == 1)
            .collect()
    }

    /// Each escape route once, and the summary of a call is the callee's
    /// summary applied to what the caller passed.
    #[test]
    fn a_functions_summary_is_what_a_caller_can_observe() {
        let (_, eff) = summaries(
            r#"
            @group(0) @binding(0) var<storage, read_write> buf: array<i32, 4>;
            var<workgroup> w: i32;
            fn pure(x: i32) -> i32 { var t = x; t = t * 2; return t; }
            fn global(x: i32) -> i32 { buf[0] = x; return x; }
            fn param(p: ptr<function, i32>, q: ptr<function, i32>) -> i32 { *q = 1; return *p; }
            fn own_local() -> i32 { var v = 1; var u = 2; return param(&v, &u); }
            fn forwards(a: ptr<function, i32>, b: ptr<function, i32>) -> i32 { return param(b, a); }
            fn barrier() { workgroupBarrier(); }
            fn kills() { discard; }
            fn calls_global() -> i32 { return global(1); }
        "#,
        );
        let [
            pure,
            global,
            param,
            own_local,
            forwards,
            barrier,
            kills,
            calls_global,
        ] = <[FnEffects; 8]>::try_from(eff)
            .ok()
            .expect("eight functions");
        assert!(pure.inline_pure());
        assert!(global.escapes);
        assert!(!param.escapes && params(&param) == [1]);
        assert!(
            own_local.inline_pure(),
            "a param write into the caller's own local is contained"
        );
        assert!(
            !forwards.escapes && params(&forwards) == [0],
            "a forwarded pointer forwards the write"
        );
        assert!(barrier.escapes && kills.escapes && calls_global.escapes);
    }

    /// The ray-query object is a write through its pointer, so a query on
    /// the function's own local is contained.
    #[test]
    fn a_ray_query_writes_its_query_object_only() {
        let (_, eff) = summaries(
            r#"
            enable wgpu_ray_query;
            @group(0) @binding(0) var acc: acceleration_structure;
            fn own() -> u32 {
                var rq: ray_query;
                rayQueryInitialize(&rq, acc, RayDesc(0u, 0xFFu, 0.0, 100.0, vec3f(0.0), vec3f(0.0, 0.0, 1.0)));
                rayQueryProceed(&rq);
                return rayQueryGetCommittedIntersection(&rq).kind;
            }
        "#,
        );
        assert!(eff[0].inline_pure());
    }

    /// A call reports the callee's writes on the caller's operands; a
    /// non-writing callee reports nothing.
    #[test]
    fn a_call_reports_what_its_callee_writes() {
        let (module, eff) = summaries(
            r#"
            @group(0) @binding(0) var<storage, read_write> buf: array<i32, 4>;
            fn reads(p: ptr<function, i32>) -> i32 { return *p; }
            fn writes(p: ptr<function, i32>) { *p = 1; }
            fn global() { buf[0] = 1; }
            fn caller() { var v = 0; let r = reads(&v); writes(&v); global(); }
        "#,
        );
        let (_, caller) = module.functions.iter().nth(3).expect("four functions");
        let mut seen = Vec::new();
        for stmt in caller.body.iter() {
            if let naga::Statement::Call { arguments, .. } = stmt {
                let mut effects = Vec::new();
                statement_effects(stmt, &mut |f| eff[f.index()], &mut |e| {
                    effects.push(match e {
                        Effect::Write(h) => format!(
                            "write arg{}",
                            arguments.iter().position(|&a| a == h).expect("an argument")
                        ),
                        Effect::ImageWrite(_) => "image".into(),
                        Effect::Globals => "globals".into(),
                        Effect::Observable => "observable".into(),
                    })
                });
                seen.push(effects.join(","));
            }
        }
        assert_eq!(seen, ["", "write arg0", "globals"]);
    }
}
