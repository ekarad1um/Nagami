//! One classification of every [`naga::Expression`] variant: each "is this
//! pure / const / emitted / relocatable / expensive / convergent" question
//! is a mask over one exhaustive table, so no two consumers can disagree
//! about a variant (the `select` merge once speculated a texture load the
//! loop pin refused to sink), and where two questions want different
//! answers for one variant the difference is a named mask.
//!
//! Nearly every predicate is asked of a single node on an arena being
//! rewritten (`const_fold` folds in place, the merges append), so
//! [`ExprClass::node`] is a pure per-node function; [`Classes`] adds the
//! cone bits and is valid only for the arena it was computed over.

use std::ops::{BitOr, Index};

/// Bit set over the classes of one expression node: the kinds partition the
/// variants (one per node), the facets refine a kind, the cone bits summarise
/// the operand cone ([`Classes`] only).
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub(crate) struct ExprClass(u32);

impl BitOr for ExprClass {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl ExprClass {
    // Kinds.
    /// `Literal` / `Constant` / `Override` / `ZeroValue`: spelled from the
    /// declaration, never emitted, a const-expression on its own.
    pub const CONST_LEAF: Self = Self(1 << 0);
    pub const FN_ARG: Self = Self(1 << 1);
    pub const GLOBAL_REF: Self = Self(1 << 2);
    pub const LOCAL_REF: Self = Self(1 << 3);
    /// Structural and arithmetic wrappers whose value is a function of the
    /// operands alone: `Compose` `Access` `AccessIndex` `Splat` `Swizzle`
    /// `Unary` `Binary` `Select` `Relational` `Math` `As`.
    pub const PURE_OP: Self = Self(1 << 4);
    pub const LOAD: Self = Self(1 << 5);
    /// `ImageSample` / `ImageLoad` / `ImageQuery`.
    pub const IMAGE_OP: Self = Self(1 << 6);
    pub const DERIVATIVE: Self = Self(1 << 7);
    pub const ARRAY_LENGTH: Self = Self(1 << 8);
    /// A statement's result slot (`CallResult`, `AtomicResult`,
    /// `WorkGroupUniformLoadResult`, `RayQueryProceedResult`, the subgroup
    /// results): no text of its own, exists only at its statement.
    pub const STMT_RESULT: Self = Self(1 << 9);
    /// A ray-query read (`RayQueryGetIntersection`, `RayQueryVertexPositions`):
    /// emitted, yet tied to a cursor that does not relocate.
    pub const QUERY_STATE: Self = Self(1 << 10);
    /// `CooperativeLoad` / `CooperativeMultiplyAdd`: lane state.
    pub const COOPERATIVE: Self = Self(1 << 11);

    // Facets.
    /// The `Override` leaf: const to the passes (a slot made of one moves its
    /// error to pipeline creation), opaque to the emitter's hazard guard.
    pub const OVERRIDE: Self = Self(1 << 12);
    /// `&&` / `||`: the right operand is evaluated conditionally.
    pub const SHORT_CIRCUIT: Self = Self(1 << 13);
    /// Float `-x` `x*y` `x/y` `x%y`: the value depends on const-ness
    /// (`is_sign_sensitive_op`).
    pub const SIGN_SENSITIVE_OP: Self = Self(1 << 14);
    /// What tint's uniformity analysis constrains: a derivative or an
    /// implicit-LOD / bias non-gather sample (the front-end lowers gathers
    /// to `level: Zero`, so the gather clause only guards an upstream change).
    pub const CONVERGENT: Self = Self(1 << 15);

    // Cone bits, set by `Classes::of` only.
    /// The whole cone reads as a WGSL const-expression.
    pub const CONST_CONE: Self = Self(1 << 16);
    /// A `STMT_RESULT` sits in the cone.
    pub const STMT_RESULT_IN_CONE: Self = Self(1 << 17);
    /// An `EXPENSIVE` node sits in the cone.
    pub const EXPENSIVE_IN_CONE: Self = Self(1 << 18);

    // The questions, as masks.
    /// The evaluation a guard exists to save: kept at its count, its
    /// control dependence and its loop depth.
    pub const EXPENSIVE: Self = Self::IMAGE_OP;
    /// Produced without an `Emit`: declarations and statement results.
    pub const PRE_EMIT: Self = Self(
        Self::CONST_LEAF.0
            | Self::FN_ARG.0
            | Self::GLOBAL_REF.0
            | Self::LOCAL_REF.0
            | Self::STMT_RESULT.0,
    );
    /// Cannot be cloned into another function: a statement result, a cursor
    /// or lane read.  The inliner adds `LOCAL_REF` (a function-scoped slot),
    /// which `PURE_TO_CLONE` admits because a fold clones within one arena.
    pub const NOT_RELOCATABLE: Self =
        Self(Self::STMT_RESULT.0 | Self::QUERY_STATE.0 | Self::COOPERATIVE.0);
    /// Cheap to reference from many sites (`load_dedup`): pre-emit leaves and
    /// an already-emitted `Load`.  Disagrees with `PURE_TO_CLONE` by design:
    /// a forwarded `Load` is N references to one read, a cloned one a second
    /// read; a cloned `Binary` is fine, a forwarded one is not simple.
    pub const SIMPLE_FORWARD: Self = Self(
        Self::CONST_LEAF.0 | Self::FN_ARG.0 | Self::GLOBAL_REF.0 | Self::LOCAL_REF.0 | Self::LOAD.0,
    );
    /// A clone into another arena slot reproduces the value (`const_fold`).
    pub const PURE_TO_CLONE: Self = Self(
        Self::CONST_LEAF.0
            | Self::FN_ARG.0
            | Self::GLOBAL_REF.0
            | Self::LOCAL_REF.0
            | Self::PURE_OP.0,
    );
    /// Reordering past memory writes is unobservable (`call_inline`).  Holds
    /// `DERIVATIVE`, which `PURE_TO_CLONE` does not: a derivative reads no
    /// memory but must not be duplicated.
    pub const MEMORY_FREE_NODE: Self = Self(Self::PURE_TO_CLONE.0 | Self::DERIVATIVE.0);
    /// May not run where its guard no longer decides (`merge_trailing_returns`).
    pub const NOT_SPECULATABLE: Self =
        Self(Self::CONVERGENT.0 | Self::EXPENSIVE.0 | Self::QUERY_STATE.0 | Self::COOPERATIVE.0);

    /// The classes of one node.  Exhaustive with no `_` arm: a naga variant
    /// added upstream fails the build here, the single point of truth.
    pub fn node(expr: &naga::Expression) -> Self {
        use naga::Expression as E;
        match expr {
            E::Literal(_) | E::Constant(_) | E::ZeroValue(_) => Self::CONST_LEAF,
            E::Override(_) => Self::CONST_LEAF | Self::OVERRIDE,
            E::FunctionArgument(_) => Self::FN_ARG,
            E::GlobalVariable(_) => Self::GLOBAL_REF,
            E::LocalVariable(_) => Self::LOCAL_REF,
            E::Compose { .. }
            | E::Access { .. }
            | E::AccessIndex { .. }
            | E::Splat { .. }
            | E::Swizzle { .. }
            | E::Select { .. }
            | E::Relational { .. }
            | E::Math { .. }
            | E::As { .. } => Self::PURE_OP,
            E::Unary { op, .. } => {
                if matches!(op, naga::UnaryOperator::Negate) {
                    Self::PURE_OP | Self::SIGN_SENSITIVE_OP
                } else {
                    Self::PURE_OP
                }
            }
            E::Binary { op, .. } => match op {
                naga::BinaryOperator::Multiply
                | naga::BinaryOperator::Divide
                | naga::BinaryOperator::Modulo => Self::PURE_OP | Self::SIGN_SENSITIVE_OP,
                naga::BinaryOperator::LogicalAnd | naga::BinaryOperator::LogicalOr => {
                    Self::PURE_OP | Self::SHORT_CIRCUIT
                }
                _ => Self::PURE_OP,
            },
            E::Load { .. } => Self::LOAD,
            E::ImageSample { gather, level, .. } => {
                if gather.is_none()
                    && matches!(level, naga::SampleLevel::Auto | naga::SampleLevel::Bias(_))
                {
                    Self::IMAGE_OP | Self::CONVERGENT
                } else {
                    Self::IMAGE_OP
                }
            }
            E::ImageLoad { .. } | E::ImageQuery { .. } => Self::IMAGE_OP,
            E::Derivative { .. } => Self::DERIVATIVE | Self::CONVERGENT,
            E::ArrayLength(_) => Self::ARRAY_LENGTH,
            E::CallResult(_)
            | E::AtomicResult { .. }
            | E::WorkGroupUniformLoadResult { .. }
            | E::RayQueryProceedResult
            | E::SubgroupBallotResult
            | E::SubgroupOperationResult { .. } => Self::STMT_RESULT,
            E::RayQueryGetIntersection { .. } | E::RayQueryVertexPositions { .. } => {
                Self::QUERY_STATE
            }
            E::CooperativeLoad { .. } | E::CooperativeMultiplyAdd { .. } => Self::COOPERATIVE,
        }
    }

    /// Any bit of `mask` set.
    pub const fn any(self, mask: Self) -> bool {
        self.0 & mask.0 != 0
    }

    /// Const-ness of the node itself as a WGSL const-expression: `Some`
    /// decides, `None` defers to the operands (`const_expression_leaf`).
    pub const fn const_leaf(self) -> Option<bool> {
        if self.any(Self::CONST_LEAF) {
            Some(true)
        } else if self.any(Self::PURE_OP) {
            None
        } else {
            Some(false)
        }
    }
}

/// [`ExprClass`] of every handle of one arena, cone bits included.  Valid
/// only while the arena is unmodified: a memo read on a rewritten arena is
/// the one way this module can lie.
pub(crate) struct Classes {
    bits: Vec<ExprClass>,
}

impl Classes {
    /// One forward pass (children precede parents in a naga arena).
    pub fn of(arena: &naga::Arena<naga::Expression>) -> Self {
        let mut bits: Vec<ExprClass> = Vec::with_capacity(arena.len());
        for (_, expr) in arena.iter() {
            let node = ExprClass::node(expr);
            let mut const_cone = node.const_leaf().unwrap_or(true);
            let mut cone = ExprClass::default();
            if node.any(ExprClass::STMT_RESULT) {
                cone = cone | ExprClass::STMT_RESULT_IN_CONE;
            }
            if node.any(ExprClass::EXPENSIVE) {
                cone = cone | ExprClass::EXPENSIVE_IN_CONE;
            }
            crate::ir::visit::visit_expression_children(expr, |child| {
                let c = bits[child.index()];
                if node.const_leaf().is_none() {
                    const_cone &= c.any(ExprClass::CONST_CONE);
                }
                cone = cone
                    | ExprClass(
                        c.0 & (ExprClass::STMT_RESULT_IN_CONE.0 | ExprClass::EXPENSIVE_IN_CONE.0),
                    );
            });
            if const_cone {
                cone = cone | ExprClass::CONST_CONE;
            }
            bits.push(node | cone);
        }
        Self { bits }
    }
}

impl Classes {
    /// The classes of `h`, `None` for a handle appended after the pass.
    pub fn get(&self, h: naga::Handle<naga::Expression>) -> Option<ExprClass> {
        self.bits.get(h.index()).copied()
    }
}

impl Index<naga::Handle<naga::Expression>> for Classes {
    type Output = ExprClass;
    fn index(&self, h: naga::Handle<naga::Expression>) -> &ExprClass {
        &self.bits[h.index()]
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use naga::Expression as E;

    /// One sample per variant, with the parameters the classes read
    /// (sample level and gather, the operators, `As` conversion, the opaque
    /// math functions) spread over their values.
    fn samples() -> Vec<E> {
        let mut types = naga::UniqueArena::<naga::Type>::default();
        let ty = types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Scalar(naga::Scalar::F32),
            },
            naga::Span::UNDEFINED,
        );
        let mut exprs = naga::Arena::<E>::new();
        let h = exprs.append(E::Literal(naga::Literal::F32(0.0)), naga::Span::UNDEFINED);
        let mut constants = naga::Arena::<naga::Constant>::new();
        let c = constants.append(
            naga::Constant {
                name: None,
                ty,
                init: h,
            },
            naga::Span::UNDEFINED,
        );
        let mut overrides = naga::Arena::<naga::Override>::new();
        let o = overrides.append(
            naga::Override {
                name: None,
                id: None,
                ty,
                init: None,
            },
            naga::Span::UNDEFINED,
        );
        let mut globals = naga::Arena::<naga::GlobalVariable>::new();
        let g = globals.append(
            naga::GlobalVariable {
                name: None,
                space: naga::AddressSpace::Private,
                binding: None,
                ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::UNDEFINED,
        );
        let mut locals = naga::Arena::<naga::LocalVariable>::new();
        let l = locals.append(
            naga::LocalVariable {
                name: None,
                ty,
                init: None,
            },
            naga::Span::UNDEFINED,
        );
        let mut functions = naga::Arena::<naga::Function>::new();
        let f = functions.append(naga::Function::default(), naga::Span::UNDEFINED);

        let binary_ops = {
            use naga::BinaryOperator::*;
            let ops = [
                Add,
                Subtract,
                Multiply,
                Divide,
                Modulo,
                Equal,
                NotEqual,
                Less,
                LessEqual,
                Greater,
                GreaterEqual,
                And,
                ExclusiveOr,
                InclusiveOr,
                LogicalAnd,
                LogicalOr,
                ShiftLeft,
                ShiftRight,
            ];
            // Exhaustive, so a new operator lands in the table.
            for op in ops {
                match op {
                    Add | Subtract | Multiply | Divide | Modulo | Equal | NotEqual | Less
                    | LessEqual | Greater | GreaterEqual | And | ExclusiveOr | InclusiveOr
                    | LogicalAnd | LogicalOr | ShiftLeft | ShiftRight => {}
                }
            }
            ops
        };
        let levels = [
            naga::SampleLevel::Auto,
            naga::SampleLevel::Zero,
            naga::SampleLevel::Exact(h),
            naga::SampleLevel::Bias(h),
            naga::SampleLevel::Gradient { x: h, y: h },
        ];

        let mut out = vec![
            E::Literal(naga::Literal::F32(1.0)),
            E::Constant(c),
            E::Override(o),
            E::ZeroValue(ty),
            E::Compose {
                ty,
                components: vec![h],
            },
            E::Access { base: h, index: h },
            E::AccessIndex { base: h, index: 0 },
            E::Splat {
                size: naga::VectorSize::Bi,
                value: h,
            },
            E::Swizzle {
                size: naga::VectorSize::Bi,
                vector: h,
                pattern: [naga::SwizzleComponent::X; 4],
            },
            E::FunctionArgument(0),
            E::GlobalVariable(g),
            E::LocalVariable(l),
            E::Load { pointer: h },
            E::ImageLoad {
                image: h,
                coordinate: h,
                array_index: None,
                sample: None,
                level: None,
            },
            E::ImageQuery {
                image: h,
                query: naga::ImageQuery::NumLevels,
            },
            E::Select {
                condition: h,
                accept: h,
                reject: h,
            },
            E::Derivative {
                axis: naga::DerivativeAxis::X,
                ctrl: naga::DerivativeControl::Coarse,
                expr: h,
            },
            E::Relational {
                fun: naga::RelationalFunction::Any,
                argument: h,
            },
            E::Math {
                fun: naga::MathFunction::Sin,
                arg: h,
                arg1: None,
                arg2: None,
                arg3: None,
            },
            E::Math {
                fun: naga::MathFunction::Unpack4x8snorm,
                arg: h,
                arg1: None,
                arg2: None,
                arg3: None,
            },
            E::As {
                expr: h,
                kind: naga::ScalarKind::Float,
                convert: None,
            },
            E::As {
                expr: h,
                kind: naga::ScalarKind::Float,
                convert: Some(4),
            },
            E::CallResult(f),
            E::AtomicResult {
                ty,
                comparison: false,
            },
            E::WorkGroupUniformLoadResult { ty },
            E::ArrayLength(h),
            E::RayQueryVertexPositions {
                query: h,
                committed: true,
            },
            E::RayQueryProceedResult,
            E::RayQueryGetIntersection {
                query: h,
                committed: true,
            },
            E::SubgroupBallotResult,
            E::SubgroupOperationResult { ty },
            E::CooperativeLoad {
                columns: naga::CooperativeSize::Eight,
                rows: naga::CooperativeSize::Eight,
                role: naga::CooperativeRole::A,
                data: naga::CooperativeData {
                    pointer: h,
                    stride: h,
                    row_major: true,
                },
            },
            E::CooperativeMultiplyAdd { a: h, b: h, c: h },
        ];
        for op in binary_ops {
            out.push(E::Binary {
                op,
                left: h,
                right: h,
            });
        }
        for op in [
            naga::UnaryOperator::Negate,
            naga::UnaryOperator::LogicalNot,
            naga::UnaryOperator::BitwiseNot,
        ] {
            out.push(E::Unary { op, expr: h });
        }
        for gather in [None, Some(naga::SwizzleComponent::X)] {
            for level in levels {
                out.push(E::ImageSample {
                    image: h,
                    sampler: h,
                    gather,
                    coordinate: h,
                    array_index: None,
                    offset: None,
                    level,
                    depth_ref: None,
                    clamp_to_edge: false,
                });
            }
        }
        out
    }

    /// Every variant is sampled (34 discriminants: a naga upgrade that adds
    /// one fails `node`'s match first, then this count), every node has
    /// exactly one kind, and the facets sit on the kinds they refine.
    #[test]
    fn node_class_table() {
        let samples = samples();
        let discriminants: std::collections::HashSet<_> =
            samples.iter().map(std::mem::discriminant).collect();
        assert_eq!(discriminants.len(), 34);
        let kinds = [
            ExprClass::CONST_LEAF,
            ExprClass::FN_ARG,
            ExprClass::GLOBAL_REF,
            ExprClass::LOCAL_REF,
            ExprClass::PURE_OP,
            ExprClass::LOAD,
            ExprClass::IMAGE_OP,
            ExprClass::DERIVATIVE,
            ExprClass::ARRAY_LENGTH,
            ExprClass::STMT_RESULT,
            ExprClass::QUERY_STATE,
            ExprClass::COOPERATIVE,
        ];
        for expr in &samples {
            let class = ExprClass::node(expr);
            assert_eq!(
                kinds.iter().filter(|k| class.any(**k)).count(),
                1,
                "{expr:?}"
            );
            if class.any(ExprClass::OVERRIDE) {
                assert!(matches!(expr, E::Override(_)));
            }
            if class.any(ExprClass::SHORT_CIRCUIT | ExprClass::SIGN_SENSITIVE_OP) {
                assert!(class.any(ExprClass::PURE_OP), "{expr:?}");
            }
            if class.any(ExprClass::CONVERGENT) {
                assert!(
                    class.any(ExprClass::DERIVATIVE | ExprClass::IMAGE_OP),
                    "{expr:?}"
                );
            }
            assert!(
                !class.any(
                    ExprClass::CONST_CONE
                        | ExprClass::STMT_RESULT_IN_CONE
                        | ExprClass::EXPENSIVE_IN_CONE
                ),
                "cone bits are Classes::of's: {expr:?}"
            );
        }
        let convergent = samples
            .iter()
            .filter(|e| ExprClass::node(e).any(ExprClass::CONVERGENT))
            .count();
        // dpdx plus the gather-less Auto and Bias samples.
        assert_eq!(convergent, 3);
    }

    /// The cone bits over a small arena: `a * (b + c)` with `a` a literal
    /// and `c` a call result.
    #[test]
    fn classes_pin_the_cones() {
        let mut arena = naga::Arena::<E>::new();
        let span = naga::Span::UNDEFINED;
        let a = arena.append(E::Literal(naga::Literal::F32(2.0)), span);
        let b = arena.append(E::Literal(naga::Literal::F32(3.0)), span);
        let mut functions = naga::Arena::<naga::Function>::new();
        let f = functions.append(naga::Function::default(), span);
        let c = arena.append(E::CallResult(f), span);
        let sum = arena.append(
            E::Binary {
                op: naga::BinaryOperator::Add,
                left: b,
                right: c,
            },
            span,
        );
        let ab = arena.append(
            E::Binary {
                op: naga::BinaryOperator::Add,
                left: a,
                right: b,
            },
            span,
        );
        let product = arena.append(
            E::Binary {
                op: naga::BinaryOperator::Multiply,
                left: a,
                right: sum,
            },
            span,
        );
        let classes = Classes::of(&arena);
        assert!(classes[a].any(ExprClass::CONST_CONE));
        assert!(classes[ab].any(ExprClass::CONST_CONE));
        assert!(!classes[c].any(ExprClass::CONST_CONE));
        assert!(!classes[sum].any(ExprClass::CONST_CONE));
        assert!(!classes[product].any(ExprClass::CONST_CONE));
        assert!(classes[c].any(ExprClass::STMT_RESULT_IN_CONE));
        assert!(classes[product].any(ExprClass::STMT_RESULT_IN_CONE));
        assert!(!classes[ab].any(ExprClass::STMT_RESULT_IN_CONE));
        assert!(!classes[product].any(ExprClass::EXPENSIVE_IN_CONE));
        assert!(classes[product].any(ExprClass::SIGN_SENSITIVE_OP));
    }
}
