//! Structurally equal values in one function.  The front-end shares no
//! tree (each spelling of `a*b+c` is its own expression, and a spliced
//! callee brings its own copies of the caller's), so the census aliases
//! every later twin to the first spelling of its value: the binder prices
//! the value by the uses of all of them, and a twin renders as the first's
//! `let` name once it has one (`FunctionCtx::name_twins`), as itself
//! otherwise.
//!
//! Exact by construction: a twin is the same pure operation over the same
//! operand VALUES - an emitted operand is one handle (a `Load` reads at
//! its `Emit`, a statement result at its statement, an image or derivative
//! op is never a candidate), a pre-emit operand one value (a literal's
//! bits, a constant, an argument) - and the first spelling dominates the
//! twin: the map is block-scoped, and a loop's `continuing` starts afresh,
//! since the `for` header renders it ahead of the body's `let`s.  Constant
//! cones are `const_hoist`'s, keyed by value across the module.

use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHasher};

use crate::analysis::{Classes, ExprClass};
use crate::generator::expr_emit::{compose_is_splat, matrix_flatten_scalars};
use crate::handle_set::HandleMap;
use crate::passes::expr_util::{KeyToken, lit_key};

type Handle = naga::Handle<naga::Expression>;

/// The value of `expr` as tokens: a pre-emit leaf by what it names, a
/// pure operation by its kind and fields, then its operands by class.
/// `false` for a node that is a value of its own (a load, a statement
/// result, an image or derivative op).  `SwizzleComponent` is
/// `#[repr(u8)]`, so `as u8` is exact.
fn tokens(expr: &naga::Expression, class: &[Handle], out: &mut Vec<KeyToken>) -> bool {
    use naga::Expression as E;
    let cls = |h: Handle| (0u8, class[h.index()].index() as u64);
    match *expr {
        E::Literal(l) => out.extend([(1, 0), lit_key(l)]),
        E::Constant(c) => out.push((2, c.index() as u64)),
        E::Override(o) => out.push((3, o.index() as u64)),
        E::ZeroValue(t) => out.push((4, t.index() as u64)),
        E::FunctionArgument(i) => out.push((5, u64::from(i))),
        E::Compose { ty, ref components } => {
            out.extend([(6, ty.index() as u64), (7, components.len() as u64)]);
            out.extend(components.iter().map(|&h| cls(h)));
        }
        E::Access { base, index } => out.extend([(8, 0), cls(base), cls(index)]),
        E::AccessIndex { base, index } => out.extend([(9, u64::from(index)), cls(base)]),
        E::Splat { size, value } => out.extend([(10, size as u64), cls(value)]),
        E::Swizzle {
            size,
            vector,
            pattern,
        } => {
            let pattern = pattern.iter().fold(0, |acc, p| acc << 8 | *p as u8 as u64);
            out.extend([(11, size as u64), (12, pattern), cls(vector)]);
        }
        E::Unary { op, expr } => out.extend([(13, op as u64), cls(expr)]),
        E::Binary { op, left, right } => out.extend([(14, op as u64), cls(left), cls(right)]),
        E::Select {
            condition,
            accept,
            reject,
        } => out.extend([(15, 0), cls(condition), cls(accept), cls(reject)]),
        E::Relational { fun, argument } => out.extend([(16, fun as u64), cls(argument)]),
        E::Math {
            fun,
            arg,
            arg1,
            arg2,
            arg3,
        } => {
            out.extend([(17, fun as u64), cls(arg)]);
            out.extend([arg1, arg2, arg3].map(|a| a.map_or((18, 0), cls)));
        }
        E::As {
            expr,
            kind,
            convert,
        } => {
            let convert = convert.map_or(0, |bytes| u64::from(bytes) + 1);
            out.extend([(19, kind as u64), (20, convert), cls(expr)]);
        }
        _ => return false,
    }
    true
}

/// The spellings in scope, by the hash of their value: one map type the
/// crate instantiates already, where a map per key type is a hash table
/// copy in the wasm build.  A bucket holds the spellings of one hash; the
/// log, what each scope inserted, which is the tail of every bucket it
/// touched since scopes nest.
#[derive(Default)]
struct Scope {
    buckets: FxHashMap<usize, Vec<Handle>>,
    log: Vec<usize>,
}

impl Scope {
    /// The spelling of `value` in scope, or `h` once inserted as it.
    fn first(
        &mut self,
        value: &[KeyToken],
        h: Handle,
        mut same: impl FnMut(Handle) -> bool,
    ) -> Handle {
        let mut hasher = FxHasher::default();
        value.hash(&mut hasher);
        let hash = hasher.finish() as usize;
        if let Some(first) = self
            .buckets
            .get(&hash)
            .and_then(|bucket| bucket.iter().copied().find(|&f| same(f)))
        {
            return first;
        }
        self.buckets.entry(hash).or_default().push(h);
        self.log.push(hash);
        h
    }

    fn checkpoint(&self) -> usize {
        self.log.len()
    }

    fn rollback(&mut self, checkpoint: usize) {
        for hash in self.log.drain(checkpoint..).rev() {
            self.buckets.get_mut(&hash).expect("logged").pop();
        }
    }
}

/// The census state: the class of each handle - the first spelling of its
/// value, itself until an earlier one is found - and the spellings in
/// scope.
struct Census<'a> {
    exprs: &'a naga::Arena<naga::Expression>,
    class: Vec<Handle>,
    scope: Scope,
    value: Vec<KeyToken>,
    other: Vec<KeyToken>,
}

impl Census<'_> {
    /// The class of `h`, which enters the scope where no spelling of its
    /// value is in it.
    fn classify(&mut self, h: Handle) -> Handle {
        let Self {
            exprs,
            class,
            scope,
            value,
            other,
        } = self;
        value.clear();
        if !tokens(&exprs[h], class, value) {
            return h;
        }
        let first = scope.first(value, h, |f| {
            other.clear();
            tokens(&exprs[f], class, other) && other == value
        });
        class[h.index()] = first;
        first
    }

    /// The twins among `candidate` handles of `block` and its nested
    /// blocks, into `twin`, the scope of each nested block undone after it.
    fn walk(
        &mut self,
        block: &naga::Block,
        candidate: &[bool],
        twin: &mut impl FnMut(Handle, Handle, &naga::Arena<naga::Expression>),
    ) {
        for stmt in block.iter() {
            if let naga::Statement::Emit(range) = stmt {
                for h in range.clone().filter(|&h| candidate[h.index()]) {
                    let first = self.classify(h);
                    if first != h {
                        twin(h, first, self.exprs);
                    }
                }
                continue;
            }
            let checkpoint = self.scope.checkpoint();
            for nested in crate::ir::visit::nested_blocks(stmt) {
                self.walk(nested, candidate, twin);
                self.scope.rollback(checkpoint);
            }
        }
    }
}

/// Every later twin of a value in scope, keyed by the twin, valued by the
/// first spelling, whose `counts` and `parens` then hold the twin's uses
/// too - less a use by a consumer that is itself a twin, which renders as
/// a name once its own first binds (the longer value binds first); the
/// render's measured counts settle the rest.  A value nothing renders is
/// neither a first nor a twin.
pub(super) fn structural_twins(
    func: &naga::Function,
    types: &naga::UniqueArena<naga::Type>,
    finfo: &naga::valid::FunctionInfo,
    counts: &mut [usize],
    parens: &mut [u16],
) -> HandleMap<naga::Expression, Handle> {
    use naga::Expression as E;
    let exprs = &func.expressions;
    let classes = Classes::of(exprs);
    // A value whose inline text is its consumer's choice, not its own, is
    // priced wrong by its `let` text: a splat elides to its scalar under a
    // mixed arithmetic operator, a column of a matrix the emitter flattens
    // renders as its lanes.  Neither is a candidate.
    let mut consumer_spelled = vec![false; exprs.len()];
    for (h, expr) in exprs.iter() {
        match *expr {
            E::Splat { .. } => consumer_spelled[h.index()] = true,
            E::Compose { ty, ref components } => {
                if components.len() > 1 && compose_is_splat(components, exprs, &|_| false) {
                    consumer_spelled[h.index()] = true;
                }
                if matrix_flatten_scalars(ty, components, types, exprs).is_some() {
                    for &c in components.iter() {
                        consumer_spelled[c.index()] = true;
                    }
                }
            }
            _ => {}
        }
    }
    let candidate: Vec<bool> = exprs
        .iter()
        .map(|(h, _)| {
            counts[h.index()] > 0
                && classes[h].any(ExprClass::PURE_OP)
                && !classes[h].any(ExprClass::CONST_CONE)
                && !consumer_spelled[h.index()]
                && !matches!(
                    finfo[h].ty.inner_with(types),
                    naga::TypeInner::Pointer { .. } | naga::TypeInner::ValuePointer { .. }
                )
        })
        .collect();
    let mut census = Census {
        exprs,
        class: exprs.iter().map(|(h, _)| h).collect(),
        scope: Scope::default(),
        value: Vec::new(),
        other: Vec::new(),
    };
    // Pre-emit leaves are in scope everywhere.
    for (h, expr) in exprs.iter() {
        if ExprClass::node(expr).any(ExprClass::CONST_LEAF | ExprClass::FN_ARG) {
            census.classify(h);
        }
    }
    let mut twins = HandleMap::default();
    census.walk(&func.body, &candidate, &mut |h, first, exprs| {
        twins.insert(h, first);
        counts[first.index()] += counts[h.index()];
        parens[first.index()] = parens[first.index()].saturating_add(parens[h.index()]);
        crate::ir::visit::visit_expression_children(&exprs[h], |operand| {
            if let Some(&f) = twins.get(operand) {
                counts[f.index()] = counts[f.index()].saturating_sub(1);
            }
        });
    });
    twins
}
