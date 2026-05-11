---
name: rust-code-simplification
description: Simplifies large-scale, complex Rust codebases for clarity, consistency, and maintainability without changing behavior. Use when refactoring multi-crate workspaces, taming accumulated complexity in async/unsafe/lock-free/DSP/mmap subsystems, removing duplicated code and tests, and tightening rustdoc — while strictly preserving public APIs, semver, performance characteristics, and `unsafe` invariants.
---

# Rust Code Simplification

## Overview

Simplify Rust code by reducing complexity while **preserving exact observable behavior, public API shape, semver guarantees, performance characteristics, and `unsafe` soundness**. The goal is never fewer lines — it is code a new contributor can read, modify, and prove correct faster than the original.

Every change must pass three tests:

1. **Behavior test** — `cargo test`, `cargo clippy -D warnings`, `cargo doc`, and (where applicable) `miri` / `loom` / benchmarks all pass without modification.
2. **Comprehension test** — a competent Rust engineer unfamiliar with the module reads the diff and says "this is clearer."
3. **Soundness test** — no `# Safety` invariant is weakened; no `Send`/`Sync` boundary is loosened; no atomic ordering is relaxed without proof.

If any test fails, revert.

## When to Use

- A crate compiles and tests pass, but a module has accumulated layers of `match`, nested `Result`, hand-rolled state machines, or trait spaghetti.
- Public API is stable, but internals have grown organically through several feature additions.
- During PR review when readability, clippy noise, or duplication is flagged.
- When `cargo udeps` / `cargo machete` show dead code, or duplicated tests have appeared across `tests/` and `#[cfg(test)] mod tests`.
- When the workspace `Cargo.toml`, `[features]`, or trait hierarchies have drifted out of consistency.

## When NOT to Use

- Code is already idiomatic and clean — do not refactor for its own sake.
- You do not yet understand the module's invariants, especially in `unsafe`, atomics, or FFI code.
- The "simpler" form would be measurably slower in a hot path (verify with `criterion`, not intuition).
- The crate is scheduled for replacement or a major rewrite.
- The change would alter the public API of a published crate without an accompanying semver bump and changelog entry.

---

## CRITICAL EXECUTION DIRECTIVE

This codebase is at peak complexity. The following discipline is non-negotiable.

### 1. Extreme meticulousness & 360° thinking

For every proposed change, enumerate:

- **Compile-time effects**: trait resolution, inference, monomorphization, feature gates, MSRV.
- **Runtime effects**: allocations, branch prediction, cache locality, atomic ordering, panic paths.
- **Soundness effects**: `unsafe` invariants, aliasing (`&` vs `&mut`), `Pin` projections, `Send`/`Sync` auto-traits, drop order.
- **API effects**: public items, re-exports, sealed traits, `#[non_exhaustive]`, semver category (major/minor/patch).
- **Ecosystem effects**: downstream crates in the workspace, `cargo doc` examples, `#[doc(test)]` snippets.

Leave no stone unturned. If you cannot enumerate the above for a hunk, you are not ready to change it.

### 2. Strict hierarchical, step-by-step execution

Work in phases. **Complete each phase entirely** — and verify it — before starting the next. Do not interleave. Do not skip. Do not batch.

- Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 (defined below).
- Within Phase 3, apply **one simplification per commit**. No exceptions.

### 3. Mandatory sub-agent model tier for hazardous domains

When spawning sub-agents or parallelizing analysis of any of the domains listed in [Hazardous Domains](#hazardous-domains-require-frontier-tier-sub-agents), you **must** instruct the orchestration system to use the **most capable frontier-tier reasoning model available**. Faster/smaller models are forbidden in those domains. The cost of a subtle UB or memory-ordering regression vastly exceeds the cost of a more expensive inference call.

---

## The Five Principles (Rust Edition)

### 1. Preserve behavior exactly

For every change, answer:

- Same outputs for every input, including `panic`, `Err`, and iterator laziness?
- Same side effects in the same order (logs, file I/O, atomic stores, channel sends)?
- Same `Drop` order? (Reordering bindings can change `Drop` order — verify.)
- Same observable allocation profile in hot paths? (Replacing `Vec` with `SmallVec` or vice versa is a behavior change for memory-bound callers.)
- Same `Send`/`Sync` and `Unpin` properties on public types?
- Same `unsafe` invariants documented and upheld?

If unsure, do not change.

### 2. Follow project conventions

Before touching anything, ingest:

- `rust-toolchain.toml` — pinned channel and components.
- `rustfmt.toml` and `clippy.toml` — style and lint config.
- Workspace `Cargo.toml` — `[workspace.package]`, `[workspace.dependencies]`, lints table.
- `CONTRIBUTING.md`, `ARCHITECTURE.md`, `CLAUDE.md`, or equivalent.
- Neighboring modules — match their patterns for error types, module layout, re-exports, naming.

Specifically match:

- Error strategy: `thiserror` per crate vs `anyhow` at boundaries vs hand-rolled.
- Async runtime conventions: `tokio` vs `async-std` vs runtime-agnostic.
- Trait sealing patterns, builder patterns, type-state usage.
- Module visibility (`pub(crate)` vs `pub(super)` vs `pub`).
- Doc comment conventions (`# Errors`, `# Panics`, `# Safety`, `# Examples`).

Inconsistency dressed as "simplification" is churn.

### 3. Prefer clarity over cleverness

Compact ≠ simple in Rust, where one symbol can hide a lot.

```rust
// UNCLEAR: combinator chain that hides the failure mode
let name = config
    .get("user")
    .and_then(|u| u.as_table())
    .and_then(|t| t.get("name"))
    .and_then(|n| n.as_str())
    .map(str::to_owned)
    .unwrap_or_default();

// CLEAR: explicit, with named intermediate and a real fallback
fn user_name(config: &Table) -> String {
    let Some(user) = config.get("user").and_then(Value::as_table) else {
        return String::new();
    };
    user.get("name")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_default()
}
```

```rust
// UNCLEAR: nested match for a Result<Option<T>, E>
match repo.find(id) {
    Ok(Some(user)) => match user.validate() {
        Ok(()) => Ok(user),
        Err(e) => Err(Error::Invalid(e)),
    },
    Ok(None) => Err(Error::NotFound(id)),
    Err(e) => Err(Error::Db(e)),
}

// CLEAR: `?` plus explicit not-found, one level of nesting
let user = repo.find(id).map_err(Error::Db)?
    .ok_or(Error::NotFound(id))?;
user.validate().map_err(Error::Invalid)?;
Ok(user)
```

### 4. Maintain balance — over-simplification traps in Rust

- **Inlining a `fn` that named a domain concept** ("normalize_sample_rate") makes the call site harder to read even if it was one line.
- **Collapsing multiple `match` arms into a combinator chain** that requires the reader to mentally re-expand it.
- **Removing a newtype** because "it's just a `u32`" — the newtype was the invariant.
- **Replacing `for` loops with iterator chains** in hot DSP loops where LLVM auto-vectorizes the loop but not the chain. Measure.
- **Genericizing a function** to "remove duplication" between two call sites — `<T: Trait>` adds monomorphization cost and reader cost; two `fn`s may be simpler.
- **Replacing explicit `Drop` ordering** with implicit drops when ordering matters for locks or FFI handles.
- **Removing a `#[inline]` or `#[cold]` hint** because "it looks like noise" — those are perf contracts.

### 5. Scope discipline

Default to simplifying the modules touched in the current task. Do not drive-by-refactor unrelated code. If a broader refactor is warranted, file it as a separate issue with rationale and scope.

**The Rule of 500**: if a refactor would touch >500 lines or >10 files, write a `cargo fix`-style migration, a `syn`-based codemod, or a `sed`/`tree-sitter` script. Manual edits at that scale are error-prone and unreviewable.

---

## The Simplification Process

### Phase 0 — Workspace survey (read-only)

Before any change:

1. `cargo metadata --format-version=1 | jq` — enumerate crates, features, dependency graph.
2. Identify the **public API surface** of each crate (`cargo public-api` if available; otherwise grep `pub fn`, `pub struct`, `pub trait`, `pub use`).
3. Locate `unsafe` blocks (`rg -n '\bunsafe\b'`) and FFI boundaries (`extern "C"`, `#[link]`, `bindgen`).
4. Locate atomics, locks, channels (`AtomicU*`, `Ordering::`, `Mutex`, `RwLock`, `parking_lot`, `crossbeam`, `tokio::sync`).
5. Locate macros (`macro_rules!`, `#[proc_macro*]`).
6. Note feature flags and `cfg` gates.
7. Note MSRV (`rust-version` in `Cargo.toml`).
8. Run baseline: `cargo +<pinned> check --workspace --all-features`, `cargo test --workspace`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, `cargo doc --workspace --no-deps`.

If baseline is red, **stop**. You cannot distinguish your regressions from pre-existing ones. Fix or document baseline first.

### Phase 1 — Per-module comprehension (Chesterton's Fence)

For each target module, before changing anything, answer:

- What is this module's single responsibility?
- What are its public items? Its sealed/internal items?
- What invariants does each `unsafe` block require? Are they documented in `# Safety`?
- For atomics: what is the synchronization protocol? Which `Ordering` pairs with which?
- For `async`: what is `Send`? What holds `.await` points? Any `!Send` futures?
- For traits: who implements them? Are they sealed? Object-safe? Used as `dyn`?
- What does `git log -p` / `git blame` say about the contentious sections?
- What tests pin the behavior? Are there `proptest`/`quickcheck`/fuzz targets?
- If found any issue in the original code, in-place track with a `FIXME: ` tag in code comments.

If any answer is "I don't know," read more before touching.

### Phase 2 — Catalog simplification opportunities

Scan with the [Rust Simplification Catalog](#rust-simplification-catalog) below. Produce a written list of candidate changes, **ranked by risk (low → high)** and **grouped by independence**. Do not start applying yet.

### Phase 3 — Apply incrementally

For each candidate, in low-risk → high-risk order:

1. Make exactly one change.
2. `cargo check --workspace --all-features` (and per relevant feature combo).
3. `cargo clippy --workspace --all-targets --all-features -- -D warnings`.
4. `cargo test --workspace` (plus targeted `--features` runs).
5. For `unsafe` / atomics / `Pin` changes: `cargo +nightly miri test -p <crate>` and/or `loom` model tests.
6. For perf-sensitive changes: `cargo bench` (criterion) before/after.
7. For public API changes: `cargo semver-checks`.
8. If green → commit with a focused message. If red → revert and reconsider.

**Never batch.** A failing batch hides which change broke what.

**Separate refactors from feature/bug work.** A PR doing both is two PRs.

### Phase 4 — Holistic verification

After all changes:

- Diff review: is each hunk independently justifiable?
- Did any `# Safety`, `# Errors`, `# Panics` doc become stale?
- Did public API change? If yes: changelog + semver bump + migration note.
- Did MSRV requirements move? (e.g., adopting `let-else` requires 1.65, `let chains` requires 1.65 stable form, etc.)
- Did feature-gated code paths still build under all relevant feature combinations? Test the matrix or use `cargo hack --feature-powerset`.
- Are benchmarks within tolerance?

If any "simplification" produced harder-to-read or harder-to-verify code, **revert it specifically** — partial wins are still wins.

---

## Rust Simplification Catalog

Each entry is a concrete pattern, the signal that flags it, and a directional fix. None is a license to refactor without reading the surrounding code.

### Control flow & error handling

| Pattern | Signal | Simplification |
|---|---|---|
| Nested `match` on `Result`/`Option` (3+ levels) | Hard to follow happy path | Use `?`, `let-else`, and combinators in moderation |
| Hand-written `match` that mirrors `map` / `and_then` / `ok_or` | Verbose but does nothing custom | Replace with the combinator |
| Combinator chain that reaches 5+ links | Reader must mentally unroll | Split with named bindings or extract a helper |
| `if let Some(x) = opt { ... } else { return ... }` | Pyramid of doom | `let Some(x) = opt else { return ... };` |
| `unwrap()` / `expect()` in library code | Hidden panic surface | Return `Result`; add `# Errors` doc |
| `panic!` used for control flow | Misuses unwinding | Convert to `Result` or typed error |
| Hand-rolled error enum that duplicates `std::io::Error` chains | Boilerplate | `thiserror` with `#[from]` |
| `Box<dyn Error>` everywhere in a library | Loses type info | Crate-local error enum at the public boundary |
| `anyhow!` inside library crates | Wrong layer | `anyhow` at binary boundary; `thiserror` in libs |

Note: It is acceptable to leave `unwrap()` in `#[cfg(test)]` blocks or integration tests, as test panics are the intended behavior for failures.

### Iterators, collections, and loops

| Pattern | Signal | Simplification |
|---|---|---|
| Manual index loops (`for i in 0..v.len()`) | C-style | `for x in &v` / `iter().enumerate()` |
| Manual `Vec` build inside a `for` | Imperative collect | `.iter().filter(..).map(..).collect()` — **but measure in hot loops** |
| Multiple passes over the same collection | Wasted work | Single iterator chain, or `fold` |
| Collecting then re-iterating | Allocation churn | Keep as iterator if downstream is single-pass |
| `clone()` in a hot path | Allocation | Borrow, `Cow`, or restructure ownership |
| `String + &str + &str` chains | Multiple allocations | `format!` or `String::with_capacity` + `push_str` |
| `HashMap::insert` in a loop computing counts | Verbose | `*map.entry(k).or_insert(0) += 1` |

> **Hot-path caveat**: in DSP / numerics, an explicit `for` loop over a `&[f32]` can vectorize where an iterator chain does not. Always check with `cargo asm` or `criterion` before "simplifying" inner loops.

### Types, generics, traits

| Pattern | Signal | Simplification |
|---|---|---|
| Verbose `where` clauses in function signature | Reader fatigue | Move to `where` block; consider trait alias (nightly) or helper trait |
| `<T: Trait1 + Trait2 + Trait3 + ...>` repeated across many fns | Duplication | Define a sealed marker trait that bundles them |
| `impl Trait` in argument position when generic isn't needed | Hides nothing | `impl Trait` is fine; only switch to named generic if turbofish callers need it |
| `Box<dyn Trait>` where a generic suffices | Heap + dyn dispatch | Generic `<T: Trait>` if call sites are few; keep `dyn` if many |
| Newtype wrapper that adds nothing | Pure noise | Inline — **unless** it carries an invariant or `Drop` semantics |
| Newtype that *does* carry an invariant but isn't enforced in constructor | Latent bug | Make field private; force construction via `fn new(..) -> Result<Self, _>` |
| Explicit lifetimes that elision would handle | Visual noise | Remove per [lifetime elision rules](https://doc.rust-lang.org/reference/lifetime-elision.html) |
| Phantom generics no caller uses | Dead complexity | Remove (after `cargo public-api` confirms no semver impact) |
| Trait with one impl, never used as `dyn` | Premature abstraction | Inline into the concrete type |
| Builder pattern for a struct with 2 fields | Over-engineered | Plain `Self { a, b }` or `fn new(a, b)` |

### Macros

| Pattern | Signal | Simplification |
|---|---|---|
| `macro_rules!` that expands to a single `fn` call | Macro adds nothing | Replace with the function |
| Proc-macro generating boilerplate that `derive` could do | Custom where standard exists | Use `#[derive]` or established crate (`derive_more`, `strum`) |
| Declarative macro with cascading `tt` munching for what a generic could do | Macro abuse | Generic function or trait |
| Macro hygiene workarounds (`$crate::` everywhere, ad hoc `__priv` modules) | Hard to read | Often correct — **leave alone** unless you fully understand hygiene |

### Async

| Pattern | Signal | Simplification |
|---|---|---|
| `async fn foo() -> Result<T> { bar().await }` (pure forward) | Useless wrapper | Return the future: `fn foo() -> impl Future<Output = Result<T>> { bar() }` — only if `Send`/lifetime bounds match |
| `tokio::spawn` with `move` capture that clones everything | Heavy | Restructure to share `Arc` or pass owned values |
| `Mutex` (sync) held across `.await` | Likely bug, not just style | Use `tokio::sync::Mutex`, or restructure to release before await |
| Manual `Future` impl where a combinator suffices | Boilerplate | `async {}` block or `futures` combinators |
| `block_on` inside async context | Deadlock risk | Restructure call graph |
| `select!` with hand-rolled cancellation | Error-prone | `tokio_util::sync::CancellationToken` or `select!` with `biased;` |

### Concurrency, atomics, locks

> Treat every change here as Phase-3-high-risk. See [Hazardous Domains](#hazardous-domains-require-frontier-tier-sub-agents).

| Pattern | Signal | Simplification (with caution) |
|---|---|---|
| `Ordering::SeqCst` everywhere | Over-strong by default | Downgrade only with a written acquire/release argument and `loom` test coverage |
| `Mutex<()>` used as a lock with no data | Encodes a critical section in the wrong primitive | Move the data inside the `Mutex`, or use a typed lock guard |
| `Arc<Mutex<T>>` where `&T` would do | Shared mutability not actually needed | Restructure ownership; `Arc<T>` if read-only |
| Hand-rolled spinlock | Almost always wrong | Use `parking_lot` or `std::sync` |
| Channel + flag combo that re-implements `oneshot` | Reinvention | `tokio::sync::oneshot` or `crossbeam_channel::bounded(1)` |
| Lock-free structure with no `loom` tests | Untestable invariant | Add `loom` tests *before* simplifying |

### `unsafe` and FFI

| Pattern | Signal | Simplification |
|---|---|---|
| Large `unsafe` block covering mostly safe code | Hides the actual unsafe op | Shrink to the minimal expression; document `# Safety` |
| Repeated `unsafe { ptr::read(p) }` patterns | Duplication of subtle invariants | Extract a single safe wrapper with a documented `# Safety` contract on its private constructor; expose only safe API |
| Missing `# Safety` doc on `unsafe fn` | Soundness debt | Add the contract, or re-examine why it's `unsafe` |
| `transmute` where a safe cast or `bytemuck` works | Sledgehammer | `as`, `From`/`Into`, `bytemuck::cast`, `zerocopy` |
| `mem::uninitialized` (deprecated) | UB risk | `MaybeUninit` |
| FFI types not `#[repr(C)]` | Latent UB | Add `#[repr(C)]` and audit field order |
| `bindgen` output edited by hand | Drift on regeneration | Move tweaks into `build.rs` configuration |

### `mmap` and zero-copy

| Pattern | Signal | Simplification |
|---|---|---|
| Reading `mmap`'d bytes via `unsafe { *(ptr as *const T) }` | Alignment + endianness UB | `bytemuck::from_bytes` with `Pod`/`Zeroable`, or `zerocopy::FromBytes` |
| `&[u8]` parsed by ad hoc offset arithmetic | Off-by-one farm | `nom`, `winnow`, or a typed view struct |
| `mmap` lifetime tied to nothing | Dangling slices | Encode lifetime via a guard struct that owns the mapping |

### DSP / numerics

| Pattern | Signal | Simplification |
|---|---|---|
| Magic floating-point constants | Origin unclear | Named `const`, with derivation in doc comment |
| Hand-unrolled SIMD without explanation | Unreviewable | Either use `std::simd` / `wide` / `pulp`, or document the unroll factor and target CPU |
| Mixed `f32` / `f64` accumulators in long sums | Numerical error | Promote accumulator to `f64`; document |
| Iterator chains in inner FIR/IIR loop | May defeat vectorization | Keep explicit `for` loop; benchmark before changing |
| `assert!` in inner loop | Throughput hit | `debug_assert!` with a comment justifying the invariant |

### Module structure & workspace

| Pattern | Signal | Simplification |
|---|---|---|
| `mod.rs` files mixed with `foo.rs`/`foo/` style | Inconsistency | Pick one (project convention wins) |
| Re-exports scattered across submodules | Confusing public surface | Centralize in `lib.rs` with a documented `prelude` |
| Many `pub use` items that are not part of the intended API | Accidental API | Tighten to `pub(crate)` |
| Dependency duplicated across crate `Cargo.toml`s | Drift risk | `[workspace.dependencies]` + `package.dep = { workspace = true }` |
| Lints redefined per crate | Inconsistency | `[workspace.lints]` (Cargo 1.74+) |
| Feature flags that combine nontrivially | Combinatorial bugs | Document each feature; test with `cargo hack --feature-powerset --depth 2` |

---

## Redundancy Removal

### Duplicate code

1. Run `rg --pcre2 -U` for repeated 5+-line blocks; or use a clone detector (`similarity-rs`, `tokei`+grep heuristics).
2. For each clone cluster, decide:
   - **Extract a function** if the bodies are identical modulo arguments.
   - **Extract a trait** if the bodies are identical modulo a type.
   - **Extract a macro** *only* if neither function nor trait works (rare; requires repeating syntax shape, not values).
   - **Leave alone** if the clones are coincidental and likely to diverge.
3. Place extractions at the **lowest common module** that all callers can reach.
4. Run `cargo udeps` and `cargo machete` after extraction to catch newly-dead deps.

### Duplicate tests

1. Identify tests that differ only by input/expected:
   - Convert to **table-driven**: `for (input, expected) in CASES { assert_eq!(f(input), expected); }`.
   - Or use **`rstest`** with `#[case]` for parameterization plus per-case fixture injection.
2. Identify tests that share setup:
   - Extract a `fn fixture() -> Fixture` in a `#[cfg(test)] mod common;` (or a `tests/common/mod.rs` for integration tests).
   - For async setup, use `rstest`'s async fixtures or a `OnceCell`-backed lazy init.
3. Identify property-style assertions reimplemented across tests:
   - Consolidate into a single `proptest!` or `quickcheck` target.
4. Identify duplicated golden/snapshot data:
   - Move to `tests/fixtures/` and load once.
5. **Never delete a test without confirming** another test asserts the same property. Map each removed assertion to its surviving home.

### Dead code

- `cargo +nightly udeps --workspace --all-targets` — unused dependencies.
- `cargo machete` — fast unused-dependency check.
- `RUSTFLAGS="-W dead_code" cargo check --workspace --all-targets --all-features` — dead items, **per feature combination** (an item dead under one feature may be live under another).
- Confirm with `git log -S` that the dead item isn't conditionally re-enabled in a known-dormant branch.

---

## Comments & Rustdoc Quality

> The following adapts the project's general comment guidelines to Rust's `///`, `//!`, and `//` conventions. The principles are unchanged; the surface forms are Rust-specific.

### Principles

1. **Code-aligned accuracy**: every `///` block must reflect current behavior, including edge cases and invariants. After any refactor, re-read every doc comment in the touched file.
2. **Necessity check**: before keeping or adding a comment, ask whether it carries information not already evident from well-named code, types, and signatures. If not, **delete**, do not paraphrase. Silence beats noise.
3. **Why and how, not what**: explain intent and architecture. `// increment counter` above `count += 1` is forbidden. `// Retry: upstream returns 503 under load; see issue #142` is mandatory.
4. **Concise and dense**: prefer one well-crafted sentence over several short ones. Do not split a single thought across sentences solely to satisfy Markdown line-break rules. Hack-level density, accurate, no hedging, no colloquialism.
5. **Future-proof**: name invariants, not line numbers or sibling functions. "The buffer is always 2× sample-rate samples" survives refactoring; "see line 142" does not.

### Rust-specific rustdoc structure

For every `pub` item, use the standard sections **only when they carry information**:

```rust
/// Decodes a frame from `src` into `dst`.
///
/// Returns the number of input bytes consumed.
///
/// # Errors
///
/// Returns [`DecodeError::Truncated`] if `src` ends mid-frame, and
/// [`DecodeError::InvalidHeader`] if the magic bytes do not match.
///
/// # Panics
///
/// Panics if `dst.len() < FRAME_SIZE`. Callers should pre-size `dst`.
///
/// # Safety
///
/// (only on `unsafe fn`) Caller must ensure `src` is aligned to
/// `align_of::<u32>()` and remains valid for the duration of the call.
pub fn decode(src: &[u8], dst: &mut [Sample]) -> Result<usize, DecodeError> { ... }
```

Rules:

- Omit a section if it would be empty or trivial.
- `# Safety` is **mandatory** on every `unsafe fn`. No exceptions.
- `# Panics` is mandatory if the function can panic on caller-controlled inputs.
- `# Errors` documents *which* error variants and *why*, not just "returns an error if it fails."
- `# Examples` should compile (`cargo test --doc`). If it can't, mark `ignore` and explain why.
- Use intra-doc links (`[`Type`]`, `[`module::fn`]`) instead of bare names.
- For internal items, prefer `//` line comments; reserve `///` for items intended for `cargo doc`.
- Module-level `//!` documents the module's responsibility and any invariants the module as a whole maintains.

### Anti-patterns

- "Increment counter" / "Loop over items" / "Return result" — restating the code.
- "TODO: fix this" with no issue link or context — either link an issue or do it.
- Commented-out code — delete; git remembers.
- Doc comments that drifted from the signature — delete or fix immediately on sight.
- `# Safety` blocks that say "this is safe because the caller ensures it" without naming what — useless; specify the invariant.

---

## Hazardous Domains (require frontier-tier sub-agents)

When analyzing or modifying any of these, the orchestration system **must** dispatch the most capable frontier reasoning model available. Faster/smaller models are forbidden here.

1. **`unsafe` and UB**: aliasing, provenance, `Pin` projections, `MaybeUninit`, raw pointer arithmetic, custom `Drop` interacting with `unsafe`.
2. **Atomics & memory ordering**: anything beyond `Relaxed` load/store, custom synchronization protocols, ABA scenarios.
3. **Lock-free data structures**: queues, stacks, hazard pointers, epoch-based reclamation (`crossbeam-epoch`), RCU-style schemes.
4. **`async` runtime internals**: custom `Future`, `Waker`, `Pin` projections, executors, `!Send` futures crossing task boundaries.
5. **DSP & numeric algorithms**: filter design, FFT, fixed-point, accumulator precision, denormal handling, SIMD correctness across targets.
6. **`mmap` / zero-copy / FFI**: alignment, endianness, lifetime of mapped regions, ABI/`repr` correctness, callbacks across language boundaries.
7. **Macro internals**: hygiene, span manipulation, proc-macro determinism, `$crate` paths in re-exported macros.
8. **Public API & semver**: any change that touches a `pub` item in a published crate.

In these domains: **plan in writing, run `miri`/`loom`/`cargo-fuzz`/benchmarks, justify ordering and invariants in the commit message.**

---

## Verification Checklist (Rust toolchain)

After every Phase-3 change, **all** must hold; after Phase-4, re-run the full set:

- [ ] `cargo +<pinned-toolchain> check --workspace --all-targets --all-features`
- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- [ ] `cargo test --workspace --all-features` (and any non-default feature combos in CI)
- [ ] `cargo doc --workspace --no-deps --all-features` with `RUSTDOCFLAGS="-D warnings"`
- [ ] `cargo test --doc --workspace`
- [ ] Feature-matrix: `cargo hack --feature-powerset --depth 2 check` (or project-defined matrix)
- [ ] `cargo +nightly miri test -p <crate>` — for every crate touched in `unsafe`/atomics/`Pin` paths
- [ ] `loom` model tests — for every lock-free or custom-sync change
- [ ] `cargo bench` — within tolerance for hot paths (criterion comparison report)
- [ ] `cargo semver-checks check-release` — no unintended public API change
- [ ] `cargo udeps` and `cargo machete` — no new unused deps; previously-unused removed
- [ ] `cargo deny check` — licenses, advisories, bans unaffected
- [ ] MSRV preserved: `cargo +<msrv> check --workspace`
- [ ] No `# Safety`, `# Errors`, `# Panics` doc became stale
- [ ] No comment paraphrases the code; remaining comments justify *why*
- [ ] Each commit is one simplification, independently revertable
- [ ] Diff has no unrelated reformatting (rustfmt-only changes belong in their own commit)

---

## Common Rationalizations (Rust flavor)

| Rationalization | Reality |
|---|---|
| "It compiles, so it's fine." | Type-checking does not prove `Send`/`Sync` correctness across `await`, atomic ordering correctness, or `unsafe` invariants. |
| "Iterators are always idiomatic." | In hot DSP loops, an explicit `for` over a `&[f32]` may auto-vectorize where a chain does not. Measure. |
| "`SeqCst` is just safer." | It is correct more often, but it imposes global ordering. If `Acquire`/`Release` suffice, `SeqCst` is over-strong and hides the actual protocol from the reader. |
| "Generic everything for flexibility." | Each generic instantiation costs compile time and binary size. Unused flexibility is debt. |
| "We can shrink this `unsafe` block later." | Later never comes, and meanwhile the block hides safe code under an `unsafe` umbrella, weakening review. |
| "The lint is annoying; allow it." | `#[allow(clippy::...)]` is a contract with future readers. Justify in a comment, or fix the cause. |
| "I'll fix the doc after the refactor." | Stale docs are worse than no docs. Update in the same commit. |
| "Just one more cleanup while I'm here." | Scope creep. File a follow-up. |
| "Tests are duplicative; delete the older ones." | First confirm each assertion has a surviving home. Otherwise you are deleting coverage. |
| "The original author must have had a reason." | Sometimes. Check `git blame` and PR history. But accumulated complexity often has no reason — it is residue of iteration under pressure. Apply Chesterton's Fence honestly: understand, then decide. |

---

## Red Flags — stop immediately if you see these in your own work

- A simplification requires editing a test to keep it green.
- The "simpler" version is longer, or harder to follow.
- You weakened, removed, or broadened an `unsafe` block's `# Safety` contract.
- You downgraded an atomic `Ordering` without a written argument and `loom` coverage.
- You removed an `#[inline]`, `#[cold]`, `#[repr(...)]`, or `#[track_caller]` attribute because it "looked unnecessary."
- You changed a public item's signature, name, or visibility without a semver review.
- You batched multiple simplifications into one commit.
- You touched code outside the task's scope.
- You cannot articulate, in one sentence, why each remaining comment exists.

---

## Summary Decision Flow

```
Encounter Rust code that feels complex
        │
        ▼
Phase 0: Survey workspace; baseline must be green
        │
        ▼
Phase 1: Comprehend module; answer Chesterton's Fence questions
        │
        ▼
Phase 2: Catalog candidates; rank by risk; group by independence
        │
        ▼
Phase 3: For each candidate (low → high risk):
            apply ─► verify (check/clippy/test/miri/loom/bench)
                        │
                ┌───────┴───────┐
              green             red
                │                │
              commit          revert + reconsider
        │
        ▼
Phase 4: Holistic review; semver/changelog/MSRV/feature matrix
        │
        ▼
Done — or revert any net-negative change
```

If at any step you are not sure: **stop, read more, or escalate to a frontier-tier sub-agent**. The cost of a subtle regression in this codebase exceeds the cost of patience.
