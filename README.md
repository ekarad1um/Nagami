# Nagami[n]

[Naga + Minify](https://nagami.0xfff8e7.dev/), a tiny and self-contained compiler that shrinks your WGSL shaders - not by squishing text, but by understanding them. Nagami lowers WGSL into [Naga IR](https://github.com/gfx-rs/wgpu/tree/trunk/naga), optimizes the IR in multiple passes and emits the smallest valid WGSL it can, which typically converges in 3 sweeps.

## What it does

Working on typed IR rather than the source text lets Nagami run real compiler-level WGSL optimizations with correctness and integrity guarantees:

- **Dead code elimination** - removes unused declarations and parameters, dead stores and branches, code after `return`/`break`/`discard`, empty `if` and degenerate `switch`
- **Constant folding and algebraic simplification** - evaluates constant expressions, removes algebraic identities and identity swizzles, turns splat operands into scalars and component-wise constructors into swizzles, spells zero-value and matrix constructors in their shortest form
- **Redundancy elimination** - merges redundant loads, packs non-overlapping locals into one slot, collapses member-wise struct builds into a single constructor, hoists repeated vector literals and scalars into one shared `const`
- **Inlining and forward substitution** - inlines small helpers into their callers and single-use `let`s into their use site
- **Control-flow restructuring** - rebuilds `for` loops from `loop`/`continuing` and `&&`/`||` from Naga's `if` chains, restores compound assignment and `++`/`--`, negates an empty `if` to keep only its `else`, drops `else` after a terminator, merges guarded early returns into one `select`
- **Mangling and lexical minimization** - renames identifiers by frequency, adds type aliases when they pay for themselves, drops redundant type annotations from `var`, `const` and array constructors, binds a `let` only where that beats repeating the expression, spells literals in their shortest form (hex floats included), keeps only the parentheses precedence requires
- **Float precision reduction** - caps decimal places or significant figures, overall or per float type (lossy, opt-in)
- **Preamble** - external declarations that take part in parsing and optimization but stay out of the output
- **Library modules** - a shader fragment without entry points keeps all its declarations
- **Name map** - original -> final identifier mapping for hosts that address shaders by source name

## Getting started

Install CLI with cargo (or use as a Rust/WASM library, see bottom):

```sh
cargo install nagami
```

Example usage:

```sh
nagami shader.wgsl -o shader.min.wgsl       # minify (max profile by default)
nagami shader.wgsl --in-place --stats       # in-place, print savings
nagami shader.wgsl -o out.wgsl -p baseline  # DCE + folding only
cat shader.wgsl | nagami - > out.wgsl       # stdin -> stdout
nagami shader.wgsl --check                  # exit 1 if not minified
nagami --help                               # for advanced usage
```

## Profiles

Three profiles control which IR passes run; `--mangle`/`--no-mangle` override the profile's choice. Generator-level rewrites (folding, control flow, naming and spelling) run in every profile.

| | `baseline` | `aggressive` | **`max`** |
|---|:---:|:---:|:---:|
| Dead code elimination, constant folding, dead parameters, emit merge | ✓ | ✓ | ✓ |
| Renaming of globals, functions, params, locals | ✓ | ✓ | ✓ |
| Single-call function splicing; multi-site inlining budget (nodes / call sites) | - | ✓ (24 / 3) | ✓ (48 / 6) |
| Load dedup, dead stores, variable coalescing, struct-build coalescing | - | ✓ | ✓ |
| Vector-constant hoisting (only with mangling) | - | - | ✓ |
| Mangling of struct types and members, constants, overrides | - | - | ✓ |

## Preamble

Some shader playgrounds (Shadertoy-style) inject uniform bindings and structs at runtime, so your shader references declarations it doesn't contain. Pass these external declarations as a *preamble*, Nagami prepends them for parsing and optimization, then strips them from the output.

```wgsl
// preamble.wgsl
struct Inputs { time: f32, size: vec2f, mouse: vec4f, }
@group(0) @binding(0) var<uniform> inputs: Inputs;
```

```sh
nagami shader.wgsl --preamble preamble.wgsl -o out.wgsl
```

## Use in Rust

Add the library without the CLI dependency:

```sh
cargo add nagami --no-default-features
```

```rust
use nagami::config::{Config, Profile};

let config = Config {
    profile: Profile::Max,
    ..Default::default() // extra: preserve_symbols, mangle, preamble, float_precision, beautify, indent, trace, etc.
};
let output = nagami::run(src, &config)?;
println!("{}", output.source);
```

## Use in JavaScript / TypeScript

Install with npm:

```sh
npm install nagami-rs
```

Browser / bundler (every config field is optional):

```js
import init, { run } from 'nagami-rs';
await init();                     // load the WASM module once
const { source, report, nameMap } = run(shader, {
  profile: 'max',                 // "baseline" | "aggressive" | "max" (default)
  mangle: true,                   // also rename struct types/members, constants, overrides (default on for "max")
  preserveSymbols: ['uniforms'],  // names to keep untouched (entry points always are)
  preserveInterface: false,       // keep bindings, overrides, their struct types and members
  preamble: preambleSrc,          // external declarations, stripped from the output
  floatPrecision: 6,              // N decimal places for all float kinds (lossy, opt-in), also accept { decimalPlaces: 6 }, { significantFigures: 4 }, or per type { f32: 6 }
  maxInlineNodeCount: 48,         // node budget for cloning a helper into several call sites (a single-call helper is always spliced)
  maxInlineCallSites: 6,          // max call sites a helper may have and still be cloned into them
  beautify: false, indent: 2,     // indented output
  validateEachPass: false,        // re-validate WGSL after every pass
  optBisectLimit: undefined,      // stop after N accepted pass changes
});
console.log(source);
```

Node.js 20.6+ (synchronous init):

```js
import { readFileSync } from 'node:fs';
import { initSync, run } from 'nagami-rs';
initSync({ module: readFileSync(new URL('nagami_bg.wasm', import.meta.resolve('nagami-rs'))) });
const { source, report } = run(shader);
```

## License

```
MIT License - Copyright (c) 2026 ekarad1ium
```
