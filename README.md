# Nagami[n]

[Naga + Minify](https://nagami.0xfff8e7.dev/), shrinks your WGSL shaders - not by squishing text, but by understanding them.

Nagami lowers WGSL into [Naga IR](https://github.com/gfx-rs/wgpu/tree/trunk/naga), optimizes the IR in multiple passes, and emits the smallest valid WGSL it can.

**What it does that grep-and-replace can't:**

- Dead code elimination - unused declarations and parameters vanish
- Constant folding - `1.0 + 2.0` -> `3.0`, `x * 1` -> `x`
- Function inlining - small helpers absorbed into callers
- CSE - duplicate pure expressions share one evaluation
- Load dedup & dead stores - redundant reads merge, `var x = a; x = b` -> `var x = b`
- Variable coalescing - non-overlapping locals share one slot
- Struct construction - `t.a = x; t.b = y; return t` -> `return T(x, y)`
- Vector-constant hoisting - a vector literal repeated across functions -> one shared `const`
- Literal extraction - a repeated scalar constant -> one shared `const`
- For-loop reconstruction - `loop`/`break`/`continuing` -> `for`
- Branch flipping - `if c {} else { x; }` -> `if !c { x; }`
- Short-circuit re-sugaring - Naga's lowered `if`/`else` chains fold back into `&&`/`||`
- Else block elision - `if c { return; } else { x; }` -> `if c { return; } x;`
- Dead code after terminators - unreachable past `return`/`break`/`discard` stripped
- Empty construct removal - vacuous `if` and degenerate `switch` vanish
- Compound assignment - `x = x + 1` -> `x += 1`
- Splat elision - `vec3(x) * v` -> `x * v`
- Swizzle coalescing - `vec3(v.x, v.y, v.z)` -> `v.xyz`
- Identity-swizzle elision - `v.xy` on a `vec2` -> `v`
- Zero-value construction - `vec3f(0, 0, 0)` -> `vec3f()`
- Sub-vector splatting - `vec4f(0, 0, 0, 2)` -> `vec4f(vec3f(), 2)`
- Matrix flattening - `mat2x2f(vec2f(a, b), vec2f(c, d))` -> `mat2x2f(a, b, c, d)`
- Precedence-aware parens - only necessary parentheses survive
- Identifier mangling - `myLongVariableName` -> `a`
- Type aliasing - `vec3f` used five times -> `alias T = vec3f;`
- Type elision - redundant `var`/`const` type annotations stripped
- Shortest literal form - `1048576f` -> `0x1p20f`
- Float precision trimming - cap decimal places or significant figures, per type (lossy, opt-in)
- Preamble support - external declarations excluded from output
- Library modules - shader fragments without entry points preserved

Runs passes in fixed-point sweeps until the output stops shrinking. Typically converges in 3 sweeps.

## CLI

Install with cargo:

```sh
cargo install nagami --features cli
```

Example usage:

```sh
nagami shader.wgsl -o shader.min.wgsl               # minify (max profile by default)
nagami shader.wgsl --in-place --stats               # in-place, print savings
nagami shader.wgsl -o out.wgsl -p baseline          # lighter touch, no mangle
cat shader.wgsl | nagami - > out.wgsl               # stdin -> stdout
nagami shader.wgsl --check                          # exit 1 if not minified
nagami shader.wgsl --preamble env.wgsl -o out.wgsl  # external declarations
nagami shader.wgsl --decimal-places 6 -o out.wgsl   # lossy: cap fractional digits
nagami shader.wgsl --sig-figs 4 -o out.wgsl         # lossy: cap significant figures
```

## Profiles

Three optimization profiles control which IR passes run. Generator-level optimizations (for-loop reconstruction, swizzle coalescing, identity-swizzle elision, splat elision, zero-value construction, matrix flattening, sub-vector splatting, array-type elision, compound assignment, type elision, branch flipping, precedence-aware parens, shortest literal form, cost-aware let binding, type aliasing, literal extraction) are always applied regardless of profile.

| Optimization | `baseline` | `aggressive` | **`max`** |
|---|:---:|:---:|:---:|
| Dead code elimination | ✓ | ✓ | ✓ |
| Constant folding | ✓ | ✓ | ✓ |
| Dead parameter elimination | ✓ | ✓ | ✓ |
| Emit merge | ✓ | ✓ | ✓ |
| Rename (preserve names) | ✓ | ✓ | ✓ |
| Function inlining | - | ✓ (24 nodes / 3 call sites) | ✓ (48 nodes / 6 call sites) |
| Load dedup + dead stores | - | ✓ | ✓ |
| Variable coalescing | - | ✓ | ✓ |
| Struct-build coalescing | - | ✓ | ✓ |
| Vector-constant hoisting | - | ✓ | ✓ |
| Common subexpression elim | - | - | ✓ |
| Identifier mangling | - | - | ✓ |

Passes run in fixed-point sweeps (up to 16) until the output stops shrinking. `baseline` is fast and safe; `aggressive` adds the full IR pipeline without mangling; `max` enables CSE and raises inlining limits for maximum compression.

## Preamble

Some shader playgrounds (Shadertoy-style) inject uniform bindings and structs at runtime. Your shader code references them but doesn't define them. Pass these external declarations as a **preamble** - Nagami will prepend them for parsing and optimization, then strip them from the final output.

```wgsl
// preamble.wgsl
struct Inputs { time: f32, size: vec2f, mouse: vec4f, }
@group(0) @binding(0) var<uniform> inputs: Inputs;
```

```sh
nagami shader.wgsl --preamble preamble.wgsl -o out.wgsl
```

Preamble names are automatically preserved from renaming so that member access expressions (e.g. `inputs.time`) remain valid.

## Use in Rust

Install with cargo:

```sh
cargo add nagami
```

Run with default config:

```rust
let output = nagami::run(src, &nagami::config::Config::default())?;
println!("{}", output.source); // smol shader
```

With a preamble (external declarations excluded from output):

```rust
let config = nagami::config::Config {
    preamble: Some(preamble_src.to_string()),
    ..Default::default()
};
let output = nagami::run(src, &config)?;
```

## Use in JavaScript / TypeScript

Install with npm:

```sh
npm install nagami-rs
```

Browser / bundler:

```js
import init, { run } from 'nagami-rs';
await init(); // load the WASM module once
const { source, report } = run(shader);
console.log(source); // minified WGSL
console.log(report); // optimization report
```

With config (all fields optional):

```js
const { source, report } = run(shader, {
  profile: 'max',             // "baseline" | "aggressive" | "max" (default)
  mangle: true,               // rename identifiers (default: on for "max")
  preserveSymbols: ['main'],  // names to keep untouched
  beautify: false,            // compact output (default: false)
  indent: 2,                  // spaces per level when beautify is true
  floatPrecision: 6,          // shorthand: N decimal places, all float kinds (lossy, opt-in)
                              // also accepts { decimalPlaces: 6 }, { significantFigures: 4 },
                              // or per-type: { f32: 6, f64: { significantFigures: 12 } }
  maxInlineNodeCount: 48,     // inlining budget per function
  maxInlineCallSites: 6,      // inlining budget per call site
  preamble: preambleSrc,      // external decls prepended for parsing, stripped from output
  validateEachPass: false,    // re-validate WGSL after every pass
});
```

Node.js (synchronous init):

```js
import { readFileSync } from 'node:fs';
import { initSync, run } from 'nagami-rs';
const wasm = readFileSync(new URL('nagami_bg.wasm', import.meta.resolve('nagami-rs')));
initSync({ module: wasm });
const { source, report } = run(shader); // or with config as above
```

## License

```
MIT License - Copyright (c) 2026 ekarad1ium
```
