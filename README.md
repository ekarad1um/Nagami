# Nagami[n]

[Naga + Minify](https://nagami.0xfff8e7.dev/), a tiny and self-contained compiler that shrinks your WGSL shaders - not by squishing text, but by understanding them.

Nagami lowers WGSL into [Naga IR](https://github.com/gfx-rs/wgpu/tree/trunk/naga), optimizes the IR in multiple passes (typically converges in 3 sweeps), and emits the smallest valid WGSL it can.

## What it does

Working on typed IR rather than the source text lets Nagami run real compiler-level WGSL optimizations with correctness and integrity guarantees:

- **Dead code elimination** - unused declarations and parameters, dead stores (`var x = a; x = b` -> `var x = b`), dead branches, anything after `return`/`break`/`discard`, empty `if` and degenerate `switch`
- **Constant folding and algebraic simplification** - constants (`1.0 + 2.0` -> `3`, `x * 1` -> `x`), splats (`vec3(x) * v` -> `x * v`), swizzles (`vec3(v.x, v.y, v.z)` -> `v.xyz`, `v.xy` on a `vec2` -> `v`), zero values (`vec3f(0, 0, 0)` -> `vec3f()`, `vec4f(0, 0, 0, 2)` -> `vec4f(vec3f(), 2)`), matrices (`mat2x2f(vec2f(a, b), vec2f(c, d))` -> `mat2x2f(a, b, c, d)`)
- **Redundancy elimination** - duplicate pure expressions share one evaluation (CSE), redundant loads merge, non-overlapping locals share one slot, member-wise struct builds become one constructor (`t.a = x; t.b = y; return t` -> `return T(x, y)`), repeated vector literals and scalars become one shared `const`
- **Inlining and forward substitution** - small helpers into their callers, single-use `let`s into their use
- **Control-flow restructuring** - `for` loops rebuilt from `loop`/`continuing`, `&&`/`||` rebuilt from Naga's `if` chains, compound assignment (`x = x * y` -> `x *= y`, `x = x + 1` -> `x++`), flipped branches (`if c {} else { x; }` -> `if !c { x; }`), `else` dropped after a terminator
- **Mangling and lexical minimization** - identifiers renamed by frequency (`myLongVariableName` -> `a`), `alias T = vec3f;` when it pays for itself, redundant type annotations dropped (`var`/`const` types, array constructor types), a `let` introduced only where binding is cheaper than repeating the expression, shortest literal form (`1048576f` -> `0x1p20f`), only the parentheses precedence requires
- **Float precision reduction** - cap decimal places or significant figures, per type (lossy, opt-in)
- **Preamble** - external declarations used for parsing and optimization, excluded from the output
- **Library modules** - shader fragments without entry points keep every declaration
- **Name map** - original -> final identifier mapping for hosts that address shaders by source names

## Getting started

Install CLI with cargo (or use as a Rust/WASM library, see bottom):

```sh
cargo install nagami
```

Example usage:

```sh
nagami shader.wgsl -o shader.min.wgsl               # minify (max profile by default)
nagami shader.wgsl --in-place --stats               # in-place, print savings
nagami shader.wgsl -o out.wgsl -p baseline          # DCE + folding only
cat shader.wgsl | nagami - > out.wgsl               # stdin -> stdout
nagami shader.wgsl --check                          # exit 1 if not minified
nagami shader.wgsl --preamble env.wgsl -o out.wgsl  # external declarations
nagami shader.wgsl -o out.wgsl --name-map map.json  # original -> final identifier map
nagami shader.wgsl --format json                    # one JSON document on stdout
nagami shader.wgsl -o out.wgsl --strict-fallback    # fail instead of shipping a text-only bailout
nagami shader.wgsl --sig-figs 4 -o out.wgsl         # lossy: cap significant figures (or --decimal-places N)
```

## Profiles

Three profiles control which IR passes run. Generator-level rewrites (folding, control flow, naming and spelling) are applied in every profile.

| | `baseline` | `aggressive` | **`max`** |
|---|:---:|:---:|:---:|
| Dead code elimination, constant folding, dead parameters, emit merge | ✓ | ✓ | ✓ |
| Renaming of globals, functions, params, locals | ✓ | ✓ | ✓ |
| Function inlining (nodes / call sites) | - | 24 / 3 | 48 / 6 |
| Load dedup, dead stores, variable coalescing, struct-build coalescing | - | ✓ | ✓ |
| CSE, vector-constant hoisting | - | - | ✓ |
| Mangling of struct types and members, constants, overrides | - | - | ✓ |

The `baseline` is fast and safe; `aggressive` adds the full IR pipeline without mangling; `max` raises the inlining limits and enables CSE and vector-constant hoisting (both only while mangling is on). `--no-mangle` disables mangling in any profile, `--mangle` enables it.

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

Install with cargo:

```sh
cargo add nagami --no-default-features
```

```rust
let config = nagami::config::Config {
    preamble: Some(preamble_src.to_string()),  // optional
    ..Default::default()
};
let output = nagami::run(src, &config)?;
println!("{}", output.source);
```

## Use in JavaScript / TypeScript

Install with npm:

```sh
npm install nagami-rs
```

Browser / bundler (config is optional, all fields have defaults):

```js
import init, { run } from 'nagami-rs';
await init();                          // load the WASM module once
const { source, report, nameMap } = run(shader, {
  profile: 'max',                      // "baseline" | "aggressive" | "max" (default)
  mangle: true,                        // also rename struct types/members, constants, overrides (default: on for "max")
  preserveSymbols: ['main'],           // names to keep untouched
  preamble: preambleSrc,               // external declarations, stripped from the output
  floatPrecision: 6,                   // N decimal places for all float kinds (lossy, opt-in);
                                       // also { decimalPlaces: 6 }, { significantFigures: 4 }, or per type { f32: 6 }
  maxInlineNodeCount: 48,              // inlining budget per function
  maxInlineCallSites: 6,               // max call sites a function may have and still inline
  beautify: false, indent: 2,          // indented output
  validateEachPass: false,             // re-validate WGSL after every pass
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
