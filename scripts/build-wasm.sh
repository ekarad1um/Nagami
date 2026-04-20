#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/pkg"

RUSTFLAGS="--remap-path-prefix=$HOME=~ --remap-path-prefix=$(rustc --print sysroot)=rustc" \
CARGO_PROFILE_RELEASE_OPT_LEVEL=z \
  wasm-pack build "$ROOT" --release \
    --target web -- --no-default-features --features wasm

node -e "
const fs = require('fs');
const path = require('path');
const pkg = JSON.parse(fs.readFileSync(path.join('$PKG', 'package.json'), 'utf8'));

pkg.name = 'nagami-rs';
pkg.keywords = ['wgsl','shader','minify','minifier','webgpu','naga','optimizer'];
pkg.files = ['nagami_bg.wasm','nagami_bg.wasm.d.ts','nagami.js','nagami.d.ts'];
pkg.exports = { '.': { types: './nagami.d.ts', import: './nagami.js' } };
pkg.sideEffects = false;
pkg.engines = { node: '>=16' };

delete pkg._dependencies;

fs.writeFileSync(
  path.join('$PKG', 'package.json'),
  JSON.stringify(pkg, null, 2) + '\n'
);
"
