//! Tests covering the generator's dead-declaration pruning for
//! constants, structs, and globals.  The assertions treat
//! reachability (not textual absence) as the contract, which leaves
//! room for naga's inliner to additionally drop references the
//! generator would otherwise keep.

use super::helpers::*;

// MARK: Dead constant elimination

#[test]
fn dead_constant_is_omitted() {
    // DEAD is never referenced from any live code.
    // naga may inline USED into the function body as a literal, making it
    // dead too - that's fine; the key assertion is that DEAD is gone.
    let src = r#"
        const USED: array<f32, 2> = array<f32, 2>(1.0, 2.0);
        const DEAD: array<f32, 2> = array<f32, 2>(3.0, 4.0);
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(USED[0]);
        }
    "#;
    let out = compact(src);
    assert!(
        !out.contains("DEAD"),
        "dead constant must be omitted: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn multiple_dead_constants_all_omitted() {
    // Several dead constants of various types - none should appear.
    let src = r#"
        const D1: f32 = 1.0;
        const D2: vec3f = vec3f(1.0, 2.0, 3.0);
        const D3: array<f32, 3> = array<f32, 3>(1.0, 2.0, 3.0);
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(0.0);
        }
    "#;
    let out = compact(src);
    assert!(!out.contains("D1"), "dead D1 must be omitted: {out}");
    assert!(!out.contains("D2"), "dead D2 must be omitted: {out}");
    assert!(!out.contains("D3"), "dead D3 must be omitted: {out}");
    // The output should only be the entry point.
    assert_valid_wgsl(&out);
}

#[test]
fn constant_from_global_var_init_is_live() {
    let src = r#"
        const INIT_VAL: f32 = 42.0;
        const DEAD: f32 = 0.0;
        var<private> g: f32 = INIT_VAL;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(g);
        }
    "#;
    let out = compact(src);
    assert!(
        out.contains("INIT_VAL") || out.contains("42"),
        "constant used in global var init must be live: {out}"
    );
    assert!(
        !out.contains("DEAD"),
        "dead constant must be omitted: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn library_module_preserves_all_constants() {
    // No entry points -> library module; all constants should survive.
    let src = r#"
        const A: f32 = 1.0;
        const B: f32 = 2.0;
        fn helper() -> f32 { return A; }
    "#;
    let out = compact(src);
    assert!(out.contains('A'), "library constant A must survive: {out}");
    assert!(out.contains('B'), "library constant B must survive: {out}");
    assert_valid_wgsl(&out);
}

// MARK: Dead struct elimination

#[test]
fn dead_struct_is_omitted() {
    let src = r#"
        struct Used { x: f32 }
        struct Dead { y: f32 }
        @group(0) @binding(0) var<uniform> u: Used;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(u.x);
        }
    "#;
    let out = compact(src);
    assert!(out.contains("Used"), "live struct must appear: {out}");
    assert!(!out.contains("Dead"), "dead struct must be omitted: {out}");
    assert_valid_wgsl(&out);
}

#[test]
fn struct_used_only_by_dead_constant_is_omitted() {
    let src = r#"
        struct LiveS { x: f32 }
        struct DeadS { a: f32, b: f32 }
        const DEAD: DeadS = DeadS(1.0, 2.0);
        @group(0) @binding(0) var<uniform> u: LiveS;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(u.x);
        }
    "#;
    let out = compact(src);
    assert!(out.contains("LiveS"), "live struct must appear: {out}");
    assert!(
        !out.contains("DeadS"),
        "struct used only by dead constant must be omitted: {out}"
    );
    assert!(
        !out.contains("DEAD"),
        "dead constant must be omitted: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn library_module_preserves_all_structs() {
    let src = r#"
        struct A { x: f32 }
        struct B { y: f32 }
        fn helper(a: A) -> f32 { return a.x; }
    "#;
    let out = compact(src);
    assert!(out.contains('A'), "library struct A must survive: {out}");
    assert!(out.contains('B'), "library struct B must survive: {out}");
    assert_valid_wgsl(&out);
}

// MARK: Combined DCE

#[test]
fn dead_constant_and_its_struct_both_omitted() {
    let src = r#"
        struct Piece { pos: vec2f, rot: f32 }
        const LAYOUT: array<Piece, 2> = array<Piece, 2>(
            Piece(vec2f(0.0, 0.0), 0.0),
            Piece(vec2f(1.0, 1.0), 1.57),
        );
        const ALIVE: f32 = 6.28;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(ALIVE);
        }
    "#;
    let out = compact(src);
    assert!(!out.contains("Piece"), "dead struct must be omitted: {out}");
    assert!(
        !out.contains("LAYOUT"),
        "dead constant must be omitted: {out}"
    );
    assert!(
        out.contains("ALIVE") || out.contains("6.28"),
        "live constant must survive: {out}"
    );
    assert_valid_wgsl(&out);
}

#[test]
fn preserve_symbols_keeps_dead_constant() {
    let src = r#"
        const KEEP_ME: f32 = 1.0;
        const DROP_ME: f32 = 2.0;
        @fragment fn fs() -> @location(0) vec4f {
            return vec4f(0.0);
        }
    "#;
    let out = compact_mangled_preserved(src, &["KEEP_ME"]);
    assert!(
        out.contains("KEEP_ME"),
        "preserved constant must survive even when dead: {out}"
    );
    assert!(
        !out.contains("DROP_ME"),
        "non-preserved dead constant must be omitted: {out}"
    );
    assert_valid_wgsl(&out);
}
