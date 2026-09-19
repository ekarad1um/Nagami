//! `#[path]` child of `interface.rs`.

use super::*;
use crate::config::{Config, PrecisionMode};
use crate::io;

fn parsed(src: &str) -> naga::Module {
    let module = io::parse_wgsl(src).expect("parses");
    io::validate_module(&module).expect("validates");
    module
}

fn interface_of(src: &str, options: &Options) -> Interface {
    Interface::of(&parsed(src), options.clone())
}

/// The default reading: structure only, float values compared.
fn structural() -> Options {
    Options {
        float_values: true,
        ..Options::default()
    }
}

fn mismatch(before: &str, after: &str, options: &Options) -> Option<String> {
    interface_of(before, options).mismatch(&interface_of(after, options))
}

fn assert_same(before: &str, after: &str, options: &Options) {
    if let Some(diff) = mismatch(before, after, options) {
        panic!("interfaces differ:{diff}");
    }
}

fn assert_differs(before: &str, after: &str, options: &Options, expected: &[&str]) {
    let diff = mismatch(before, after, options).expect("interfaces should differ");
    for line in expected {
        assert!(diff.contains(line), "missing `{line}` in:{diff}");
    }
}

const RICH: &str = "
    diagnostic(off, derivative_uniformity);
    struct Light { color: vec3f, @align(16) radius: f32 }
    struct Camera { view: mat4x4f, lights: array<Light, 2>, count: u32 }
    struct Particles { data: array<vec4f> }
    struct VOut {
        @builtin(position) p: vec4f,
        @location(0) @interpolate(perspective) c: vec3f,
        @location(1) @interpolate(flat, first) k: u32,
        @location(2) @interpolate(linear, centroid) m: f32,
    }
    override BIAS: f32 = 1.0;
    @id(3) override SCALE: f32 = 2.0 * BIAS;
    override COUNT: u32;
    override ZERO: f32 = f32();
    const HALF = 0.5;
    @group(0) @binding(0) var<uniform> camera: Camera;
    @group(0) @binding(1) var<storage, read_write> particles: Particles;
    @group(1) @binding(0) var tex: texture_2d<f32>;
    @group(1) @binding(1) var samp: sampler;
    @group(1) @binding(2) var depth: texture_depth_2d;
    @group(1) @binding(3) var store: texture_storage_2d<rgba8unorm, write>;
    @group(2) @binding(0) var<uniform> unused_binding: vec4f;
    var<private> scratch: f32;
    var<workgroup> tile: array<f32, COUNT * 2u>;
    fn helper(x: f32, unused: f32) -> f32 { scratch += x; return scratch * HALF; }
    @vertex fn vs(@builtin(vertex_index) i: u32, @location(0) pos: vec4f) -> VOut {
        var v: VOut;
        v.p = pos * camera.view;
        v.k = i + camera.count + COUNT;
        v.m = helper(pos.x, pos.y) + ZERO;
        return v;
    }
    @fragment @early_depth_test(force) fn fs(v: VOut, @location(3) extra: f32) -> @location(0) vec4f {
        _ = depth;
        return textureSample(tex, samp, v.c.xy) * BIAS * SCALE + camera.lights[v.k % 2u].radius + extra;
    }
    @compute @workgroup_size(64, 1, 1) fn cs(@builtin(global_invocation_id) id: vec3u) {
        tile[id.x] = 1.0;
        particles.data[id.x] = vec4f(tile[0]);
        textureStore(store, vec2i(id.xy), vec4f(0.0));
    }
";

/// The same interface spelled the way the emitter would: renamed
/// declarations, short type spellings, default attributes omitted, the
/// unused binding and the private global gone, the helper spliced, the
/// zero default a literal, the array size in place.
const RICH_MINIFIED: &str = "
    diagnostic(off, derivative_uniformity);
    struct a { A: vec3<f32>, @align(16) B: f32 }
    struct b { A: mat4x4<f32>, B: array<a, 2>, C: u32 }
    struct c { A: array<vec4<f32>> }
    struct d {
        @builtin(position) A: vec4<f32>,
        @location(0) B: vec3<f32>,
        @location(1) @interpolate(flat) C: u32,
        @location(2) @interpolate(linear, centroid) D: f32,
    }
    override BIAS: f32 = 1;
    @id(3) override e: f32 = 2 * BIAS;
    override COUNT: u32;
    override ZERO: f32 = 0;
    @group(0) @binding(0) var<uniform> f: b;
    @group(0) @binding(1) var<storage, read_write> g: c;
    @group(1) @binding(0) var h: texture_2d<f32>;
    @group(1) @binding(1) var i: sampler;
    @group(1) @binding(2) var j: texture_depth_2d;
    @group(1) @binding(3) var k: texture_storage_2d<rgba8unorm, write>;
    var<workgroup> l: array<f32, (COUNT * 2u)>;
    @vertex fn vs(@builtin(vertex_index) A: u32, @location(0) B: vec4f) -> d {
        var C: d;
        C.A = B * f.A;
        C.C = A + f.C + COUNT;
        C.D = B.x * 0.5 + ZERO;
        return C;
    }
    @fragment @early_depth_test(force) fn fs(A: d, @location(3) B: f32) -> @location(0) vec4f {
        _ = j;
        return textureSample(h, i, A.B.xy) * BIAS * e + f.B[A.C % 2u].B + B;
    }
    @compute @workgroup_size(64) fn cs(@builtin(global_invocation_id) A: vec3u) {
        l[A.x] = 1.0;
        g.A[A.x] = vec4f(l[0]);
        textureStore(k, vec2i(A.xy), vec4f());
    }
";

#[test]
fn renaming_and_respelling_leave_the_interface_alone() {
    assert_same(RICH, RICH_MINIFIED, &structural());
}

const PHONY: &str = "@group(0) @binding(0) var<uniform> u: vec4f;
    @group(0) @binding(1) var<storage, read_write> o: vec4f;
    @compute @workgroup_size(1) fn main() { _ = u; o = vec4f(1.0); }";
const NO_PHONY: &str = "@group(0) @binding(1) var<storage, read_write> o: vec4f;
    @compute @workgroup_size(1) fn main() { o = vec4f(1.0); }";

/// A binding accessed only from a phony assignment (or any dead code) is
/// accessed: losing the access is a change like any other.
#[test]
fn a_lost_static_access_is_a_change() {
    assert_differs(
        PHONY,
        NO_PHONY,
        &structural(),
        &["- entry compute main uses @group(0)@binding(0) var<uniform>:vec4<f32>"],
    );
}

/// An entry point named `uses` reads like any other.
#[test]
fn an_entry_point_named_uses_is_read_like_any_other() {
    let base = "@group(0) @binding(0) var<uniform> u: vec4f;
        @compute @workgroup_size(1) fn uses() { _ = u; }";
    let resized = base.replace("@workgroup_size(1)", "@workgroup_size(2)");
    assert_differs(
        base,
        &resized,
        &structural(),
        &[
            "- entry compute uses @workgroup_size(1,1,1)",
            "+ entry compute uses @workgroup_size(2,1,1)",
        ],
    );
    let unused = "@group(0) @binding(0) var<uniform> u: vec4f;
        @compute @workgroup_size(1) fn uses() { }";
    assert_differs(
        base,
        unused,
        &structural(),
        &["- entry compute uses uses @group(0)@binding(0) var<uniform>:vec4<f32>"],
    );
}

/// A library function's call graph is part of every entry point a host
/// composes it into, so its accesses are read too - through its callees,
/// and by structure.
#[test]
fn a_library_function_keeps_its_call_graph_accesses() {
    let library = "@group(0) @binding(0) var t: texture_2d<f32>;
        fn leaf() { _ = t; }
        fn root(x: f32) -> f32 { leaf(); return x; }";
    let lost = "@group(0) @binding(0) var t: texture_2d<f32>;
        fn leaf() { }
        fn root(x: f32) -> f32 { leaf(); return x; }";
    assert_differs(
        library,
        lost,
        &structural(),
        &[
            "- fn () uses @group(0)@binding(0) var:texture_2d<f32>",
            "- fn (f32)->f32 uses @group(0)@binding(0) var:texture_2d<f32>",
        ],
    );
    let renamed = library.replace("leaf", "l").replace("root", "r");
    assert_same(library, &renamed, &structural());
}

/// The pins the driver plants are the accesses the check reads: planted,
/// they add no line; cleared of every other name, the module still lists
/// each.
#[test]
fn pins_are_the_accesses_and_add_none() {
    let mut module = parsed(
        "@group(0) @binding(0) var<uniform> u: vec4f;
         @group(0) @binding(1) var<storage, read_write> o: vec4f;
         fn f() -> vec4f { return u; }
         @compute @workgroup_size(1) fn main() { let x = f(); o = vec4f(1.0); }",
    );
    let before = Interface::of(&module, structural());
    crate::pins::plant(&mut module);
    io::validate_module(&module).expect("valid with pins");
    assert!(
        before
            .mismatch(&Interface::of(&module, structural()))
            .is_none()
    );
    assert_eq!(
        crate::pins::of(
            &module.global_variables,
            module.functions.iter().next().unwrap().1
        )
        .len(),
        1
    );
    assert_eq!(
        crate::pins::of(&module.global_variables, &module.entry_points[0].function).len(),
        2
    );
    let globals = &module.global_variables;
    for (_, function) in module.functions.iter_mut() {
        crate::pins::retain(globals, function);
    }
    crate::pins::retain(globals, &mut module.entry_points[0].function);
    assert!(module.entry_points[0].function.named_expressions.len() == 2);
    // Planting again adds nothing.
    let count = module.entry_points[0].function.expressions.len();
    crate::pins::plant(&mut module);
    assert_eq!(module.entry_points[0].function.expressions.len(), count);
}

#[test]
fn a_gained_static_access_is_a_change() {
    assert_differs(
        NO_PHONY,
        PHONY,
        &structural(),
        &["+ entry compute main uses @group(0)@binding(0) var<uniform>:vec4<f32>"],
    );
}

#[test]
fn a_binding_the_input_never_used_is_not_interface() {
    let before = "@group(0) @binding(0) var<uniform> u: vec4f;
        @group(0) @binding(1) var<storage, read_write> o: vec4f;
        @compute @workgroup_size(1) fn main() { o = vec4f(1.0); }";
    assert_same(before, NO_PHONY, &structural());
    assert!(
        interface_of(before, &structural())
            .differences(&interface_of(NO_PHONY, &structural()))
            .is_empty()
    );
}

/// naga's analysis attributes no access to a global passed by pointer;
/// WGSL's static access does, and the spliced form agrees.
#[test]
fn a_global_passed_by_pointer_is_a_static_access() {
    let through_pointer = "struct S { a: i32 }
        @group(0) @binding(0) var<storage, read_write> s: S;
        fn f(p: ptr<storage, S, read_write>) { (*p).a = 1; }
        @compute @workgroup_size(1) fn main() { f(&s); }";
    let spliced = "struct S { a: i32 }
        @group(0) @binding(0) var<storage, read_write> s: S;
        @compute @workgroup_size(1) fn main() { s.a = 1; }";
    assert_same(through_pointer, spliced, &structural());
    assert!(
        interface_of(through_pointer, &structural())
            .differences(&interface_of(spliced, &structural()))
            .is_empty()
    );
}

#[test]
fn layout_binding_number_and_access_changes_are_changes() {
    let base = "struct S { a: f32, @align(16) b: f32 }
        @group(0) @binding(0) var<storage, read_write> s: S;
        @compute @workgroup_size(1) fn main() { s.a = s.b; }";
    let realigned = base.replace("@align(16) ", "");
    assert_differs(
        base,
        &realigned,
        &structural(),
        &[
            "+ entry compute main uses @group(0)@binding(0) var<storage,read_write>:struct{f32@0,f32@4}@size(8)",
        ],
    );
    let renumbered = base.replace("@binding(0)", "@binding(2)");
    assert_differs(
        base,
        &renumbered,
        &structural(),
        &["+ entry compute main uses @group(0)@binding(2)"],
    );
    let readonly = "struct S { a: f32, @align(16) b: f32 }
        @group(0) @binding(0) var<storage, read> s: S;
        @compute @workgroup_size(1) fn main() { _ = s.b; }";
    assert_differs(
        base,
        readonly,
        &structural(),
        &["+ entry compute main uses @group(0)@binding(0) var<storage,read>:"],
    );
}

#[test]
fn an_override_is_its_id_or_name_and_its_default() {
    let base = "override a: f32 = 1.0; @id(7) override b: u32 = 6u; override c: f32;
        @group(0) @binding(0) var<storage, read_write> o: f32;
        @compute @workgroup_size(1) fn main() { o = a + f32(b) + c; }";
    // An `@id` override's name is not the host's key; an id-less one's is.
    let renamed_id = base
        .replace("override b", "override z")
        .replace("f32(b)", "f32(z)");
    assert_same(base, &renamed_id, &structural());
    let renamed_key = base
        .replace("override a", "override y")
        .replace("o = a", "o = y");
    assert_differs(
        base,
        &renamed_key,
        &structural(),
        &["- override a:", "+ override y:"],
    );
    let default_changed = base.replace("= 1.0;", "= 1.5;");
    assert_differs(
        base,
        &default_changed,
        &structural(),
        &["- override a:f32=lit0:"],
    );
    let default_dropped = base.replace("override c: f32;", "override c: f32 = 0.0;");
    assert_differs(
        base,
        &default_dropped,
        &structural(),
        &["- override c:f32 required"],
    );
    let phantom_dropped = base.replace("override c: f32;", "").replace(" + c;", ";");
    assert_differs(
        base,
        &phantom_dropped,
        &structural(),
        &["- override c:f32 required"],
    );
    let zero_spelled = base.replace("= 1.0;", "= f32();");
    let zero_literal = base.replace("= 1.0;", "= 0.0;");
    assert_same(&zero_spelled, &zero_literal, &structural());
}

/// naga's own override for an override-expression array size has no key
/// a host could set: neither its presence nor its spelling counts.
#[test]
fn an_anonymous_override_is_not_a_key() {
    let sized = "override n: u32; var<workgroup> w: array<f32, n * 2u>;
        @compute @workgroup_size(1) fn main() { w[0] = 1.0; }";
    let in_place = sized.replace("n * 2u", "(n * 2u)");
    assert_same(sized, &in_place, &structural());
    let gone = "override n: u32; @compute @workgroup_size(1) fn main() { _ = n; }";
    assert_same(sized, gone, &structural());
}

#[test]
fn entry_point_signature_changes_are_changes() {
    let base =
        "@fragment fn fs(@location(0) a: vec2f, @location(1) unused: f32) -> @location(0) vec4f {
            return vec4f(a, 0.0, 1.0);
        }
        @compute @workgroup_size(4, 2, 1) fn cs() {}";
    let dropped_input = base.replace(", @location(1) unused: f32", "");
    assert_differs(
        base,
        &dropped_input,
        &structural(),
        &["- entry fragment fs in @location(1) f32"],
    );
    let resized = base.replace("@workgroup_size(4, 2, 1)", "@workgroup_size(4, 1, 1)");
    assert_differs(
        base,
        &resized,
        &structural(),
        &[
            "- entry compute cs @workgroup_size(4,2,1)",
            "+ entry compute cs @workgroup_size(4,1,1)",
        ],
    );
    let renamed = base.replace("fn cs()", "fn compute_main()");
    assert_differs(
        base,
        &renamed,
        &structural(),
        &["- entry compute cs", "+ entry compute compute_main"],
    );
    let retyped = base
        .replace("a: vec2f", "a: vec3f")
        .replace("vec4f(a, 0.0, 1.0)", "vec4f(a, 1.0)");
    assert_differs(
        base,
        &retyped,
        &structural(),
        &["in @location(0) vec2<f32>", "in @location(0) vec3<f32>"],
    );
    let early = base.replace(
        "@fragment fn fs",
        "@fragment @early_depth_test(less_equal) fn fs",
    );
    assert_differs(
        base,
        &early,
        &structural(),
        &["+ entry fragment fs @early_depth_test(less_equal)"],
    );
}

#[test]
fn interpolation_defaults_are_one_binding() {
    let explicit = "@fragment fn fs(@location(0) @interpolate(perspective) a: f32, @location(1) @interpolate(flat) k: u32) -> @location(0) vec4f { return vec4f(a, f32(k), 0.0, 1.0); }";
    let implicit = "@fragment fn fs(@location(0) a: f32, @location(1) @interpolate(flat, first) k: u32) -> @location(0) vec4f { return vec4f(a, f32(k), 0.0, 1.0); }";
    assert_same(explicit, implicit, &structural());
    let centroid = explicit.replace(
        "@interpolate(perspective)",
        "@interpolate(perspective, centroid)",
    );
    assert_differs(
        explicit,
        &centroid,
        &structural(),
        &["@interpolate(perspective,centroid)"],
    );
    let either = implicit.replace("flat, first", "flat, either");
    assert_differs(
        implicit,
        &either,
        &structural(),
        &["@interpolate(flat,either)"],
    );
}

#[test]
fn names_count_only_when_asked() {
    let base = "struct S { a: f32 } @group(0) @binding(0) var<uniform> u: S;
        @compute @workgroup_size(1) fn main() { _ = u.a; }";
    let renamed = "struct T { b: f32 } @group(0) @binding(0) var<uniform> v: T;
        @compute @workgroup_size(1) fn main() { _ = v.b; }";
    assert_same(base, renamed, &structural());
    let names = Options {
        names: true,
        ..structural()
    };
    assert_differs(
        base,
        renamed,
        &names,
        &[
            "- global @group(0)@binding(0) var<uniform> u:struct S{a:f32@0}@size(4)",
            "+ global @group(0)@binding(0) var<uniform> v:struct T{b:f32@0}@size(4)",
        ],
    );
}

#[test]
fn float_values_are_read_only_under_full_precision() {
    let base = "override a: f32 = 0.123456; @compute @workgroup_size(1) fn main() { _ = a; }";
    let rounded = base.replace("0.123456", "0.12");
    assert_differs(base, &rounded, &structural(), &["- override a:f32=lit0:"]);
    let lossy = Options {
        float_values: false,
        ..structural()
    };
    assert_same(base, &rounded, &lossy);
    let config = Config {
        float_precision: crate::config::FloatPrecision::all(PrecisionMode::DecimalPlaces(2)),
        ..Config::default()
    };
    assert!(!Options::from_config(&config).float_values);
    assert!(Options::from_config(&Config::default()).float_values);
}

#[test]
fn a_library_exports_every_declaration_by_structure() {
    let base = "struct S { a: f32, b: f32 }
        @group(0) @binding(0) var<uniform> u: S;
        fn helper(x: f32, unused: f32) -> f32 { return x * u.a; }
        fn other(s: S) -> f32 { return helper(s.a, s.b); }";
    let renamed = "struct T { c: f32, d: f32 }
        @group(0) @binding(0) var<uniform> v: T;
        fn h(x: f32, y: f32) -> f32 { return x * v.c; }
        fn o(s: T) -> f32 { return h(s.c, s.d); }";
    assert_same(base, renamed, &structural());
    let arity = base
        .replace("x: f32, unused: f32", "x: f32")
        .replace("helper(s.a, s.b)", "helper(s.a)");
    assert_differs(
        base,
        &arity,
        &structural(),
        &["- fn (f32,f32)->f32", "+ fn (f32)->f32"],
    );
    let member_dropped = "struct S { a: f32 }
        @group(0) @binding(0) var<uniform> u: S;
        fn helper(x: f32, unused: f32) -> f32 { return x * u.a; }
        fn other(s: S) -> f32 { return helper(s.a, s.a); }";
    assert_differs(
        base,
        member_dropped,
        &structural(),
        &[
            "- type struct{f32@0,f32@4}@size(8)",
            "+ type struct{f32@0}@size(4)",
        ],
    );
}

/// Under names, a library reads exactly the names `preserve_interface`
/// keeps: the bound global, its struct and members.  Its functions and a
/// struct no binding reaches are renamed like any (the name map covers
/// them); read as interface, every library minified under the option
/// shipped its input compacted.
#[test]
fn a_library_under_names_reads_only_the_names_the_option_keeps() {
    let base = "struct S { a: f32, b: f32 } struct P { x: f32 }
        @group(0) @binding(0) var<uniform> u: S;
        fn helper(p: P) -> f32 { return p.x * u.a; }
        fn other(s: S) -> f32 { return helper(P(s.b)); }";
    let renamed = "struct S { a: f32, b: f32 } struct Q { y: f32 }
        @group(0) @binding(0) var<uniform> u: S;
        fn h(p: Q) -> f32 { return p.y * u.a; }
        fn o(s: S) -> f32 { return h(Q(s.b)); }";
    let names = Options {
        names: true,
        ..structural()
    };
    assert_same(base, renamed, &names);
    let bound_renamed = renamed
        .replace("struct S { a: f32, b: f32 }", "struct T { a: f32, b: f32 }")
        .replace("u: S", "u: T")
        .replace("s: S", "s: T");
    assert_differs(
        base,
        &bound_renamed,
        &names,
        &[
            "- global @group(0)@binding(0) var<uniform> u:struct S{a:f32@0,b:f32@4}@size(8)",
            "+ global @group(0)@binding(0) var<uniform> u:struct T{a:f32@0,b:f32@4}@size(8)",
        ],
    );
    let library = crate::run(
        base,
        &Config {
            preserve_interface: true,
            ..Config::default()
        },
    )
    .expect("minifies");
    assert!(
        library.report.fallback.is_none(),
        "{:?}",
        library.report.fallback
    );
    assert!(
        library.report.bailout.is_none(),
        "{:?}",
        library.report.bailout
    );
    assert!(
        library.source.contains("struct S{a:f32,b:f32}") && !library.source.contains("helper"),
        "the bound struct keeps its names, the helper is renamed: {}",
        library.source
    );
}

#[test]
fn a_preserve_listed_function_keeps_its_signature() {
    let base = "fn helper(x: f32, unused: f32) -> f32 { return x; }
        @compute @workgroup_size(1) fn main() { _ = helper(1.0, 2.0); }";
    let arity = "fn helper(x: f32) -> f32 { return x; }
        @compute @workgroup_size(1) fn main() { _ = helper(1.0); }";
    // Unlisted, a helper's signature is the minifier's to change.
    assert_same(base, arity, &structural());
    let listed = Options {
        symbols: vec!["helper".to_string()],
        ..structural()
    };
    assert_differs(
        base,
        arity,
        &listed,
        &[
            "- symbol fn helper(f32,f32)->f32",
            "+ symbol fn helper(f32)->f32",
        ],
    );
}

#[test]
fn check_text_names_the_text_as_the_subject() {
    let before = "@group(0) @binding(0) var<uniform> u: vec4f;
        @compute @workgroup_size(1) fn main() { _ = u; }";
    let interface = interface_of(before, &structural());
    interface
        .check_text(before)
        .expect("the input is its own interface");
    let err = interface
        .check_text("@group(0) @binding(1) var<uniform> u: vec4f; @compute @workgroup_size(1) fn main() { _ = u; }")
        .expect_err("a renumbered binding");
    let msg = err.to_string();
    assert!(
        msg.starts_with("the text changed the host interface"),
        "{msg}"
    );
    assert!(
        msg.contains("+ entry compute main uses @group(0)@binding(1)"),
        "{msg}"
    );
    assert!(err.kind() == "validation", "{msg}");
}

/// The shipped text of a run meets the postcondition without a fallback:
/// the ladder would otherwise degrade to naga's text or the compacted input.
#[test]
fn a_run_ships_the_generator_text_for_the_rich_shader() {
    for config in [
        Config::default(),
        Config {
            preserve_interface: true,
            ..Config::default()
        },
        Config {
            preserve_symbols: vec!["helper".to_string(), "scratch".to_string()],
            ..Config::default()
        },
    ] {
        let output = crate::run(RICH, &config).expect("minifies");
        assert!(
            output.report.fallback.is_none(),
            "{:?}",
            output.report.fallback
        );
        assert!(
            output.report.bailout.is_none(),
            "{:?}",
            output.report.bailout
        );
        assert!(
            output.source.contains("@compute@workgroup_size(64)"),
            "{}",
            output.source
        );
        assert!(
            output.source.contains("@early_depth_test(force)"),
            "{}",
            output.source
        );
    }
}
