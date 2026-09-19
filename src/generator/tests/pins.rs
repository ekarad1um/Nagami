//! The pins the emitter prints (`crate::pins`): decided by what the body's
//! text mentions, so a statement the emitter itself elides - the no-op
//! `p = p`, a vacuous tail - leaves the access to the pin.  The generator
//! alone, no pass: the shapes reach it as the front end lowers them.

use super::super::{GenerateOptions, generate};
use super::helpers::assert_valid_wgsl;

fn pinned(src: &str) -> String {
    let mut module = naga::front::wgsl::parse_str(src).expect("parse failed");
    crate::pins::plant(&mut module);
    let info = crate::io::validate_module(&module).expect("validation failed");
    generate(&module, &info, GenerateOptions::default())
        .expect("generate failed")
        .source
}

const DECLS: &str = "@group(0) @binding(0) var<uniform> u: vec4f;
    @group(0) @binding(1) var<storage, read_write> o: array<f32>;";

#[test]
fn a_vacuous_tail_the_emitter_skips_leaves_its_pin() {
    let src = format!(
        "{DECLS}
        @compute @workgroup_size(1) fn main() {{ o[0] = 1.0; if u.x > 0.0 {{ return; }} }}"
    );
    let out = pinned(&src);
    assert!(out.ends_with("fn main(){_=u;o[0]=1;}"), "{out}");
    assert_valid_wgsl(&out);
}

#[test]
fn a_no_op_store_the_emitter_drops_leaves_its_pin() {
    let src = format!(
        "{DECLS}
        @compute @workgroup_size(1) fn main() {{ o[1] = o[1]; }}"
    );
    let out = pinned(&src);
    assert!(out.ends_with("fn main(){_=&o;}"), "{out}");
    assert_valid_wgsl(&out);
}

#[test]
fn a_mentioned_global_prints_no_pin() {
    let src = format!(
        "{DECLS}
        @compute @workgroup_size(1) fn main() {{ o[0] = u.x; }}"
    );
    let out = pinned(&src);
    assert!(!out.contains("_="), "{out}");
    assert!(out.ends_with("fn main(){o[0]=u.x;}"), "{out}");
}
