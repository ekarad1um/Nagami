//! [`crate::generator::price`]: the prices a pass reads are the emitter's
//! own text.

use std::collections::HashSet;

use crate::config::Config;
use crate::generator::price::Pricer;
use crate::generator::{GenerateOptions, generate};
use crate::passes::rename::plan_names;

/// The callee's declaration prices as the bytes the module render spells
/// it in; binding a `let` as the emitter would makes the roots after it
/// render the name.
#[test]
fn a_declaration_prices_as_rendered_and_a_bound_let_renders_as_its_name() {
    let src = r#"
fn f(a: f32, b: f32) -> f32 {
    let t = a * 2.0 + b - a;
    return t * t + a;
}
@fragment fn main(@location(0) p: f32) -> @location(0) vec4f { return vec4f(f(p, 1.0)); }
"#;
    let module = naga::front::wgsl::parse_str(src).expect("parse failed");
    let config = Config::default();
    let plan = plan_names(&module, &HashSet::new(), config.mangle());
    let renamed = plan.applied(&module);
    let info = crate::io::validate_module(&renamed).expect("validation failed");
    let rendered = generate(&renamed, &info, GenerateOptions::from_config(&config))
        .expect("generate failed")
        .source;
    let (fh, f) = renamed.functions.iter().next().expect("one function");
    let name = f.name.as_deref().expect("a name");
    let start = rendered
        .find(&format!("fn {name}("))
        .expect("the declaration");
    let end = rendered[start..].find("}").expect("its close") + start + 1;
    let declaration = &rendered[start..end];
    assert!(declaration.contains("let "), "{declaration}");

    let mut pricer = Pricer::new(&renamed, &info, &config, &plan);
    let mut fp = pricer.function(fh);
    assert_eq!(fp.definition_len(), Some(declaration.len()));
    assert_eq!(fp.name_len(), name.len());
    assert_eq!(fp.argument_name_len(0), 1);

    let (product, value) = {
        let mut emitted = f.body.iter().filter_map(|s| match s {
            naga::Statement::Emit(range) => Some(range.clone()),
            _ => None,
        });
        let first = emitted
            .next()
            .expect("the let's range")
            .last()
            .expect("a handle");
        let value = match f.body.last() {
            Some(naga::Statement::Return { value: Some(v) }) => *v,
            _ => panic!("a returned value"),
        };
        (first, value)
    };
    assert!(fp.binds(product), "two uses of `a*2+b-a` bind");
    let unbound = fp.expr_len(value).expect("priced");
    let let_len = fp.let_name_len();
    assert_eq!(fp.emit(product), Some("a*2+b-a".len() + let_len + 6));
    assert!(fp.bound(product));
    let bound = fp.expr_len(value).expect("priced");
    assert_eq!(bound, "t*t+a".len());
    assert!(unbound > bound, "{unbound} vs {bound}");
}
