//! One JSON rendering of a run, printed by the CLI and parsed by the wasm
//! binding, so both surfaces share a schema by construction.  Hand-rolled:
//! the shapes are tiny and fixed (`BTreeMap` orders keys) and serde is not
//! worth a dependency for this.  The wasm TypeScript interfaces document
//! the schema.

use std::collections::BTreeMap;
use std::fmt::Display;

use crate::Output;
use crate::name_map::NameMap;

/// `{"source","report","nameMap"}`; `nameMap` is `null` iff
/// [`Output::name_map`] is `None`.
pub fn render_output(output: &Output) -> String {
    let report = &output.report;
    let mut out = String::from("{\"source\":");
    push_string(&output.source, &mut out);
    out.push_str(&format!(
        ",\"report\":{{\"inputBytes\":{},\"outputBytes\":{},\"converged\":{},\"sweeps\":{},\"bailout\":",
        report.input_bytes, report.output_bytes, report.converged, report.sweeps
    ));
    match &report.bailout {
        None => out.push_str("null"),
        Some(reason) => push_string(reason, &mut out),
    }
    out.push_str(",\"passReports\":[");
    for (i, p) in report.pass_reports.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"passName\":");
        push_string(&p.pass_name, &mut out);
        out.push_str(&format!(
            ",\"beforeBytes\":{},\"afterBytes\":{},\"changed\":{},\"durationUs\":{},\
             \"validationOk\":{},\"textValidationOk\":{},\"rolledBack\":{}}}",
            opt(p.before_bytes),
            opt(p.after_bytes),
            p.changed,
            p.duration_us,
            p.validation_ok,
            opt(p.text_validation_ok),
            p.rolled_back
        ));
    }
    out.push_str("]},\"nameMap\":");
    out.push_str(&render_name_map(output.name_map.as_ref()));
    out.push('}');
    out
}

/// The `NameMap` object, or `null`.
pub fn render_name_map(map: Option<&NameMap>) -> String {
    let Some(m) = map else {
        return "null".to_string();
    };
    let mut out = String::from("{");
    push_string_map("entryPoints", &m.entry_points, &mut out);
    out.push(',');
    push_string_map("globals", &m.globals, &mut out);
    out.push(',');
    push_string_map("functions", &m.functions, &mut out);
    out.push(',');
    push_string_map("constants", &m.constants, &mut out);
    out.push(',');
    push_string_map("overrides", &m.overrides, &mut out);
    out.push_str(",\"structs\":{");
    for (i, (orig, sr)) in m.structs.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push_string(orig, &mut out);
        out.push_str(":{\"name\":");
        push_string(&sr.name, &mut out);
        out.push(',');
        push_string_map("members", &sr.members, &mut out);
        out.push('}');
    }
    out.push_str("}}");
    out
}

fn opt<T: Display>(v: Option<T>) -> String {
    v.map_or_else(|| "null".to_string(), |x| x.to_string())
}

fn push_string_map(key: &str, entries: &BTreeMap<String, String>, out: &mut String) {
    push_string(key, out);
    out.push_str(":{");
    for (i, (k, v)) in entries.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push_string(k, out);
        out.push(':');
        push_string(v, out);
    }
    out.push('}');
}

/// Control characters as `\uXXXX`, so shader text stays on one line.
fn push_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn strings_escape_quotes_backslashes_and_controls() {
        let mut out = String::new();
        push_string("a\"b\\c\n\t\u{1}\u{2028}", &mut out);
        assert_eq!(out, "\"a\\\"b\\\\c\\u000a\\u0009\\u0001\u{2028}\"");
    }

    /// Nested `report`, matching the wasm `Output` interface.
    #[test]
    fn output_document_has_the_wasm_shape() {
        let output = crate::run(
            "@group(0) @binding(0) var<storage, read_write> data: f32;\n\
             @compute @workgroup_size(1) fn m() { data = 2.0 * 3.0; }",
            &Config::default(),
        )
        .expect("minifies");
        let doc = render_output(&output);
        assert!(doc.starts_with("{\"source\":\""), "{doc}");
        assert!(
            doc.contains(",\"report\":{\"inputBytes\":")
                && doc.contains(",\"bailout\":null,\"passReports\":[{\"passName\":\""),
            "{doc}"
        );
        assert!(
            doc.contains("\"nameMap\":{\"entryPoints\":{\"m\":\"m\"},\"globals\":{\"data\":\"")
                && doc.ends_with("}}}"),
            "{doc}"
        );
    }

    #[test]
    fn bailout_document_carries_the_reason_and_no_map() {
        let output = crate::run(
            "@compute @workgroup_size(1) fn m() { var x = 1; let d = x / 0; }",
            &Config::default(),
        )
        .expect("bails out, not Err");
        let doc = render_output(&output);
        assert!(
            doc.contains(",\"bailout\":\"naga rejects the input: "),
            "{doc}"
        );
        assert!(
            doc.ends_with(",\"passReports\":[]},\"nameMap\":null}"),
            "{doc}"
        );
        assert_eq!(render_name_map(None), "null");
    }
}
