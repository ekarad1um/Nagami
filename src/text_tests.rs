//! Child module of `text.rs` (via `#[path]`), so `use super::*` keeps
//! private items reachable.

use super::*;

#[test]
fn compact_wgsl_text_is_token_safe_and_idempotent() {
    // Comments become whitespace; runs collapse; ident-ident keeps one
    // space; token-fusing pairs keep one space; the rest joins.
    let src = "enable f16;  // trailing comment\n\
                   /* block */ fn  m ( a : f32 , p : ptr<function, f32> ) {\n\
                     let b = a - -a;\n\
                     let c = b / *p;\n\
                   }";
    let compacted = compact_wgsl_text(src);
    assert_eq!(
        compacted,
        "enable f16;fn m(a:f32,p:ptr<function,f32>){let b=a- -a;let c=b/ *p;}"
    );
    // `- -` must not fuse into the reserved `--`, nor `/ *` into `/*`.
    assert!(!compacted.contains("--") && !compacted.contains("/*"));
    // Idempotent: a second pass is byte-identical.
    assert_eq!(compact_wgsl_text(&compacted), compacted);
}

#[test]
fn split_directives_extracts_leading_enables() {
    let src = "enable f16;\nenable subgroups;\nstruct S { x: f32 }\nfn f() {}\n";
    let (dirs, rest) = split_directives(src);
    assert_eq!(dirs, "enable f16;\nenable subgroups;\n");
    assert_eq!(rest, "struct S { x: f32 }\nfn f() {}\n");
}

#[test]
fn split_directives_handles_comments_and_blanks() {
    let src = "// header\n\nenable f16;\n\nstruct S { x: f32 }\n";
    let (dirs, rest) = split_directives(src);
    assert_eq!(dirs, "// header\n\nenable f16;\n\n");
    assert_eq!(rest, "struct S { x: f32 }\n");
}

#[test]
fn split_directives_terminator_scan_ignores_semicolon_in_block_comment() {
    // A `;` inside a comment BETWEEN the directive keyword and its real
    // terminator must not split the directive mid-comment (which would
    // splice a preamble into the broken comment region).
    let src = "diagnostic /* a;b */ (off, derivative_uniformity);\n\
                   @fragment fn main() -> @location(0) vec4f { return vec4f(0.); }\n";
    let (dirs, rest) = split_directives(src);
    assert_eq!(dirs, "diagnostic /* a;b */ (off, derivative_uniformity);\n");
    assert!(
        rest.starts_with("@fragment"),
        "body must start at the real declaration, not inside the comment: {rest:?}"
    );
}

#[test]
fn split_directives_terminator_scan_ignores_semicolon_in_line_comment() {
    let src = "enable f16; // trailing ; comment\nstruct S { x: f16 }\n";
    let (dirs, rest) = split_directives(src);
    // The directive's own `;` terminates it; the line comment is body
    // trivia.  (The point is the line-comment `;` is not mistaken for a
    // second directive terminator.)
    assert!(dirs.starts_with("enable f16;"));
    assert!(rest.contains("struct S"));
}

#[test]
fn split_directives_no_directives() {
    let src = "struct S { x: f32 }\nfn f() {}\n";
    let (dirs, rest) = split_directives(src);
    assert_eq!(dirs, "");
    assert_eq!(rest, src);
}

#[test]
fn split_directives_diagnostic() {
    let src = "diagnostic(off, derivative_uniformity);\nfn f() {}\n";
    let (dirs, rest) = split_directives(src);
    assert_eq!(dirs, "diagnostic(off, derivative_uniformity);\n");
    assert_eq!(rest, "fn f() {}\n");
}

#[test]
fn split_directives_line_comment_ends_at_vertical_tab() {
    // A `//` comment ends at ANY WGSL line break (VT/CR/FF/...), not just `\n`.
    // A directive after such a comment must still be hoisted; otherwise the
    // preamble path ships it after the preamble's declarations (invalid, exit 0).
    for brk in ["\u{0b}", "\r", "\u{0c}"] {
        let src = format!("//x{brk}diagnostic(off, derivative_uniformity);\nfn f() {{}}\n");
        let (dirs, rest) = split_directives(&src);
        assert!(
            dirs.contains("diagnostic(off, derivative_uniformity);"),
            "directive after a line comment ended by {brk:?} must be hoisted: {dirs:?}"
        );
        assert_eq!(rest, "fn f() {}\n");
    }
}

#[test]
fn has_enable_directive_accepts_line_break_after_keyword() {
    // `enable` may be separated from the extension by any blankspace, incl. a
    // line break (`enable\nf16;` is valid WGSL); the guard must not read that as
    // an identifier and miss the directive (which spuriously fires the f16
    // preamble guard on a preamble that DOES enable f16).
    assert!(has_enable_directive("enable\nf16;", "f16"));
    assert!(has_enable_directive("enable\r\nf16;", "f16"));
    assert!(has_enable_directive("enable f16;", "f16"));
    assert!(!has_enable_directive("enablef16;", "f16"));
}

/// Regression: when `split_directives` returns a fragment that
/// lacks a trailing newline (e.g. a source whose last directive
/// is the final byte), splicing it directly in front of the
/// preamble's directives glued them together into one syntax
/// error.  `join_with_newline` must insert a separator.
#[test]
fn join_with_newline_inserts_separator_between_fragments() {
    let joined = join_with_newline(&["enable f16;", "enable subgroups;\n", "fn body() {}\n"]);
    assert_eq!(joined, "enable f16;\nenable subgroups;\nfn body() {}\n");
}

#[test]
fn join_with_newline_skips_empty_fragments() {
    let joined = join_with_newline(&["enable f16;\n", "", "fn body() {}\n"]);
    assert_eq!(joined, "enable f16;\nfn body() {}\n");
}

#[test]
fn join_with_newline_keeps_existing_trailing_newline() {
    let joined = join_with_newline(&["enable f16;\n", "fn body() {}\n"]);
    assert_eq!(joined, "enable f16;\nfn body() {}\n");
}

/// CRLF fragments survive `split_directives` (verified by
/// `split_directives_crlf_line_endings`), so the join helper must
/// preserve them unchanged.  `ends_with('\n')` matches the LF in
/// `\r\n`, so no extra newline is appended after a CRLF-ending
/// fragment.
#[test]
fn join_with_newline_preserves_crlf_fragments() {
    let joined = join_with_newline(&["enable f16;\r\n", "fn body() {}\r\n"]);
    assert_eq!(joined, "enable f16;\r\nfn body() {}\r\n");
}

#[test]
fn split_directives_requires() {
    let src = "requires readonly_and_readwrite_storage_textures;\nfn f() {}\n";
    let (dirs, rest) = split_directives(src);
    assert_eq!(dirs, "requires readonly_and_readwrite_storage_textures;\n");
    assert_eq!(rest, "fn f() {}\n");
}

#[test]
fn split_directives_crlf_line_endings() {
    let src = "enable f16;\r\nstruct S { x: f32 }\r\n";
    let (dirs, rest) = split_directives(src);
    assert_eq!(dirs, "enable f16;\r\n");
    assert_eq!(rest, "struct S { x: f32 }\r\n");
}

#[test]
fn split_directives_no_trailing_newline() {
    let src = "enable f16;";
    let (dirs, rest) = split_directives(src);
    assert_eq!(dirs, "enable f16;");
    assert_eq!(rest, "");
}

#[test]
fn split_directives_all_directives() {
    let src = "enable f16;\nrequires something;\n";
    let (dirs, rest) = split_directives(src);
    assert_eq!(dirs, src);
    assert_eq!(rest, "");
}

#[test]
fn split_directives_empty_source() {
    let (dirs, rest) = split_directives("");
    assert_eq!(dirs, "");
    assert_eq!(rest, "");
}

#[test]
fn split_directives_compact_single_line() {
    // Compact generator output is one physical line; the splitter must
    // still stop right after the directive's `;`, not swallow the body.
    let src = "enable f16;@group(0)@binding(0)var<storage,read_write>A:f16;\
                   @compute @workgroup_size(1) fn m(){A=1h;}";
    let (dirs, body) = split_directives(src);
    assert_eq!(dirs, "enable f16;");
    assert_eq!(
        body,
        "@group(0)@binding(0)var<storage,read_write>A:f16;\
             @compute @workgroup_size(1) fn m(){A=1h;}"
    );
    // Multiple directives run together on one line are all consumed.
    let (dirs, body) =
        split_directives("enable f16;enable dual_source_blending;@fragment fn m(){}");
    assert_eq!(dirs, "enable f16;enable dual_source_blending;");
    assert_eq!(body, "@fragment fn m(){}");
}

#[test]
fn split_directives_word_boundary_single_line() {
    // Identifiers that merely start with a directive keyword must NOT be
    // hoisted, even when the whole input is one line.
    assert_eq!(
        split_directives("enablef16;fn m(){}"),
        ("", "enablef16;fn m(){}")
    );
    assert_eq!(
        split_directives("requires_foo();fn m(){}"),
        ("", "requires_foo();fn m(){}")
    );
    // A real directive followed mid-line by a `diagnostic`-prefixed
    // identifier stops at that identifier.
    assert_eq!(
        split_directives("enable f16;diagnostic_counter_thing fn m(){}"),
        ("enable f16;", "diagnostic_counter_thing fn m(){}")
    );
}

#[test]
fn references_f16_token_ignores_identifiers() {
    // `myf16var` and `f16_test` must NOT be detected as an f16 token.
    assert!(!references_f16_token("var myf16var: i32;"));
    assert!(!references_f16_token("fn f16_test() {}"));
    assert!(!references_f16_token("let x = ff16;"));
}

#[test]
fn references_f16_token_detects_real_use() {
    assert!(references_f16_token("var x: f16 = 1.0h;"));
    assert!(references_f16_token("let v = vec3<f16>(0h);"));
    assert!(references_f16_token("fn f() -> f16 { return 0h; }"));
}

#[test]
fn references_f16_token_detects_aliases_and_suffix() {
    // Predeclared half-precision aliases (no `f16` substring).
    assert!(references_f16_token("var v: vec2h = vec2h(1.0h, 2.0h);"));
    assert!(references_f16_token("let v = vec3h(0h);"));
    assert!(references_f16_token("var m: mat4x4h;"));
    assert!(references_f16_token("let m = mat2x3h();"));
    // `h` float-literal suffix in its various spellings, no alias/keyword.
    assert!(references_f16_token("let x = 1.0h + 2.0h;"));
    assert!(references_f16_token("let x = 0h;"));
    assert!(references_f16_token("let x = 1.5e2h;"));
    assert!(references_f16_token("let x = 1.0e-3h;"));
    assert!(references_f16_token("let x = 0x1p2h;"));
    // Negatives: longer identifiers, other float suffixes, plain ints.
    assert!(!references_f16_token("var width: f32; var height: f32;"));
    assert!(!references_f16_token("let mesh = 1.0;"));
    assert!(!references_f16_token("var vec2hh: i32;"));
    assert!(!references_f16_token("let x = 1.0f + 2u + 3;"));
    assert!(!references_f16_token("let x = 1.0e-3;"));
    assert!(!references_f16_token("fn vec2h_helper() {}"));
}

#[test]
fn references_f16_token_ignores_comments() {
    // f16 mentioned only inside comments should not trigger injection.
    assert!(!references_f16_token("// uses f16 later\nvar x: i32;"));
    assert!(!references_f16_token("/* f16 */ var x: i32;"));
    assert!(!references_f16_token(
        "/* multiline\n   f16\n */\nvar x: i32;"
    ));
}

#[test]
fn references_f16_token_ignores_nested_block_comments() {
    assert!(!references_f16_token(
        "/* outer /* inner */ f16 still in outer */ var x: i32;"
    ));
    assert!(!references_f16_token(
        "/* /* /* deeply nested */ */ f16 inside */ var x: i32;"
    ));
    // A real f16 after a properly-closed nested comment still
    // detects.
    assert!(references_f16_token(
        "/* nest /* inner */ done */ var x: f16 = 0h;"
    ));
}

#[test]
fn normalize_line_endings_handles_lone_cr() {
    // Lone `\r` becomes `\n`.
    assert_eq!(normalize_line_endings("a\rb\rc"), "a\nb\nc");
    // `\r\n` stays intact (str::lines already handles it).
    assert_eq!(normalize_line_endings("a\r\nb\r\nc"), "a\r\nb\r\nc");
    // Mixed.
    assert_eq!(normalize_line_endings("a\rb\r\nc\nd"), "a\nb\r\nc\nd");
    // Source with no `\r` returns identical content.
    assert_eq!(normalize_line_endings("a\nb\nc"), "a\nb\nc");
    // Multi-byte UTF-8 around line endings preserved.
    assert_eq!(normalize_line_endings("α\rβ\r\nγ"), "α\nβ\r\nγ");
}

#[test]
fn has_enable_f16_directive_matches_canonical() {
    assert!(has_enable_f16_directive("enable f16;\n"));
    assert!(has_enable_f16_directive("enable f16;"));
}

#[test]
fn has_enable_f16_directive_matches_extra_whitespace() {
    assert!(has_enable_f16_directive("enable  f16;\n"));
    assert!(has_enable_f16_directive("enable\tf16;\n"));
    assert!(has_enable_f16_directive("enable f16 ;\n"));
}

#[test]
fn has_enable_f16_directive_rejects_commented_out() {
    assert!(!has_enable_f16_directive("// enable f16;"));
    assert!(!has_enable_f16_directive("/* enable f16; */"));
}

#[test]
fn has_enable_f16_directive_rejects_non_directive() {
    assert!(!has_enable_f16_directive("var enable_f16: bool;"));
    assert!(!has_enable_f16_directive("fn enable() {} // f16"));
}

#[test]
fn has_enable_f16_directive_matches_comma_separated_list() {
    // WGSL permits a comma-separated enable list; `f16` in any position must
    // be recognised.  A false negative here is not harmless: the preamble
    // guard in `run` turns it into a hard error on valid input.
    assert!(has_enable_f16_directive("enable f16, clip_distances;\n"));
    assert!(has_enable_f16_directive("enable clip_distances, f16;\n"));
    assert!(has_enable_f16_directive(
        "enable dual_source_blending , f16 ;"
    ));
    assert!(!has_enable_f16_directive(
        "enable clip_distances, dual_source_blending;"
    ));
}

#[test]
fn has_enable_f16_directive_matches_second_directive_on_same_line() {
    // Several directives may share one physical line; f16 in any of them
    // must be found (a false negative hard-errors the preamble guard).
    assert!(has_enable_f16_directive(
        "enable dual_source_blending; enable f16;"
    ));
    assert!(has_enable_f16_directive(
        "enable clip_distances; enable f16, subgroups;"
    ));
    // Directives on separate lines (each ends in `;`, all before any decl).
    assert!(has_enable_f16_directive(
        "enable dual_source_blending;\nenable f16;\nstruct S { x: f32 }"
    ));
    assert!(!has_enable_f16_directive(
        "enable dual_source_blending; enable clip_distances;"
    ));
}

#[test]
fn split_directives_recognizes_paren_diagnostic() {
    let source = "diagnostic(off, derivative_uniformity);\nfn main(){}\n";
    let (dirs, body) = split_directives(source);
    assert_eq!(dirs, "diagnostic(off, derivative_uniformity);\n");
    assert_eq!(body, "fn main(){}\n");
}

#[test]
fn split_directives_recognizes_space_diagnostic() {
    let source = "diagnostic (off, derivative_uniformity);\nfn main(){}\n";
    let (dirs, body) = split_directives(source);
    assert_eq!(dirs, "diagnostic (off, derivative_uniformity);\n");
    assert_eq!(body, "fn main(){}\n");
}

#[test]
fn split_directives_does_not_capture_diagnostic_prefixed_identifier() {
    // `diagnostic_counter` is a plain identifier; it is not a
    // directive and must not be hoisted above the preamble.  But the
    // realistic failure path is a top-level declaration whose RHS
    // happens to start with `diagnostic_...`.  In WGSL the top-level
    // line would start with `const`/`var`/etc., so the split would
    // terminate there.  Still, guard against pathological inputs
    // that begin a line with an identifier-looking token.
    let source = "diagnostic_counter_alias\nfn main(){}\n";
    let (dirs, body) = split_directives(source);
    assert_eq!(
        dirs, "",
        "identifier starting with 'diagnostic' must not be treated as a directive"
    );
    assert_eq!(body, source);
}
