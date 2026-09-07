//! `#[path]` child of `text.rs`, so `use super::*` reaches private items.

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
    // Splitting at the comment's `;` would splice the preamble into the
    // comment.
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
    // A `//` comment ends at any WGSL line break (VT/CR/FF), so a directive
    // after one must still hoist; otherwise the preamble path ships it after
    // the preamble's declarations (invalid, exit 0).
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
    // `enable\nf16;` is valid WGSL; reading it as an identifier misses the
    // directive and spuriously fires the f16 preamble guard.
    assert!(has_enable_directive("enable\nf16;", "f16"));
    assert!(has_enable_directive("enable\r\nf16;", "f16"));
    assert!(has_enable_directive("enable f16;", "f16"));
    assert!(!has_enable_directive("enablef16;", "f16"));
}

/// A fragment without a trailing newline (a source ending in a directive)
/// spliced before the preamble's directives would glue them into one syntax
/// error.
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

/// `ends_with('\n')` matches the LF of `\r\n`, so no extra newline follows a
/// CRLF-ending fragment.
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
    // Compact output is one physical line; the splitter must stop after the
    // directive's `;`.
    let src = "enable f16;@group(0)@binding(0)var<storage,read_write>A:f16;\
                   @compute @workgroup_size(1) fn m(){A=1h;}";
    let (dirs, body) = split_directives(src);
    assert_eq!(dirs, "enable f16;");
    assert_eq!(
        body,
        "@group(0)@binding(0)var<storage,read_write>A:f16;\
             @compute @workgroup_size(1) fn m(){A=1h;}"
    );
    // Directives run together on one line are all consumed.
    let (dirs, body) =
        split_directives("enable f16;enable dual_source_blending;@fragment fn m(){}");
    assert_eq!(dirs, "enable f16;enable dual_source_blending;");
    assert_eq!(body, "@fragment fn m(){}");
}

#[test]
fn split_directives_word_boundary_single_line() {
    assert_eq!(
        split_directives("enablef16;fn m(){}"),
        ("", "enablef16;fn m(){}")
    );
    assert_eq!(
        split_directives("requires_foo();fn m(){}"),
        ("", "requires_foo();fn m(){}")
    );
    assert_eq!(
        split_directives("enable f16;diagnostic_counter_thing fn m(){}"),
        ("enable f16;", "diagnostic_counter_thing fn m(){}")
    );
}

#[test]
fn references_f16_token_ignores_identifiers() {
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
    // `h` suffix spellings, no alias or keyword.
    assert!(references_f16_token("let x = 1.0h + 2.0h;"));
    assert!(references_f16_token("let x = 0h;"));
    assert!(references_f16_token("let x = 1.5e2h;"));
    assert!(references_f16_token("let x = 1.0e-3h;"));
    assert!(references_f16_token("let x = 0x1p2h;"));
    // Negatives.
    assert!(!references_f16_token("var width: f32; var height: f32;"));
    assert!(!references_f16_token("let mesh = 1.0;"));
    assert!(!references_f16_token("var vec2hh: i32;"));
    assert!(!references_f16_token("let x = 1.0f + 2u + 3;"));
    assert!(!references_f16_token("let x = 1.0e-3;"));
    assert!(!references_f16_token("fn vec2h_helper() {}"));
}

#[test]
fn references_f16_token_ignores_comments() {
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
    assert!(references_f16_token(
        "/* nest /* inner */ done */ var x: f16 = 0h;"
    ));
}

#[test]
fn normalize_line_endings_handles_lone_cr() {
    assert_eq!(normalize_line_endings("a\rb\rc"), "a\nb\nc");
    // `\r\n` stays: `str::lines` handles it.
    assert_eq!(normalize_line_endings("a\r\nb\r\nc"), "a\r\nb\r\nc");
    assert_eq!(normalize_line_endings("a\rb\r\nc\nd"), "a\nb\r\nc\nd");
    assert_eq!(normalize_line_endings("a\nb\nc"), "a\nb\nc");
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
    // A false negative hard-errors the preamble guard in `run` on valid input.
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
    // Several directives may share one physical line.
    assert!(has_enable_f16_directive(
        "enable dual_source_blending; enable f16;"
    ));
    assert!(has_enable_f16_directive(
        "enable clip_distances; enable f16, subgroups;"
    ));
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
    // A line opening with a `diagnostic_`-prefixed identifier is pathological
    // (top-level WGSL starts with a keyword) but must still not hoist.
    let source = "diagnostic_counter_alias\nfn main(){}\n";
    let (dirs, body) = split_directives(source);
    assert_eq!(
        dirs, "",
        "identifier starting with 'diagnostic' must not be treated as a directive"
    );
    assert_eq!(body, source);
}

#[test]
fn requires_entry_spans_blank_the_directive_or_one_entry() {
    let ext = "unrestricted_pointer_parameters";
    let blank = |src: &str| -> String {
        let mut out = src.to_string();
        for span in requires_entry_spans(src, ext) {
            out.replace_range(span.clone(), &" ".repeat(span.len()));
        }
        out
    };
    // Sole entry: the whole directive goes, `;` and a trailing comma included.
    assert_eq!(
        blank("requires unrestricted_pointer_parameters;\nfn f() {}\n"),
        format!("{}\nfn f() {{}}\n", " ".repeat(41))
    );
    assert_eq!(
        blank("requires unrestricted_pointer_parameters,;"),
        " ".repeat(42)
    );
    // Listed with others: through to the next entry, or back to the previous
    // one and on to the `;` (a trailing comma goes with it).
    assert_eq!(
        blank("requires unrestricted_pointer_parameters, pointer_composite_access;"),
        format!("requires {}pointer_composite_access;", " ".repeat(33))
    );
    assert_eq!(
        blank("requires pointer_composite_access, unrestricted_pointer_parameters;"),
        format!("requires pointer_composite_access{};", " ".repeat(33))
    );
    assert_eq!(
        blank("requires pointer_composite_access, unrestricted_pointer_parameters,;"),
        format!("requires pointer_composite_access{};", " ".repeat(34))
    );
    // Longer identifiers, and directives after a declaration, are left alone.
    assert!(requires_entry_spans("requires unrestricted_pointer_parameters_v2;", ext).is_empty());
    assert!(
        requires_entry_spans("fn f() {}\nrequires unrestricted_pointer_parameters;", ext)
            .is_empty()
    );
}

#[test]
fn vertical_tab_separates_a_directive_keyword() {
    // WGSL blankspace includes VT and FF, which `is_ascii_whitespace` and a
    // hand-written space list both miss; tint accepts `enable<VT>f16;`.
    for sep in ['\u{0B}', '\u{0C}', ' ', '\t', '\n'] {
        let src = format!("enable{sep}f16;");
        assert!(
            cleaned_has_enable_directive(&src, "f16"),
            "separator {:?} must delimit the keyword",
            sep
        );
        assert_eq!(
            split_directives(&src).0,
            src,
            "separator {:?} must keep the directive in the prefix",
            sep
        );
    }
    assert!(!cleaned_has_enable_directive("enablef16;", "f16"));
}

#[test]
fn vertical_tab_separates_a_requires_entry() {
    let src = "requires\u{0B}unrestricted_pointer_parameters;";
    assert_eq!(
        requires_entry_spans(src, "unrestricted_pointer_parameters").len(),
        1
    );
}
