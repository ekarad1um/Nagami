//! Short-identifier generator used by the rename and mangling passes.
//!
//! Produces compact, unique names via bijective numeration while
//! avoiding every WGSL keyword, reserved word, predeclared type, and
//! built-in function.  The source-of-truth tables live in this module;
//! the binary-search lookup and const-sort are the performance-critical
//! machinery that keeps [`next_name`] cheap in the inner rename loop.

use std::collections::HashSet;

/// Length of [`RESERVED_SORTED`], equal to the sum of both source tables.
///
/// Exposed as a named constant so the fixed-size array binding and the
/// merge loop below share a single length expression.  The actual table
/// is built and sorted at compile time; see [`RESERVED_SORTED`] for the
/// design rationale.
const RESERVED_SORTED_LEN: usize = WGSL_RESERVED.len() + WGSL_PREDECLARED.len();

/// `&str` lexicographic less-than usable in a `const` context.
/// `<str as Ord>` is not yet const-callable, so we open-code a byte-wise
/// comparison that short-circuits on the first differing byte.
const fn const_str_lt(a: &str, b: &str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    let n = if a.len() < b.len() { a.len() } else { b.len() };
    let mut i = 0;
    while i < n {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
        i += 1;
    }
    a.len() < b.len()
}

/// Merged and sorted reserved-word table, built entirely in const-eval
/// so [`is_reserved`] reduces to a pure `slice::binary_search` with no
/// runtime initialisation cost.
///
/// Design choices:
///
/// - **Binary search over `HashSet`** - the hot caller [`next_name`]
///   probes 1-3 character identifiers; hashing such tiny strings still
///   pays the `SipHash` setup, whereas binary search short-circuits on the
///   first byte and resolves in ~9 compares for `N < 400`.  The sorted
///   contiguous `[&'static str; N]` also lives in `.rodata`.
/// - **No `LazyLock`** - both source arrays are `const`, so const-eval
///   performs the merge and sort during `rustc` compilation.  The result
///   skips atomic loads, first-call init branches, and heap allocation.
///
/// The `WGSL_RESERVED` and `WGSL_PREDECLARED` source arrays stay in
/// human-curated, categorised order for review-friendliness; the sort
/// invariant is guarded by `reserved_table_is_sorted_and_unique`.
const RESERVED_SORTED: [&str; RESERVED_SORTED_LEN] = {
    // Concatenate both source slices into a fixed-size array.
    let mut arr: [&str; RESERVED_SORTED_LEN] = [""; RESERVED_SORTED_LEN];
    let mut i = 0;
    while i < WGSL_RESERVED.len() {
        arr[i] = WGSL_RESERVED[i];
        i += 1;
    }
    let mut j = 0;
    while j < WGSL_PREDECLARED.len() {
        arr[WGSL_RESERVED.len() + j] = WGSL_PREDECLARED[j];
        j += 1;
    }
    // Const-eval-friendly insertion sort.  N < 400 so worst-case
    // ~125k byte-compares at compile time is negligible for `rustc` and
    // amortised exactly once across the whole binary.  Stable ordering
    // is irrelevant here because duplicates are forbidden and locked by
    // `reserved_table_is_sorted_and_unique`.
    let mut k = 1;
    while k < RESERVED_SORTED_LEN {
        let mut m = k;
        while m > 0 && const_str_lt(arr[m], arr[m - 1]) {
            let tmp = arr[m];
            arr[m] = arr[m - 1];
            arr[m - 1] = tmp;
            m -= 1;
        }
        k += 1;
    }
    arr
};

const FIRST_LETTERS: [char; 52] = [
    'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i', 'J',
    'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's',
    'T', 't', 'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z',
];

const NEXT_LETTERS: [char; 63] = [
    'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i', 'J',
    'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's',
    'T', 't', 'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z', '1', '2', '3', '4', '5',
    '6', '7', '8', '9', '0', '_',
];

// MARK: Reserved and predeclared tables

/// WGSL keywords and reserved words that must never be used as identifiers.
///
/// Source: <https://www.w3.org/TR/WGSL/#keyword-summary> and
/// <https://www.w3.org/TR/WGSL/#reserved-words>.
const WGSL_RESERVED: &[&str] = &[
    // Keywords
    "alias",
    "break",
    "case",
    "const",
    "const_assert",
    "continue",
    "continuing",
    "default",
    "diagnostic",
    "discard",
    "else",
    "enable",
    "false",
    "fn",
    "for",
    "if",
    "let",
    "loop",
    "override",
    "return",
    "struct",
    "switch",
    "true",
    "var",
    "while",
    // Reserved words
    "NULL",
    "Self",
    "abstract",
    "active",
    "alignas",
    "alignof",
    "as",
    "asm",
    "asm_fragment",
    "async",
    "attribute",
    "auto",
    "await",
    "become",
    "binding_array",
    "cast",
    "catch",
    "class",
    "co_await",
    "co_return",
    "co_yield",
    "coherent",
    "column_major",
    "common",
    "compile",
    "compile_fragment",
    "concept",
    "const_cast",
    "consteval",
    "constexpr",
    "constinit",
    "crate",
    "debugger",
    "decltype",
    "delete",
    "demote",
    "demote_to_helper",
    "do",
    "dynamic_cast",
    "enum",
    "explicit",
    "export",
    "extends",
    "extern",
    "external",
    "fallthrough",
    "filter",
    "final",
    "finally",
    "friend",
    "from",
    "fxgroup",
    "get",
    "goto",
    "groupshared",
    "highp",
    "impl",
    "implements",
    "import",
    "in",
    "inline",
    "instanceof",
    "interface",
    "layout",
    "lowp",
    "macro",
    "macro_rules",
    "match",
    "mediump",
    "meta",
    "mod",
    "module",
    "move",
    "mut",
    "mutable",
    "namespace",
    "new",
    "nil",
    "noexcept",
    "noinline",
    "nointerpolation",
    "noperspective",
    "null",
    "nullptr",
    "of",
    "operator",
    "package",
    "packoffset",
    "partition",
    "pass",
    "patch",
    "pixelfragment",
    "precise",
    "precision",
    "premerge",
    "private",
    "protected",
    "pub",
    "public",
    "readonly",
    "ref",
    "regardless",
    "register",
    "reinterpret_cast",
    "require",
    "requires",
    "resource",
    "restrict",
    "self",
    "set",
    "shared",
    "sizeof",
    "smooth",
    "snorm",
    "static",
    "static_assert",
    "static_cast",
    "std",
    "subroutine",
    "super",
    "target",
    "template",
    "this",
    "thread_local",
    "throw",
    "trait",
    "try",
    "type",
    "typedef",
    "typeid",
    "typename",
    "typeof",
    "union",
    "unless",
    "unorm",
    "unsafe",
    "unsized",
    "use",
    "using",
    "varying",
    "virtual",
    "volatile",
    "wgsl",
    "where",
    "with",
    "writeonly",
    "yield",
];

/// WGSL predeclared type names and built-in function names that generated
/// identifiers must never collide with to avoid shadowing visible scopes.
///
/// Source: <https://www.w3.org/TR/WGSL/#predeclared-types> and
/// <https://www.w3.org/TR/WGSL/#builtin-functions>.
const WGSL_PREDECLARED: &[&str] = &[
    // Scalar types
    "bool",
    "f16",
    "f32",
    "f64",
    "i32",
    "i64",
    "u32",
    "u64",
    // Vector / matrix constructor types
    "vec2",
    "vec3",
    "vec4",
    "mat2x2",
    "mat2x3",
    "mat2x4",
    "mat3x2",
    "mat3x3",
    "mat3x4",
    "mat4x2",
    "mat4x3",
    "mat4x4",
    // Convenience type aliases
    "vec2i",
    "vec2u",
    "vec2f",
    "vec2h",
    "vec3i",
    "vec3u",
    "vec3f",
    "vec3h",
    "vec4i",
    "vec4u",
    "vec4f",
    "vec4h",
    "mat2x2f",
    "mat2x2h",
    "mat2x3f",
    "mat2x3h",
    "mat2x4f",
    "mat2x4h",
    "mat3x2f",
    "mat3x2h",
    "mat3x3f",
    "mat3x3h",
    "mat3x4f",
    "mat3x4h",
    "mat4x2f",
    "mat4x2h",
    "mat4x3f",
    "mat4x3h",
    "mat4x4f",
    "mat4x4h",
    // Composite / pointer types
    "array",
    "atomic",
    "ptr",
    // Sampler types
    "sampler",
    "sampler_comparison",
    // Texture types
    "texture_1d",
    "texture_2d",
    "texture_2d_array",
    "texture_3d",
    "texture_cube",
    "texture_cube_array",
    "texture_multisampled_2d",
    "texture_storage_1d",
    "texture_storage_2d",
    "texture_storage_2d_array",
    "texture_storage_3d",
    "texture_depth_2d",
    "texture_depth_2d_array",
    "texture_depth_cube",
    "texture_depth_cube_array",
    "texture_depth_multisampled_2d",
    "texture_external",
    // Built-in functions - value constructors / conversion
    "bitcast",
    // Built-in functions - logical
    "all",
    "any",
    "select",
    // Built-in functions - numeric
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "ceil",
    "clamp",
    "cos",
    "cosh",
    "countLeadingZeros",
    "countOneBits",
    "countTrailingZeros",
    "cross",
    "degrees",
    "determinant",
    "distance",
    "dot",
    "dot4I8Packed",
    "dot4U8Packed",
    "exp",
    "exp2",
    "extractBits",
    "faceForward",
    "firstLeadingBit",
    "firstTrailingBit",
    "floor",
    "fma",
    "fract",
    "frexp",
    "insertBits",
    "inverseSqrt",
    "ldexp",
    "length",
    "log",
    "log2",
    "max",
    "min",
    "mix",
    "modf",
    "normalize",
    "pow",
    "quantizeToF16",
    "radians",
    "reflect",
    "refract",
    "reverseBits",
    "round",
    "saturate",
    "sign",
    "sin",
    "sinh",
    "smoothstep",
    "sqrt",
    "step",
    "tan",
    "tanh",
    "transpose",
    "trunc",
    // Built-in functions - derivative
    "dpdx",
    "dpdxCoarse",
    "dpdxFine",
    "dpdy",
    "dpdyCoarse",
    "dpdyFine",
    "fwidth",
    "fwidthCoarse",
    "fwidthFine",
    // Built-in functions - texture
    "textureDimensions",
    "textureGather",
    "textureGatherCompare",
    "textureLoad",
    "textureNumLayers",
    "textureNumLevels",
    "textureNumSamples",
    "textureSample",
    "textureSampleBias",
    "textureSampleCompare",
    "textureSampleCompareLevel",
    "textureSampleGrad",
    "textureSampleLevel",
    "textureStore",
    // Built-in functions - data packing / unpacking
    "pack2x16float",
    "pack2x16snorm",
    "pack2x16unorm",
    "pack4x8snorm",
    "pack4x8unorm",
    "unpack2x16float",
    "unpack2x16snorm",
    "unpack2x16unorm",
    "unpack4x8snorm",
    "unpack4x8unorm",
    // Built-in functions - synchronization
    "storageBarrier",
    "textureBarrier",
    "workgroupBarrier",
    "workgroupUniformLoad",
    // Built-in functions - array
    "arrayLength",
    // Built-in functions - atomic
    "atomicAdd",
    "atomicAnd",
    "atomicCompareExchangeWeak",
    "atomicExchange",
    "atomicLoad",
    "atomicMax",
    "atomicMin",
    "atomicOr",
    "atomicStore",
    "atomicSub",
    "atomicXor",
    // Built-in functions - texture atomics
    "textureAtomicAdd",
    "textureAtomicAnd",
    "textureAtomicMax",
    "textureAtomicMin",
    "textureAtomicOr",
    "textureAtomicXor",
    // Built-in functions - subgroup
    "subgroupAdd",
    "subgroupAll",
    "subgroupAnd",
    "subgroupAny",
    "subgroupBallot",
    "subgroupBarrier",
    "subgroupBroadcast",
    "subgroupBroadcastFirst",
    "subgroupElect",
    "subgroupExclusiveAdd",
    "subgroupExclusiveMul",
    "subgroupInclusiveAdd",
    "subgroupInclusiveMul",
    "subgroupMax",
    "subgroupMin",
    "subgroupMul",
    "subgroupOr",
    "subgroupShuffle",
    "subgroupShuffleDown",
    "subgroupShuffleUp",
    "subgroupShuffleXor",
    "subgroupXor",
    // Built-in functions - quad
    "quadBroadcast",
    "quadSwapDiagonal",
    "quadSwapX",
    "quadSwapY",
    // Built-in functions - ray query
    "rayQueryConfirmIntersection",
    "rayQueryGenerateIntersection",
    "rayQueryGetCandidateIntersection",
    "rayQueryGetCommittedIntersection",
    "rayQueryInitialize",
    "rayQueryProceed",
    "rayQueryTerminate",
    // Predeclared ray-query / acceleration structure types
    "ray_query",
    "acceleration_structure",
    "RayDesc",
    "RayIntersection",
];

// MARK: Name generation

/// Encode `counter` as a short identifier via bijective numeration.
/// The first character is drawn from 52 letters (A-Z and a-z interleaved)
/// and each subsequent character from 63 symbols (letters, digits, and
/// underscore).  The encoding is a bijection, so distinct counters always
/// produce distinct strings without collisions.
fn name_from_counter(counter: usize) -> String {
    let mut id = counter;
    let mut name = String::from(FIRST_LETTERS[id % FIRST_LETTERS.len()]);
    id /= FIRST_LETTERS.len();
    while id > 0 {
        id -= 1;
        name.push(NEXT_LETTERS[id % NEXT_LETTERS.len()]);
        id /= NEXT_LETTERS.len();
    }
    name
}

/// Return `true` when `name` appears in [`RESERVED_SORTED`].  Binary
/// search is safe because the table's ascending, duplicate-free ordering
/// is locked at compile time and re-checked at runtime by
/// `reserved_table_is_sorted_and_unique`.
fn is_reserved(name: &str) -> bool {
    RESERVED_SORTED.binary_search(&name).is_ok()
}

/// Advance `counter` and return the next non-reserved short identifier.
/// Counter values that encode a WGSL reserved or predeclared name are
/// skipped; the returned string is guaranteed safe to use as an
/// unqualified WGSL identifier.
pub fn next_name(counter: &mut usize) -> String {
    loop {
        let name = name_from_counter(*counter);
        *counter += 1;
        if !is_reserved(&name) {
            return name;
        }
    }
}

/// Like [`next_name`] but additionally skips any name present in `used`.
/// The chosen name is NOT inserted into `used`; the caller decides
/// whether to claim it (see [`next_name_insert`] for the claim variant).
pub fn next_name_unique(counter: &mut usize, used: &HashSet<String>) -> String {
    loop {
        let name = next_name(counter);
        if !used.contains(&name) {
            return name;
        }
    }
}

/// Like [`next_name_unique`] but atomically inserts the chosen name into
/// `used` before returning, so back-to-back callers never pick the same
/// identifier.
pub fn next_name_insert(counter: &mut usize, used: &mut HashSet<String>) -> String {
    loop {
        let name = next_name(counter);
        if used.insert(name.clone()) {
            return name;
        }
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_names_are_single_letters() {
        let mut counter = 0;
        assert_eq!(next_name(&mut counter), "A");
        assert_eq!(next_name(&mut counter), "a");
        assert_eq!(next_name(&mut counter), "B");
        assert_eq!(next_name(&mut counter), "b");
    }

    #[test]
    fn never_produces_wgsl_reserved_words() {
        let mut counter = 0;
        for _ in 0..5000 {
            let name = next_name(&mut counter);
            assert!(
                !is_reserved(&name),
                "generated name \"{name}\" at counter {} is a WGSL reserved word",
                counter - 1
            );
        }
    }

    #[test]
    fn unique_skips_used_names() {
        let mut counter = 0;
        let mut used = HashSet::new();
        used.insert("A".to_string());
        let name = next_name_unique(&mut counter, &used);
        assert_eq!(name, "a", "should skip 'A' which is in used set");
    }

    #[test]
    fn insert_adds_to_used_set() {
        let mut counter = 0;
        let mut used = HashSet::new();
        let name = next_name_insert(&mut counter, &mut used);
        assert_eq!(name, "A");
        assert!(used.contains("A"));
    }

    #[test]
    fn predeclared_type_names_are_reserved() {
        for name in &[
            "bool", "f32", "i32", "u32", "vec2", "vec3", "vec4", "mat4x4", "array", "atomic",
            "ptr", "sampler",
        ] {
            assert!(
                is_reserved(name),
                "predeclared type name \"{name}\" should be reserved"
            );
        }
    }

    #[test]
    fn predeclared_builtin_names_are_reserved() {
        for name in &[
            "abs", "sin", "cos", "dot", "min", "max", "mix", "pow", "exp", "log", "select", "clamp",
        ] {
            assert!(
                is_reserved(name),
                "predeclared built-in function name \"{name}\" should be reserved"
            );
        }
    }

    /// Regression guard: atomic, subgroup, quad, ray-query, and
    /// texture-barrier builtins must all be reserved so short-name
    /// generation cannot collide with them.  Every pass that treats
    /// [`is_reserved`] as the sole "cannot shadow" oracle relies on
    /// this coverage staying complete.
    #[test]
    fn extended_builtin_names_are_reserved() {
        for name in &[
            // atomic
            "atomicAdd",
            "atomicAnd",
            "atomicCompareExchangeWeak",
            "atomicExchange",
            "atomicLoad",
            "atomicMax",
            "atomicMin",
            "atomicOr",
            "atomicStore",
            "atomicSub",
            "atomicXor",
            // texture atomics
            "textureAtomicAdd",
            "textureAtomicAnd",
            "textureAtomicMax",
            "textureAtomicMin",
            "textureAtomicOr",
            "textureAtomicXor",
            // subgroup
            "subgroupAdd",
            "subgroupAll",
            "subgroupAnd",
            "subgroupAny",
            "subgroupBallot",
            "subgroupBarrier",
            "subgroupBroadcast",
            "subgroupBroadcastFirst",
            "subgroupElect",
            "subgroupExclusiveAdd",
            "subgroupExclusiveMul",
            "subgroupInclusiveAdd",
            "subgroupInclusiveMul",
            "subgroupMax",
            "subgroupMin",
            "subgroupMul",
            "subgroupOr",
            "subgroupShuffle",
            "subgroupShuffleDown",
            "subgroupShuffleUp",
            "subgroupShuffleXor",
            "subgroupXor",
            // quad
            "quadBroadcast",
            "quadSwapDiagonal",
            "quadSwapX",
            "quadSwapY",
            // ray query
            "rayQueryConfirmIntersection",
            "rayQueryGenerateIntersection",
            "rayQueryGetCandidateIntersection",
            "rayQueryGetCommittedIntersection",
            "rayQueryInitialize",
            "rayQueryProceed",
            "rayQueryTerminate",
            // ray query predeclared types
            "ray_query",
            "acceleration_structure",
            "RayDesc",
            "RayIntersection",
            // synchronization
            "textureBarrier",
        ] {
            assert!(
                is_reserved(name),
                "extended built-in name \"{name}\" must be reserved",
            );
        }
    }

    /// Guard against accidental duplicates in [`WGSL_PREDECLARED`].
    /// The list is hand-maintained, and duplicates typically signal a
    /// copy-paste mistake that may mask a real omission elsewhere.
    #[test]
    fn predeclared_list_has_no_duplicates() {
        let mut seen = HashSet::new();
        for &name in WGSL_PREDECLARED {
            assert!(
                seen.insert(name),
                "duplicate entry in WGSL_PREDECLARED: {name:?}",
            );
        }
    }

    /// Sibling of `predeclared_list_has_no_duplicates` for [`WGSL_RESERVED`].
    #[test]
    fn reserved_list_has_no_duplicates() {
        let mut seen = HashSet::new();
        for &name in WGSL_RESERVED {
            assert!(
                seen.insert(name),
                "duplicate entry in WGSL_RESERVED: {name:?}",
            );
        }
    }

    /// [`is_reserved`] relies on [`RESERVED_SORTED`] being strictly
    /// ascending and duplicate-free.  The sort runs in const-eval, but
    /// we re-verify at runtime so any future tweak to the const-sort
    /// that breaks ordering or introduces a duplicate fails fast here.
    #[test]
    fn reserved_table_is_sorted_and_unique() {
        let table: &[&str] = &RESERVED_SORTED;
        assert_eq!(
            table.len(),
            WGSL_RESERVED.len() + WGSL_PREDECLARED.len(),
            "merged table length must equal sum of source lists",
        );
        for window in table.windows(2) {
            assert!(
                window[0] < window[1],
                "RESERVED_SORTED is not strictly ascending: {:?} >= {:?}",
                window[0],
                window[1],
            );
        }
    }

    /// Equivalence check spanning both source lists: every entry must
    /// resolve as reserved, and a handful of obvious non-reserved
    /// identifiers (including the first generated names) must not.
    /// Locks the behavioural contract for future refactors of the
    /// merged lookup table.
    #[test]
    fn is_reserved_matches_source_lists() {
        for &name in WGSL_RESERVED.iter().chain(WGSL_PREDECLARED.iter()) {
            assert!(is_reserved(name), "{name:?} should be reserved");
        }
        for name in &["A", "a", "B", "b", "Aa", "zz", "x_1"] {
            assert!(!is_reserved(name), "{name:?} should NOT be reserved");
        }
    }
}
