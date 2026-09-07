//! Short-identifier generator for the rename and mangling passes: bijective
//! numeration over a const-sorted table of every WGSL keyword, reserved
//! word, predeclared type and built-in function.

use std::collections::HashSet;

const RESERVED_SORTED_LEN: usize = WGSL_RESERVED.len() + WGSL_PREDECLARED.len();

/// `<str as Ord>` is not const-callable.
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

/// Both source tables merged and sorted in const-eval, so [`is_reserved`]
/// is a `binary_search` over a `.rodata` array with no runtime init.
/// Binary search beats a `HashSet` here: the hot caller probes 1-3
/// character identifiers, where SipHash setup dwarfs the ~9 compares that
/// short-circuit on the first byte.  The source arrays stay in curated,
/// categorised order; the sorted, duplicate-free invariant is enforced at
/// compile time by the `const _` assertion.
const RESERVED_SORTED: [&str; RESERVED_SORTED_LEN] = {
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
    // Insertion sort: N < 400, so ~125k byte-compares at compile time is
    // negligible.
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

/// Strictly ascending adjacent pairs prove sorted and duplicate-free; a
/// duplicate in either source array fails the build instead of degrading
/// `binary_search`.
const _: () = {
    let mut i = 1;
    while i < RESERVED_SORTED_LEN {
        assert!(
            const_str_lt(RESERVED_SORTED[i - 1], RESERVED_SORTED[i]),
            "RESERVED_SORTED must be strictly ascending and duplicate-free"
        );
        i += 1;
    }
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

/// <https://www.w3.org/TR/WGSL/#keyword-summary> and
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
    "non_coherent",
    "noncoherent",
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
    "priv",
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

/// Predeclared types and built-in functions a generated identifier must not
/// shadow: <https://www.w3.org/TR/WGSL/#predeclared-types> and
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
    "textureSampleBaseClampToEdge",
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
    "pack4xI8",
    "pack4xI8Clamp",
    "pack4xU8",
    "pack4xU8Clamp",
    "unpack2x16float",
    "unpack2x16snorm",
    "unpack2x16unorm",
    "unpack4x8snorm",
    "unpack4x8unorm",
    "unpack4xI8",
    "unpack4xU8",
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

// MARK: Module name census

/// Every named module-scope declaration the rename pass owns; types and
/// struct members are the generator's namespace.
pub(crate) fn module_scope_names(module: &naga::Module) -> impl Iterator<Item = &str> {
    module
        .constants
        .iter()
        .filter_map(|(_, c)| c.name.as_deref())
        .chain(
            module
                .overrides
                .iter()
                .filter_map(|(_, o)| o.name.as_deref()),
        )
        .chain(
            module
                .global_variables
                .iter()
                .filter_map(|(_, g)| g.name.as_deref()),
        )
        .chain(
            module
                .functions
                .iter()
                .filter_map(|(_, f)| f.name.as_deref()),
        )
        .chain(module.entry_points.iter().map(|ep| ep.name.as_str()))
}

pub(crate) fn type_names(module: &naga::Module) -> impl Iterator<Item = &str> {
    module.types.iter().filter_map(|(_, ty)| ty.name.as_deref())
}

pub(crate) fn struct_member_names(module: &naga::Module) -> impl Iterator<Item = &str> {
    module
        .types
        .iter()
        .flat_map(|(_, ty)| match &ty.inner {
            naga::TypeInner::Struct { members, .. } => members.as_slice(),
            _ => &[],
        })
        .filter_map(|m| m.name.as_deref())
}

/// Argument and local-variable names of one function body; a module-scope
/// name minted equal to one would be shadowed inside that function.
pub(crate) fn function_local_names(func: &naga::Function) -> impl Iterator<Item = &str> {
    func.arguments
        .iter()
        .filter_map(|a| a.name.as_deref())
        .chain(
            func.local_variables
                .iter()
                .filter_map(|(_, l)| l.name.as_deref()),
        )
}

// MARK: Name generation

/// Bijective numeration (first char from 52 letters, later chars from 63
/// symbols), so distinct counters give distinct names.
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

fn is_reserved(name: &str) -> bool {
    RESERVED_SORTED.binary_search(&name).is_ok()
}

/// Advance `counter` to the next identifier that is neither a WGSL
/// reserved nor a predeclared name.
pub fn next_name(counter: &mut usize) -> String {
    loop {
        let name = name_from_counter(*counter);
        *counter += 1;
        if !is_reserved(&name) {
            return name;
        }
    }
}

/// [`next_name`] skipping `used` as well; the result is not inserted.
pub fn next_name_unique(counter: &mut usize, used: &HashSet<String>) -> String {
    loop {
        let name = next_name(counter);
        if !used.contains(&name) {
            return name;
        }
    }
}

/// [`next_name_unique`] that also claims the name in `used`.
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

    /// Every pass treats [`is_reserved`] as the sole cannot-shadow oracle.
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
            // data packing (8-bit packed integer ops) + clamp-to-edge sample
            "pack4xI8",
            "pack4xI8Clamp",
            "pack4xU8",
            "pack4xU8Clamp",
            "unpack4xI8",
            "unpack4xU8",
            "textureSampleBaseClampToEdge",
        ] {
            assert!(
                is_reserved(name),
                "extended built-in name \"{name}\" must be reserved",
            );
        }
    }

    /// A duplicate in the hand-maintained list usually masks an omission.
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

    /// Checks the const sort against `str`'s own `Ord`, which the `const _`
    /// assertion (built on `const_str_lt`) cannot.
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
