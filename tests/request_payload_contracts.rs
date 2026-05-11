//! Integration tests pinning the wire shapes for
//! `POST /workspace/{id}/train` and `POST /workspace/{id}/convert`.
//!
//! Coverage:
//! - Train body is the flattened `TrainingCfg` (no wrapper, no
//!   `dataset_path`); a wrapper shape parse-fails; numeric range
//!   gates surface as `ValidationError`.
//! - Convert body is internally tagged on `converter_type`; a
//!   flat shape (no discriminator) parse-fails; per-variant
//!   `deny_unknown_fields` rejects strays after dispatch.
//! - Every converter-provided file path is rooted under the
//!   converter tree.  Canonical wire form is slashless; a legacy
//!   leading `/` is stripped during the compatibility window;
//!   traversal sequences fail closed at deserialize.
//! - Manifest helper round-trip for [`TrainingCfg`] stays canonical
//!   under serialize/deserialize key-order changes.

#![allow(clippy::disallowed_methods)]

use acoustics_lab::file_mgr::{
    ConvertRequest, ConverterPath, LabelsFormat, MAX_BATCH_SIZE, MAX_CONVERT_SHARDS, MAX_EPOCHS,
    MAX_LEARNING_RATE, TfjsConvertParams, TrainRequest, TrainingCfg, ValidationError,
    canonical_training_cfg_sha256, from_manifest_value, to_manifest_value,
    validate_convert_request, validate_training_cfg,
};

// MARK: TrainRequest (= flattened TrainingCfg)

#[test]
fn train_request_round_2_flat_body_parses() {
    // Body is the flat TrainingCfg (no wrapper).
    let body = r#"{
        "epochs": 4,
        "batch_size": 16,
        "learning_rate": 0.001,
        "seed": 42
    }"#;
    let req: TrainRequest = serde_json::from_str(body).expect("flat body parses");
    assert_eq!(req.epochs, 4);
    assert_eq!(req.batch_size, 16);
    assert!((req.learning_rate - 1e-3).abs() < 1e-9);
    assert_eq!(req.seed, Some(42));
    validate_training_cfg(&req).expect("body within bounds");
}

#[test]
fn train_request_seed_is_optional() {
    let body = r#"{"epochs":1,"batch_size":1,"learning_rate":0.5}"#;
    let req: TrainRequest = serde_json::from_str(body).expect("seed defaults to None");
    assert_eq!(req.seed, None);
    validate_training_cfg(&req).expect("body within bounds");
}

#[test]
fn train_request_rejects_round_1_wrapper_shape() {
    // Legacy `{dataset_path, training_cfg}` body: both keys are
    // unknown to the flat shape under `deny_unknown_fields`.
    let body = r#"{
        "dataset_path": "audio",
        "training_cfg": {"epochs":1,"batch_size":1,"learning_rate":0.001}
    }"#;
    let res: Result<TrainRequest, _> = serde_json::from_str(body);
    assert!(res.is_err(), "wrapper body must parse-fail");
}

#[test]
fn train_request_rejects_unknown_top_level_field() {
    let body = r#"{
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.001,
        "momentum": 0.9
    }"#;
    let res: Result<TrainRequest, _> = serde_json::from_str(body);
    assert!(res.is_err(), "stray field is a 400");
}

// MARK: TrainingCfg numeric validators

#[test]
fn validate_training_cfg_pins_redesign_bounds() {
    for cfg in [
        TrainingCfg {
            epochs: 1,
            batch_size: 1,
            learning_rate: 1e-9,
            seed: None,
            validation_split: 0.0,
        },
        TrainingCfg {
            epochs: MAX_EPOCHS,
            batch_size: MAX_BATCH_SIZE,
            learning_rate: MAX_LEARNING_RATE,
            seed: Some(0),
            validation_split: 0.999,
        },
    ] {
        validate_training_cfg(&cfg).expect("boundary cfg within bounds");
    }

    for (cfg, expected) in [
        (
            TrainingCfg {
                epochs: 0,
                batch_size: 1,
                learning_rate: 1e-3,
                seed: None,
                validation_split: 0.0,
            },
            "EpochsOutOfRange",
        ),
        (
            TrainingCfg {
                epochs: 1,
                batch_size: MAX_BATCH_SIZE + 1,
                learning_rate: 1e-3,
                seed: None,
                validation_split: 0.0,
            },
            "BatchSizeOutOfRange",
        ),
        (
            TrainingCfg {
                epochs: 1,
                batch_size: 1,
                learning_rate: f32::NAN,
                seed: None,
                validation_split: 0.0,
            },
            "LearningRateOutOfRange",
        ),
        (
            TrainingCfg {
                epochs: 1,
                batch_size: 1,
                learning_rate: 1e-3,
                seed: None,
                validation_split: -0.1,
            },
            "ValidationSplitOutOfRange",
        ),
        (
            TrainingCfg {
                epochs: 1,
                batch_size: 1,
                learning_rate: 1e-3,
                seed: None,
                validation_split: 1.0,
            },
            "ValidationSplitOutOfRange",
        ),
        (
            TrainingCfg {
                epochs: 1,
                batch_size: 1,
                learning_rate: 1e-3,
                seed: None,
                validation_split: f32::NAN,
            },
            "ValidationSplitOutOfRange",
        ),
    ] {
        let err = validate_training_cfg(&cfg).expect_err("expected reject");
        let s = format!("{err:?}");
        assert!(s.contains(expected), "got {err:?}, expected {expected}");
    }
}

/// `validation_split` defaults to `0.0` when omitted from the
/// wire, matching the pre-feature behavior (no holdout,
/// last-epoch head).  Pinned so a future caller-facing change
/// to the default surfaces in CI rather than as a silent
/// trainer-behavior shift.
#[test]
fn validation_split_default_is_zero_when_omitted() {
    let body = r#"{"epochs": 4, "batch_size": 16, "learning_rate": 0.001}"#;
    let cfg: TrainingCfg = serde_json::from_str(body).expect("parse");
    assert_eq!(cfg.validation_split, 0.0);
    validate_training_cfg(&cfg).expect("default cfg validates");
}

/// A non-zero `validation_split` round-trips through JSON
/// without precision loss inside the validator's accepted
/// half-open `[0.0, 1.0)` range.  Pins the wire-shape promise
/// to clients that drive the trainer's stratified-split path.
#[test]
fn validation_split_round_trips_through_json() {
    let body =
        r#"{"epochs": 4, "batch_size": 16, "learning_rate": 0.001, "validation_split": 0.25}"#;
    let cfg: TrainingCfg = serde_json::from_str(body).expect("parse");
    assert_eq!(cfg.validation_split, 0.25);
    validate_training_cfg(&cfg).expect("0.25 split validates");
}

// MARK: ConverterPath wire shape

#[test]
fn converter_path_accepts_both_canonical_and_legacy_forms() {
    // Canonical wire form (slashless) is the documented shape.
    let p: ConverterPath =
        serde_json::from_str(r#""tfjs/model.json""#).expect("canonical form parses");
    assert_eq!(p.workspace_path().as_str(), "converters/tfjs/model.json");

    // Legacy leading slash still parses for one release of overlap.
    let p_legacy: ConverterPath =
        serde_json::from_str(r#""/tfjs/model.json""#).expect("legacy form parses");
    assert_eq!(p, p_legacy);

    // Empty / lone slash still reject -- a converter request must
    // name an actual file under `<workspace>/converters/`.
    for bad in [r#""""#, r#""/""#] {
        let res: Result<ConverterPath, _> = serde_json::from_str(bad);
        assert!(res.is_err(), "{bad:?} must reject");
    }
}

#[test]
fn converter_path_round_trips_canonical_form() {
    let p: ConverterPath =
        serde_json::from_str(r#""tfjs/model.json""#).expect("canonical form parses");
    assert_eq!(p.workspace_path().as_str(), "converters/tfjs/model.json");
    assert_eq!(p.wire_form(), "tfjs/model.json");
    let back = serde_json::to_string(&p).unwrap();
    assert_eq!(back, r#""tfjs/model.json""#);
}

// MARK: ConvertRequest dispatched on converter_type

#[test]
fn convert_request_round_2_tfjs_body_parses() {
    let body = r#"{
        "converter_type": "tfjs",
        "model_json_path": "/tfjs/model.json",
        "shards": ["/tfjs/group1-shard1of2.bin", "/tfjs/group1-shard2of2.bin"],
        "labels_path": "/tfjs/metadata.json",
        "labels_format": "tfjs_metadata"
    }"#;
    let req: ConvertRequest = serde_json::from_str(body).expect("body parses");
    let ConvertRequest::Tfjs(p) = &req;
    assert_eq!(p.shards.len(), 2);
    assert_eq!(p.labels_format, LabelsFormat::TfjsMetadata);
    validate_convert_request(&req).expect("within bounds");
}

#[test]
fn convert_request_round_2_lines_format_parses() {
    let body = r#"{
        "converter_type": "tfjs",
        "model_json_path": "/tfjs/model.json",
        "shards": ["/tfjs/shard.bin"],
        "labels_path": "/tfjs/labels.txt",
        "labels_format": "lines"
    }"#;
    let req: ConvertRequest = serde_json::from_str(body).expect("body parses");
    let ConvertRequest::Tfjs(p) = &req;
    assert_eq!(p.labels_format, LabelsFormat::Lines);
}

#[test]
fn convert_request_rejects_unknown_converter_type() {
    let body = r#"{
        "converter_type": "onnx",
        "model_json_path": "/m",
        "shards": ["/s"],
        "labels_path": "/l",
        "labels_format": "lines"
    }"#;
    let res: Result<ConvertRequest, _> = serde_json::from_str(body);
    assert!(res.is_err(), "unknown converter_type must parse-fail");
}

#[test]
fn convert_request_rejects_body_without_converter_type() {
    // The discriminator is mandatory; a flat body without
    // `converter_type` fails parse regardless of path form.
    let body = r#"{
        "model_json_path": "tfjs/model.json",
        "shards": ["tfjs/shard.bin"],
        "labels_path": "tfjs/labels.txt",
        "labels_format": "lines"
    }"#;
    let res: Result<ConvertRequest, _> = serde_json::from_str(body);
    assert!(res.is_err(), "body without converter_type must parse-fail",);
}

#[test]
fn convert_request_rejects_unknown_field_after_dispatch() {
    let body = r#"{
        "converter_type": "tfjs",
        "model_json_path": "/m",
        "shards": ["/s"],
        "labels_path": "/l",
        "labels_format": "lines",
        "stray": true
    }"#;
    let res: Result<ConvertRequest, _> = serde_json::from_str(body);
    assert!(res.is_err(), "stray field must be rejected");
}

#[test]
fn convert_request_accepts_canonical_slashless_paths() {
    // Both canonical (slashless) and legacy (leading `/`) forms
    // parse to the same workspace_path.
    for field in ["model_json_path", "labels_path"] {
        let mut v = serde_json::json!({
            "converter_type": "tfjs",
            "model_json_path": "/m",
            "shards": ["/s"],
            "labels_path": "/l",
            "labels_format": "lines",
        });
        v[field] = serde_json::Value::String("relative/path".into());
        let body = serde_json::to_string(&v).unwrap();
        let req: ConvertRequest = serde_json::from_str(&body).expect("slashless path is canonical");
        let ConvertRequest::Tfjs(p) = &req;
        let bound = match field {
            "model_json_path" => p.model_json_path.workspace_path().as_str(),
            "labels_path" => p.labels_path.workspace_path().as_str(),
            _ => unreachable!(),
        };
        assert_eq!(bound, "converters/relative/path");
    }

    let body = r#"{
        "converter_type": "tfjs",
        "model_json_path": "/m",
        "shards": ["relative/shard"],
        "labels_path": "/l",
        "labels_format": "lines"
    }"#;
    let req: ConvertRequest = serde_json::from_str(body).expect("slashless shard is canonical");
    let ConvertRequest::Tfjs(p) = &req;
    assert_eq!(
        p.shards[0].workspace_path().as_str(),
        "converters/relative/shard"
    );
}

#[test]
fn convert_request_rejects_traversal_in_paths() {
    for bad in [
        // model_json_path
        r#"{"converter_type":"tfjs","model_json_path":"/..","shards":["/s"],"labels_path":"/l","labels_format":"lines"}"#,
        // shards entry
        r#"{"converter_type":"tfjs","model_json_path":"/m","shards":["/../etc/passwd"],"labels_path":"/l","labels_format":"lines"}"#,
        // labels_path
        r#"{"converter_type":"tfjs","model_json_path":"/m","shards":["/s"],"labels_path":"/.hidden/file","labels_format":"lines"}"#,
        // unknown labels_format variant
        r#"{"converter_type":"tfjs","model_json_path":"/m","shards":["/s"],"labels_path":"/l","labels_format":"raw"}"#,
        // URL-encoded traversal
        r#"{"converter_type":"tfjs","model_json_path":"/%2E%2E/etc","shards":["/s"],"labels_path":"/l","labels_format":"lines"}"#,
    ] {
        let res: Result<ConvertRequest, _> = serde_json::from_str(bad);
        assert!(res.is_err(), "{bad:?} must be rejected");
    }
}

#[test]
fn validate_convert_request_pins_shard_bounds() {
    // Empty shards rejected.
    let req = ConvertRequest::Tfjs(TfjsConvertParams {
        model_json_path: "/m".parse().unwrap(),
        shards: vec![],
        labels_path: "/l".parse().unwrap(),
        labels_format: LabelsFormat::Lines,
    });
    assert_eq!(
        validate_convert_request(&req),
        Err(ValidationError::NoShards)
    );

    // One shard accepted.
    let req = ConvertRequest::Tfjs(TfjsConvertParams {
        model_json_path: "/m".parse().unwrap(),
        shards: vec!["/s".parse().unwrap()],
        labels_path: "/l".parse().unwrap(),
        labels_format: LabelsFormat::Lines,
    });
    validate_convert_request(&req).expect("single-shard accepted");

    // Cap+1 rejected.
    let mut shards = Vec::with_capacity(MAX_CONVERT_SHARDS + 1);
    for i in 0..=MAX_CONVERT_SHARDS {
        shards.push(format!("/s{i}").parse().unwrap());
    }
    let req = ConvertRequest::Tfjs(TfjsConvertParams {
        model_json_path: "/m".parse().unwrap(),
        shards,
        labels_path: "/l".parse().unwrap(),
        labels_format: LabelsFormat::Lines,
    });
    assert!(matches!(
        validate_convert_request(&req),
        Err(ValidationError::TooManyShards { .. })
    ));
}

// MARK: Manifest helper round-trip

#[test]
fn manifest_value_round_trip_is_canonical() {
    let cfg = TrainingCfg {
        epochs: 4,
        batch_size: 16,
        learning_rate: 1e-3,
        seed: Some(42),
        validation_split: 0.2,
    };
    let value = to_manifest_value(&cfg);
    let back = from_manifest_value(&value).expect("round-trip parses");
    assert_eq!(cfg, back);
    assert_eq!(
        canonical_training_cfg_sha256(&cfg),
        canonical_training_cfg_sha256(&back),
    );

    // Reordered JSON parses back to the same struct -> same hash.
    let reordered: serde_json::Value = serde_json::from_str(
        r#"{"learning_rate":0.001,"seed":42,"validation_split":0.2,"epochs":4,"batch_size":16}"#,
    )
    .unwrap();
    let back2 = from_manifest_value(&reordered).expect("reordered parses");
    assert_eq!(
        canonical_training_cfg_sha256(&cfg),
        canonical_training_cfg_sha256(&back2),
    );
}
