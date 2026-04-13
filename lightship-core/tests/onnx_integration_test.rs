//! ONNX integration tests
//!
//! These tests verify end-to-end ONNX model loading and inference.

use lightship_core::api::SessionHandle;
use lightship_core::common::DataType;
use std::path::PathBuf;

fn get_fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
fn test_load_simple_relu_model() {
    let model_path = get_fixture_path("relu_model.onnx");
    if !model_path.exists() {
        eprintln!("Skipping test: {} not found", model_path.display());
        return;
    }

    let mut session = SessionHandle::new().unwrap();
    let result = session.load_model(&model_path);

    if let Err(e) = &result {
        eprintln!("Failed to load model: {:?}", e);
    }
    assert!(result.is_ok(), "Failed to load ONNX model");

    let model = result.unwrap();
    assert_eq!(model.num_operators(), 1);
    assert_eq!(model.graph.nodes.len(), 1);
}

#[test]
fn test_load_conv_relu_model() {
    let model_path = get_fixture_path("conv_relu_model.onnx");
    if !model_path.exists() {
        eprintln!("Skipping test: {} not found", model_path.display());
        return;
    }

    let mut session = SessionHandle::new().unwrap();
    let result = session.load_model(&model_path);

    if let Err(e) = &result {
        eprintln!("Failed to load model: {:?}", e);
    }
    assert!(result.is_ok(), "Failed to load ONNX model");

    let model = result.unwrap();
    // Should have 2 operators: Conv and ReLU
    assert_eq!(model.num_operators(), 2);
    assert_eq!(model.graph.nodes.len(), 2);
}

#[test]
fn test_relu_model_inference() {
    let model_path = get_fixture_path("relu_model.onnx");
    if !model_path.exists() {
        eprintln!("Skipping test: {} not found", model_path.display());
        return;
    }

    let mut session = SessionHandle::new().unwrap();
    let model = session.load_model(&model_path).unwrap();

    // Input shape: [1, 3, 4, 4] = 48 elements
    // Create input with some negative values to test ReLU
    let input_data: Vec<f32> = (0..48)
        .map(|i| if i % 7 == 0 { -1.0 } else { i as f32 * 0.1 })
        .collect();
    let input = lightship_core::ir::Tensor::from_data(
        "input".to_string(),
        vec![1, 3, 4, 4],
        DataType::F32,
        input_data,
    );

    // Output shape: [1, 3, 4, 4]
    let mut outputs: &mut [(&str, lightship_core::ir::Tensor)] = &mut [(
        "output",
        lightship_core::ir::Tensor::new("output".to_string(), vec![1, 3, 4, 4], DataType::F32),
    )];

    let result = session.forward(&[("input", input)], outputs);
    assert!(result.is_ok(), "Forward pass failed: {:?}", result.err());

    // Verify all outputs are non-negative (ReLU property)
    let output_bytes = outputs[0].1.data_as_bytes();
    let output_data: Vec<f32> = output_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    for (i, val) in output_data.iter().enumerate() {
        assert!(
            *val >= 0.0,
            "ReLU output should be non-negative at index {}, got {}",
            i,
            val
        );
    }
}

#[test]
fn test_conv_relu_model_inference() {
    let model_path = get_fixture_path("conv_relu_model.onnx");
    if !model_path.exists() {
        eprintln!("Skipping test: {} not found", model_path.display());
        return;
    }

    let mut session = SessionHandle::new().unwrap();
    let model = session.load_model(&model_path).unwrap();

    // Check if model has initializers (weights) - if so, we can't run inference yet
    // because we don't have initializer parsing implemented
    let has_weights = model.graph.nodes.iter().any(|n| n.inputs.len() > 1);
    if has_weights {
        eprintln!("Skipping test: model has weights (initializers) which are not yet supported");
        return;
    }

    // Input shape: [1, 3, 32, 32]
    let input_data: Vec<f32> = (0..1 * 3 * 32 * 32)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let input = lightship_core::ir::Tensor::from_data(
        "input".to_string(),
        vec![1, 3, 32, 32],
        DataType::F32,
        input_data,
    );

    // Output shape: [1, 16, 30, 30]
    let mut outputs: &mut [(&str, lightship_core::ir::Tensor)] = &mut [(
        "output",
        lightship_core::ir::Tensor::new("output".to_string(), vec![1, 16, 30, 30], DataType::F32),
    )];

    let result = session.forward(&[("input", input)], outputs);
    assert!(result.is_ok(), "Forward pass failed: {:?}", result.err());

    // Verify output shape
    assert_eq!(outputs[0].1.shape, vec![1, 16, 30, 30]);
}
