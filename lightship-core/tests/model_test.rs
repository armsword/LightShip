//! Unit tests for Model loading

use lightship_core::common::ModelFormat;
use lightship_core::ir::{Graph, Node, NodeIO, OperatorType};
use lightship_core::model::{
    MagicNumber, ModelFile, ModelLoaderRegistry, ModelMetadata, ValidationResult,
};
use std::collections::HashMap;

#[test]
fn test_model_metadata_creation() {
    let metadata = ModelMetadata::new("test_model");
    assert_eq!(metadata.name, "test_model");
    assert_eq!(metadata.version, "1.0.0");
    assert!(metadata.created_at.is_some());
}

#[test]
fn test_model_metadata_builder() {
    let metadata = ModelMetadata::new("test")
        .with_version("2.0.0")
        .with_author("Test Author")
        .with_description("A test model")
        .with_custom("key1", "value1");

    assert_eq!(metadata.name, "test");
    assert_eq!(metadata.version, "2.0.0");
    assert_eq!(metadata.author, Some("Test Author".to_string()));
    assert_eq!(metadata.description, Some("A test model".to_string()));
    assert_eq!(metadata.custom.get("key1"), Some(&"value1".to_string()));
}

#[test]
fn test_model_file_creation() {
    let graph = Graph::new("test_graph".into());
    let model = ModelFile::new(ModelFormat::ONNX, graph);

    assert_eq!(model.format, ModelFormat::ONNX);
    assert_eq!(model.ir_version, 1);
    assert_eq!(model.num_operators(), 0);
}

#[test]
fn test_model_file_with_nodes() {
    let mut graph = Graph::new("test_graph".into());

    let mut node = Node::new(0, "conv1".into(), OperatorType::Conv2d);
    node.inputs.push(NodeIO {
        tensor_name: "input".into(),
        data_type: lightship_core::common::DataType::F32,
    });
    node.outputs.push(NodeIO {
        tensor_name: "output".into(),
        data_type: lightship_core::common::DataType::F32,
    });
    graph.add_node(node);

    let model = ModelFile::new(ModelFormat::Native, graph);
    assert_eq!(model.num_operators(), 1);
}

#[test]
fn test_model_format_extensions() {
    assert_eq!(ModelFormat::ONNX.extensions(), vec!["onnx"]);
    assert_eq!(ModelFormat::TensorFlow.extensions(), vec!["pb", "savedmodel"]);
    assert_eq!(ModelFormat::TFLite.extensions(), vec!["tflite"]);
}

#[test]
fn test_model_format_is_protobuf() {
    assert!(ModelFormat::ONNX.is_protobuf());
    assert!(ModelFormat::TensorFlow.is_protobuf());
    assert!(!ModelFormat::Native.is_protobuf());
}

#[test]
fn test_magic_number_detection() {
    // ONNX protobuf magic number
    let onnx_data = vec![0x08, 0x00, 0x00, 0x00];
    assert_eq!(MagicNumber::detect(&onnx_data), Some(ModelFormat::ONNX));

    // LightShip native format
    let native_data = b"LIGHTSHIP".to_vec();
    assert_eq!(MagicNumber::detect(&native_data), Some(ModelFormat::Native));

    // Unknown format
    let unknown_data = vec![0x00, 0x01, 0x02, 0x03];
    assert_eq!(MagicNumber::detect(&unknown_data), None);
}

#[test]
fn test_model_loader_registry_creation() {
    let registry = ModelLoaderRegistry::new();
    // Registry should be created successfully
    assert!(true);
}

#[test]
fn test_validation_result() {
    let result = ValidationResult {
        valid: true,
        errors: Vec::new(),
        warnings: Vec::new(),
    };
    assert!(result.valid);
    assert!(result.errors.is_empty());
    assert!(result.warnings.is_empty());
}
