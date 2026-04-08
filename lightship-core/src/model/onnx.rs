//! ONNX model loader
//!
//! Parses ONNX format models into LightShip IR.

use crate::common::types::DataType;
use crate::ir::graph::Graph;
use crate::ir::operator::OperatorType;
use crate::model::error::ModelLoaderError;
use crate::model::loader::{ModelFile, ModelLoader, ValidationResult};
use crate::model::metadata::ModelMetadata;
use crate::common::ModelFormat;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::Path;
use std::result::Result as StdResult;

/// ONNX operator type mapping
mod operators {
    use super::OperatorType;

    pub fn from_onnx_op(op_type: &str) -> OperatorType {
        match op_type {
            // Convolution
            "Conv" => OperatorType::Conv2d,
            "ConvTranspose" => OperatorType::ConvTranspose2d,

            // Pooling
            "MaxPool" => OperatorType::MaxPool2d,
            "AveragePool" => OperatorType::AvgPool2d,
            "GlobalMaxPool" => OperatorType::GlobalMaxPool2d,
            "GlobalAveragePool" => OperatorType::GlobalAvgPool2d,

            // Activation
            "Relu" => OperatorType::ReLU,
            "Sigmoid" => OperatorType::Sigmoid,
            "Tanh" => OperatorType::Tanh,
            // Note: LeakyRelu, PRelu, Clip mapped to ReLU as approximation
            "LeakyRelu" | "PRelu" | "Clip" => OperatorType::ReLU,

            // Normalization
            "BatchNormalization" => OperatorType::BatchNorm,
            "LayerNormalization" => OperatorType::LayerNorm,
            "InstanceNormalization" => OperatorType::InstanceNorm,

            // Linear
            "Gemm" | "MatMul" => OperatorType::MatMul,
            "Add" | "Sum" => OperatorType::Add,
            "Mul" => OperatorType::Mul,
            "Sub" => OperatorType::Sub,
            "Div" => OperatorType::Div,

            // Transformer
            "Softmax" | "LogSoftmax" => OperatorType::Softmax,
            "Gelu" => OperatorType::GELU,
            "Attention" => OperatorType::SelfAttention,
            "MultiHeadAttention" => OperatorType::MultiHeadAttention,

            // Tensor operations
            "Reshape" => OperatorType::Reshape,
            "Transpose" => OperatorType::Transpose,
            "Flatten" => OperatorType::Flatten,
            "Squeeze" => OperatorType::Squeeze,
            "Unsqueeze" => OperatorType::Unsqueeze,
            "Expand" => OperatorType::Expand,
            "Gather" => OperatorType::Gather,
            "Concat" => OperatorType::Concat,
            "Split" => OperatorType::Split,
            "Slice" => OperatorType::Slice,
            "Pad" => OperatorType::Pad,
            "ReduceMean" | "ReduceMax" | "ReduceMin" | "ReduceSum" => OperatorType::Reduce,
            "Resize" => OperatorType::Resize,
            "Crop" => OperatorType::Crop,

            // Operators not directly supported - use Custom
            "Identity" | "Constant" | "Pow" | "Sqrt" | "Cast" | "Scatter" | "ArgMax" | "ArgMin"
                => OperatorType::Custom,

            _ => OperatorType::Custom,
        }
    }
}

/// ONNX data type mapping
fn from_onnx_dtype(dtype: i32) -> Option<DataType> {
    match dtype {
        1 => Some(DataType::F32),      // float
        2 => Some(DataType::F64),      // double
        3 => Some(DataType::F16),      // float16
        4 => None,                     // bfloat16 (not supported)
        5 => Some(DataType::I8),       // int8
        6 => Some(DataType::I16),      // int16
        7 => Some(DataType::I32),      // int32
        8 => Some(DataType::I64),      // int64
        9 => Some(DataType::U8),       // uint8
        10 => Some(DataType::U16),     // uint16
        11 => Some(DataType::U32),     // uint32
        12 => Some(DataType::U64),     // uint64
        13 => Some(DataType::Bool),    // bool
        14 => Some(DataType::QUInt8),  // quint8
        15 => Some(DataType::QInt8),   // qint8
        16 => Some(DataType::QInt32),  // qint32
        _ => None,
    }
}

/// ONNX tensor shape dimension
#[derive(Debug, Clone)]
pub enum Dimension {
    /// Known dimension value
    Known(i64),
    /// Unknown dimension
    Unknown,
}

impl Dimension {
    /// Convert to usize if known and positive
    pub fn to_usize(&self) -> Option<usize> {
        match self {
            Dimension::Known(v) if *v >= 0 => Some(*v as usize),
            _ => None,
        }
    }
}

/// Parse ONNX shape string (e.g., "3x224x224")
#[allow(dead_code)]
fn parse_shape_string(s: &str) -> Vec<Dimension> {
    s.split('x')
        .filter_map(|part| {
            part.parse::<i64>().ok().map(Dimension::Known)
        })
        .collect()
}

/// ONNX model loader
pub struct OnnxLoader {
    // Protobuf parsing would require protobuf support
    // For now, we provide the structure with placeholder parsing
}

impl OnnxLoader {
    /// Create a new ONNX loader
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for OnnxLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for OnnxLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxLoader").finish()
    }
}

impl ModelLoader for OnnxLoader {
    fn supported_formats(&self) -> Vec<ModelFormat> {
        vec![ModelFormat::ONNX]
    }

    fn load_from_file(&self, path: &Path) -> StdResult<ModelFile, ModelLoaderError> {
        tracing::debug!("Loading ONNX model from: {}", path.display());

        let bytes = std::fs::read(path)
            .map_err(|e| ModelLoaderError::IoError(e.to_string()))?;

        self.load_from_bytes(&bytes, ModelFormat::ONNX)
    }

    fn load_from_bytes(&self, bytes: &[u8], format: ModelFormat) -> StdResult<ModelFile, ModelLoaderError> {
        if format != ModelFormat::ONNX {
            return Err(ModelLoaderError::InvalidFormat(format!("Expected ONNX format, got {:?}", format)));
        }

        // Check for ZIP header (ONNX is a ZIP archive)
        if bytes.len() >= 2 && bytes[0] == 0x50 && bytes[1] == 0x4B {
            return self.parse_protobuf(bytes);
        }

        // Check for raw protobuf
        if bytes.len() >= 2 && bytes[0] == 0x08 {
            return self.parse_protobuf(bytes);
        }

        Err(ModelLoaderError::InvalidFormat("Unrecognized ONNX file format".into()))
    }

    fn validate(&self, model: &ModelFile) -> StdResult<ValidationResult, ModelLoaderError> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check that graph has inputs
        if model.graph.inputs.is_empty() {
            warnings.push(crate::model::loader::ValidationWarning {
                message: "Model has no inputs defined".to_string(),
                node_name: None,
            });
        }

        // Check that graph has outputs
        if model.graph.outputs.is_empty() {
            warnings.push(crate::model::loader::ValidationWarning {
                message: "Model has no outputs defined".to_string(),
                node_name: None,
            });
        }

        // Check for unsupported operators
        for node in &model.graph.nodes {
            if matches!(node.operator_type, OperatorType::Custom) {
                warnings.push(crate::model::loader::ValidationWarning {
                    message: format!("Custom or unsupported operator: {:?}", node.operator_type),
                    node_name: Some(node.name.clone()),
                });
            }
        }

        Ok(ValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
        })
    }

    fn name(&self) -> &'static str {
        "ONNX Loader"
    }
}

impl OnnxLoader {
    /// Parse ONNX protobuf format
    fn parse_protobuf(&self, bytes: &[u8]) -> StdResult<ModelFile, ModelLoaderError> {
        // TODO: Full protobuf parsing requires protobuf crate or manual parsing
        // For now, we create a placeholder model structure

        tracing::warn!("ONNX protobuf parsing is simplified - using placeholder model");

        let graph = Graph::new("onnx_model".to_string());

        Ok(ModelFile {
            format: ModelFormat::ONNX,
            metadata: ModelMetadata::default(),
            ir_version: 1,
            graph,
            extra_data: HashMap::new(),
        })
    }
}

/// Create a model loader registry with all supported loaders
#[allow(dead_code)]
pub fn create_default_registry() -> crate::model::loader::ModelLoaderRegistry {
    let mut registry = crate::model::loader::ModelLoaderRegistry::new();
    registry.register(OnnxLoader::new());
    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_operator_mapping() {
        // Test that common operators map correctly
        assert!(matches!(
            operators::from_onnx_op("Conv"),
            OperatorType::Conv2d
        ));
        assert!(matches!(
            operators::from_onnx_op("Relu"),
            OperatorType::ReLU
        ));
        assert!(matches!(
            operators::from_onnx_op("MatMul"),
            OperatorType::MatMul
        ));
        assert!(matches!(
            operators::from_onnx_op("Softmax"),
            OperatorType::Softmax
        ));
    }

    #[test]
    fn test_data_type_mapping() {
        assert_eq!(from_onnx_dtype(1), Some(DataType::F32));
        assert_eq!(from_onnx_dtype(7), Some(DataType::I32));
        assert_eq!(from_onnx_dtype(10), Some(DataType::U16));
    }

    #[test]
    fn test_dimension() {
        assert_eq!(Dimension::Known(10).to_usize(), Some(10));
        assert_eq!(Dimension::Unknown.to_usize(), None);
        assert_eq!(Dimension::Known(-1).to_usize(), None);
    }

    #[test]
    fn test_shape_parsing() {
        let shape = parse_shape_string("3x224x224");
        assert_eq!(shape.len(), 3);
        assert_eq!(shape[0].to_usize(), Some(3));
        assert_eq!(shape[1].to_usize(), Some(224));
    }

    #[test]
    fn test_onnx_loader_creation() {
        let loader = OnnxLoader::new();
        assert_eq!(loader.supported_formats(), vec![ModelFormat::ONNX]);
        assert_eq!(loader.name(), "ONNX Loader");
    }
}
