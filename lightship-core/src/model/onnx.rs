//! ONNX model loader
//!
//! Parses ONNX format models into LightShip IR.

use crate::common::types::DataType;
use crate::ir::graph::{Graph, GraphIO, Node, NodeIO};
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

/// ZIP file entry
#[derive(Debug)]
struct ZipEntry {
    name: String,
    data: Vec<u8>,
}

/// Parse a ZIP file and extract entries
fn parse_zip(data: &[u8]) -> Result<Vec<ZipEntry>, ModelLoaderError> {
    if data.len() < 4 || data[0] != 0x50 || data[1] != 0x4B {
        return Err(ModelLoaderError::InvalidFormat("Invalid ZIP file".into()));
    }

    let mut entries = Vec::new();
    let mut offset = 0;

    while offset + 30 <= data.len() {
        // Read local file header
        if data[offset] != 0x50 || data[offset + 1] != 0x4B {
            break;
        }

        let compression = u16::from_le_bytes([data[offset + 8], data[offset + 9]]);
        let compressed_size = u32::from_le_bytes([
            data[offset + 18], data[offset + 19], data[offset + 20], data[offset + 21]
        ]) as usize;
        let uncompressed_size = u32::from_le_bytes([
            data[offset + 22], data[offset + 23], data[offset + 24], data[offset + 25]
        ]) as usize;
        let name_len = u16::from_le_bytes([data[offset + 26], data[offset + 27]]) as usize;
        let extra_len = u16::from_le_bytes([data[offset + 28], data[offset + 29]]) as usize;

        if offset + 30 + name_len > data.len() {
            break;
        }

        let name = String::from_utf8_lossy(&data[offset + 30..offset + 30 + name_len]).to_string();
        let data_start = offset + 30 + name_len + extra_len;

        // Only handle stored (no compression) or deflate
        let entry_data = if compression == 0 {
            // Stored - no compression
            if data_start + compressed_size > data.len() {
                break;
            }
            data[data_start..data_start + compressed_size].to_vec()
        } else if compression == 8 {
            // Deflate - for now, just skip
            tracing::warn!("Deflate compression not fully supported, skipping entry: {}", name);
            offset = data_start + compressed_size;
            continue;
        } else {
            break;
        };

        entries.push(ZipEntry {
            name,
            data: entry_data,
        });

        offset = data_start + compressed_size;
    }

    Ok(entries)
}

/// ONNX model loader
pub struct OnnxLoader {
    // Protobuf parsing structures are defined inline
}

impl OnnxLoader {
    /// Create a new ONNX loader
    pub fn new() -> Self {
        Self {}
    }

    /// Extract graph.pb from ONNX ZIP and parse it
    fn parse_onnx_zip(&self, data: &[u8]) -> StdResult<ModelFile, ModelLoaderError> {
        let entries = parse_zip(data)?;

        // Find graph.pb
        let graph_pb = entries
            .iter()
            .find(|e| e.name == "graph.pb")
            .ok_or_else(|| ModelLoaderError::InvalidFormat("No graph.pb found in ONNX file".into()))?;

        self.parse_graph_protobuf(&graph_pb.data)
    }

    /// Parse graph.pb protobuf
    fn parse_graph_protobuf(&self, data: &[u8]) -> StdResult<ModelFile, ModelLoaderError> {
        let mut graph = Graph::new("onnx_model".to_string());
        let mut offset = 0;

        // Simple protobuf parsing
        while offset < data.len() {
            let (field_number, wire_type, new_offset) = match self.read_tag(data, offset) {
                Some(v) => v,
                None => break,
            };
            offset = new_offset;

            match field_number {
                1 => {
                    // producer_name
                    if let Ok((str_val, new_offset)) = self.read_string(data, offset) {
                        graph.name = str_val;
                        offset = new_offset;
                    }
                }
                2 => {
                    // producer_version
                    if let Ok((_, new_offset)) = self.read_string(data, offset) {
                        offset = new_offset;
                    }
                }
                3 => {
                    // domain
                    if let Ok((_, new_offset)) = self.read_string(data, offset) {
                        offset = new_offset;
                    }
                }
                4 => {
                    // model_version
                    if let Ok((varint, new_offset)) = self.read_varint(data, offset) {
                        tracing::debug!("Model version: {}", varint);
                        offset = new_offset;
                    }
                }
                5 => {
                    // doc_string
                    if let Ok((_, new_offset)) = self.read_string(data, offset) {
                        offset = new_offset;
                    }
                }
                6 => {
                    // opset_import - skip for now
                    offset = self.skip_field(data, offset, wire_type)?;
                }
                7 => {
                    // graph
                    offset = self.skip_field(data, offset, wire_type)?;

                    // Try to parse nodes manually
                    let (nodes, graph_outputs, new_offset) = self.parse_nodes_from_graph(data, offset)?;
                    graph.nodes = nodes;
                    graph.outputs = graph_outputs;
                    offset = new_offset;
                }
                _ => {
                    // Unknown field, skip
                    offset = self.skip_field(data, offset, wire_type)?;
                }
            }
        }

        Ok(ModelFile {
            format: ModelFormat::ONNX,
            metadata: ModelMetadata::default(),
            ir_version: 1,
            graph,
            extra_data: HashMap::new(),
        })
    }

    /// Parse nodes from graph body
    fn parse_nodes_from_graph(&self, data: &[u8], mut offset: usize) -> StdResult<(Vec<Node>, Vec<GraphIO>, usize), ModelLoaderError> {
        let mut nodes = Vec::new();
        let mut outputs = Vec::new();
        let mut node_id = 0;

        // Look for node entries (field 9 in GraphProto)
        while offset < data.len() {
            let (field_number, wire_type, new_offset) = match self.read_tag(data, offset) {
                Some(v) => v,
                None => break,
            };

            if field_number == 9 {
                // This is a node
                let node_offset = new_offset;
                let (node, new_offset) = self.parse_node(data, node_offset, node_id)?;
                nodes.push(node);
                offset = new_offset;
                node_id += 1;
            } else if field_number == 11 {
                // input (ValueInfoProto) - field 11
                offset = self.skip_field(data, new_offset, wire_type)?;
            } else if field_number == 12 {
                // output (ValueInfoProto) - field 12
                if let Some(output_io) = self.parse_value_info(data, new_offset)? {
                    outputs.push(output_io);
                }
                // Try to continue parsing after the field
                offset = self.skip_field(data, new_offset, wire_type)?;
            } else if field_number == 8 {
                // name - string
                offset = self.skip_field(data, new_offset, wire_type)?;
            } else if field_number == 13 {
                // initializer - skip
                offset = self.skip_field(data, new_offset, wire_type)?;
            } else if field_number == 14 {
                // sparse_initializer - skip
                offset = self.skip_field(data, new_offset, wire_type)?;
            } else {
                offset = self.skip_field(data, new_offset, wire_type)?;
            }
        }

        Ok((nodes, outputs, offset))
    }

    /// Parse a single node (NodeProto)
    fn parse_node(&self, data: &[u8], offset: usize, node_id: u32) -> StdResult<(Node, usize), ModelLoaderError> {
        let mut name = format!("node_{}", node_id);
        let mut op_type = String::new();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut current_offset = offset;

        while current_offset < data.len() {
            let (field_number, wire_type, new_offset) = match self.read_tag(data, current_offset) {
                Some(v) => v,
                None => break,
            };

            match field_number {
                1 => {
                    // output (repeated string)
                    if let Ok((str_val, read_offset)) = self.read_string(data, new_offset) {
                        outputs.push(NodeIO {
                            tensor_name: str_val,
                            data_type: DataType::F32, // Default, will be refined later
                        });
                        current_offset = read_offset;
                    } else {
                        current_offset = self.skip_field(data, new_offset, wire_type)?;
                    }
                }
                2 => {
                    // name
                    if let Ok((str_val, read_offset)) = self.read_string(data, new_offset) {
                        name = str_val;
                        current_offset = read_offset;
                    } else {
                        current_offset = self.skip_field(data, new_offset, wire_type)?;
                    }
                }
                3 => {
                    // input (repeated string)
                    if let Ok((str_val, read_offset)) = self.read_string(data, new_offset) {
                        inputs.push(NodeIO {
                            tensor_name: str_val,
                            data_type: DataType::F32,
                        });
                        current_offset = read_offset;
                    } else {
                        current_offset = self.skip_field(data, new_offset, wire_type)?;
                    }
                }
                4 => {
                    // op_type
                    if let Ok((str_val, read_offset)) = self.read_string(data, new_offset) {
                        op_type = str_val;
                        current_offset = read_offset;
                    } else {
                        current_offset = self.skip_field(data, new_offset, wire_type)?;
                    }
                }
                5 => {
                    // attribute - skip for now
                    current_offset = self.skip_field(data, new_offset, wire_type)?;
                }
                6 => {
                    // doc_string - skip
                    current_offset = self.skip_field(data, new_offset, wire_type)?;
                }
                7 => {
                    // domain - skip
                    current_offset = self.skip_field(data, new_offset, wire_type)?;
                }
                _ => {
                    current_offset = self.skip_field(data, new_offset, wire_type)?;
                }
            }
        }

        let operator_type = operators::from_onnx_op(&op_type);

        Ok((Node {
            id: node_id,
            name,
            operator_type,
            inputs,
            outputs,
        }, current_offset))
    }

    /// Parse ValueInfoProto (input/output definition)
    #[allow(dead_code)]
    fn parse_value_info(&self, data: &[u8], offset: usize) -> StdResult<Option<GraphIO>, ModelLoaderError> {
        let mut name = String::new();
        let mut shape = Vec::new();
        let mut dtype = DataType::F32;
        let mut current_offset = offset;

        while current_offset < data.len() {
            let (field_number, wire_type, new_offset) = match self.read_tag(data, current_offset) {
                Some(v) => v,
                None => break,
            };

            match field_number {
                1 => {
                    // name
                    if let Ok((str_val, read_offset)) = self.read_string(data, new_offset) {
                        name = str_val;
                        current_offset = read_offset;
                    } else {
                        current_offset = self.skip_field(data, new_offset, wire_type)?;
                    }
                }
                2 => {
                    // type
                    current_offset = new_offset;
                    // Parse embedded TypeProto
                    while current_offset < data.len() {
                        let (inner_field, inner_wire, inner_new) = match self.read_tag(data, current_offset) {
                            Some(v) => v,
                            None => break,
                        };
                        if inner_field == 1 {
                            // tensor_type
                            current_offset = inner_new;
                            // Parse TensorProto
                            while current_offset < data.len() {
                                let (t_field, t_wire, t_new) = match self.read_tag(data, current_offset) {
                                    Some(v) => v,
                                    None => break,
                                };
                                if t_field == 1 {
                                    // elem_type
                                    if let Ok((varint, read_offset)) = self.read_varint(data, t_new) {
                                        dtype = from_onnx_dtype(varint as i32).unwrap_or(DataType::F32);
                                        current_offset = read_offset;
                                    } else {
                                        current_offset = self.skip_field(data, t_new, t_wire)?;
                                    }
                                } else if t_field == 2 {
                                    // shape - repeated dimension
                                    current_offset = t_new;
                                    // Try to parse dimensions
                                    while current_offset < data.len() {
                                        let (d_field, d_wire, d_new) = match self.read_tag(data, current_offset) {
                                            Some(v) => v,
                                            None => break,
                                        };
                                        if d_field == 1 {
                                            // dim_value
                                            if let Ok((varint, read_offset)) = self.read_varint(data, d_new) {
                                                shape.push(varint as usize);
                                                current_offset = read_offset;
                                            } else {
                                                current_offset = self.skip_field(data, d_new, d_wire)?;
                                            }
                                        } else if d_field == 2 {
                                            // dim_param
                                            if let Ok((str_val, read_offset)) = self.read_string(data, d_new) {
                                                tracing::debug!("Dynamic dimension: {}", str_val);
                                                current_offset = read_offset;
                                            } else {
                                                current_offset = self.skip_field(data, d_new, d_wire)?;
                                            }
                                        } else {
                                            current_offset = self.skip_field(data, d_new, d_wire)?;
                                        }
                                        if d_wire == 4 {
                                            // End of group
                                            break;
                                        }
                                    }
                                    break;
                                } else {
                                    current_offset = self.skip_field(data, t_new, t_wire)?;
                                }
                                if t_wire == 4 {
                                    break;
                                }
                            }
                        } else {
                            current_offset = self.skip_field(data, inner_new, inner_wire)?;
                        }
                        if inner_wire == 4 {
                            break;
                        }
                    }
                }
                _ => {
                    current_offset = self.skip_field(data, new_offset, wire_type)?;
                }
            }
        }

        if name.is_empty() {
            return Ok(None);
        }

        Ok(Some(GraphIO {
            name: name.clone(),
            io: NodeIO {
                tensor_name: name,
                data_type: dtype,
            },
            is_model_input: false,
            is_model_output: false,
        }))
    }

    /// Read a protobuf tag
    fn read_tag(&self, data: &[u8], offset: usize) -> Option<(u32, u8, usize)> {
        if offset >= data.len() {
            return None;
        }
        let (varint, mut new_offset) = self.read_varint_raw(data, offset)?;
        let field_number = (varint >> 3) as u32;
        let wire_type = (varint & 0x7) as u8;
        Some((field_number, wire_type, new_offset))
    }

    /// Read a varint (raw)
    fn read_varint_raw(&self, data: &[u8], offset: usize) -> Option<(u64, usize)> {
        let mut result = 0u64;
        let mut shift = 0;
        let mut pos = offset;

        loop {
            if pos >= data.len() {
                return None;
            }
            let byte = data[pos];
            result |= ((byte & 0x7F) as u64) << shift;
            pos += 1;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 64 {
                return None;
            }
        }

        Some((result, pos))
    }

    /// Read a varint
    fn read_varint(&self, data: &[u8], offset: usize) -> Result<(u64, usize), ModelLoaderError> {
        self.read_varint_raw(data, offset)
            .ok_or_else(|| ModelLoaderError::ParseError {
                location: format!("offset {}", offset),
                message: "Failed to read varint".into(),
            })
    }

    /// Read a length-delimited string
    fn read_string(&self, data: &[u8], offset: usize) -> Result<(String, usize), ModelLoaderError> {
        let (len, mut pos) = self.read_varint(data, offset)?;
        let len = len as usize;
        if pos + len > data.len() {
            return Err(ModelLoaderError::ParseError {
                location: format!("offset {}", offset),
                message: format!("String extends beyond buffer: {} + {}", pos, len),
            });
        }
        let s = String::from_utf8_lossy(&data[pos..pos + len]).to_string();
        pos += len;
        Ok((s, pos))
    }

    /// Skip a field based on wire type
    fn skip_field(&self, data: &[u8], offset: usize, wire_type: u8) -> Result<usize, ModelLoaderError> {
        match wire_type {
            0 => {
                // Varint
                let (_, new_offset) = self.read_varint(data, offset)?;
                Ok(new_offset)
            }
            1 => {
                // 64-bit
                if offset + 8 <= data.len() {
                    Ok(offset + 8)
                } else {
                    Err(ModelLoaderError::ParseError {
                        location: format!("offset {}", offset),
                        message: "Unexpected end of buffer".into(),
                    })
                }
            }
            2 => {
                // Length-delimited
                let (len, mut pos) = self.read_varint(data, offset)?;
                pos += len as usize;
                Ok(pos)
            }
            5 => {
                // 32-bit
                if offset + 4 <= data.len() {
                    Ok(offset + 4)
                } else {
                    Err(ModelLoaderError::ParseError {
                        location: format!("offset {}", offset),
                        message: "Unexpected end of buffer".into(),
                    })
                }
            }
            _ => Err(ModelLoaderError::ParseError {
                location: format!("offset {}", offset),
                message: format!("Unknown wire type: {}", wire_type),
            }),
        }
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
            return self.parse_onnx_zip(bytes);
        }

        // Check for raw protobuf
        if bytes.len() >= 2 && bytes[0] == 0x08 {
            return self.parse_graph_protobuf(bytes);
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

    #[test]
    fn test_zip_parsing() {
        // Create a minimal ZIP file with one entry
        let data = vec![
            0x50, 0x4B, 0x03, 0x04, // Local file header signature
            0x14, 0x00, // version needed
            0x00, 0x00, // general purpose bit flag
            0x00, 0x00, // compression method (stored)
            0x00, 0x00, // last mod time
            0x00, 0x00, // last mod date
            0x00, 0x00, 0x00, 0x00, // CRC-32
            0x05, 0x00, 0x00, 0x00, // compressed size (5)
            0x05, 0x00, 0x00, 0x00, // uncompressed size (5)
            0x04, 0x00, // file name length (4)
            0x00, 0x00, // extra field length (0)
            b't', b'e', b's', b't', // file name "test"
            b'h', b'e', b'l', b'l', b'o', // file data "hello"
        ];

        let entries = parse_zip(&data).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "test");
        assert_eq!(entries[0].data, b"hello");
    }
}
