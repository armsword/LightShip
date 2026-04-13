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
use std::io::{Cursor, Read};
use std::path::Path;
use std::result::Result as StdResult;
use std::sync::Arc;

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
        1 => Some(DataType::F32),
        2 => Some(DataType::F64),
        3 => Some(DataType::F16),
        4 => None,
        5 => Some(DataType::I8),
        6 => Some(DataType::I16),
        7 => Some(DataType::I32),
        8 => Some(DataType::I64),
        9 => Some(DataType::U8),
        10 => Some(DataType::U16),
        11 => Some(DataType::U32),
        12 => Some(DataType::U64),
        13 => Some(DataType::Bool),
        14 => Some(DataType::QUInt8),
        15 => Some(DataType::QInt8),
        16 => Some(DataType::QInt32),
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
    // Protobuf parsing structures are defined inline
}

impl OnnxLoader {
    /// Create a new ONNX loader
    pub fn new() -> Self {
        Self {}
    }

    /// Extract graph.pb from ONNX ZIP and parse it
    fn parse_onnx_zip(&self, data: &[u8]) -> StdResult<ModelFile, ModelLoaderError> {
        let cursor = Cursor::new(data);
        let mut archive = zip::ZipArchive::new(cursor)
            .map_err(|e| ModelLoaderError::InvalidFormat(format!("Invalid ZIP archive: {}", e)))?;

        // Find and read graph.pb
        let mut graph_pb = None;
        for i in 0..archive.len() {
            let file = archive.by_index(i)
                .map_err(|e| ModelLoaderError::InvalidFormat(format!("Failed to read ZIP entry: {}", e)))?;
            if file.name() == "graph.pb" {
                let mut buf = Vec::new();
                let mut reader = file;
                reader.read_to_end(&mut buf)
                    .map_err(|e| ModelLoaderError::IoError(format!("Failed to read graph.pb: {}", e)))?;
                graph_pb = Some(buf);
                break;
            }
        }

        let graph_pb = graph_pb
            .ok_or_else(|| ModelLoaderError::InvalidFormat("No graph.pb found in ONNX file".into()))?;

        self.parse_graph_protobuf(&graph_pb)
    }

    /// Parse graph.pb protobuf
    fn parse_graph_protobuf(&self, data: &[u8]) -> StdResult<ModelFile, ModelLoaderError> {
        let mut graph = Graph::new("onnx_model".to_string());
        let mut offset = 0;

        // Simple protobuf parsing - process based on wire_type, skip unknown fields
        while offset < data.len() {
            let (field_number, wire_type, new_offset) = match self.read_tag(data, offset) {
                Some(v) => v,
                None => break,
            };
            offset = new_offset;

            match (field_number, wire_type) {
                // field 1: ir_version (varint)
                (1, 0) => {
                    offset = self.skip_field(data, offset, wire_type)?;
                }
                // field 2: producer_name (string)
                (2, 2) => {
                    if let Ok((str_val, new_offset)) = self.read_string(data, offset) {
                        graph.name = str_val;
                        offset = new_offset;
                    } else {
                        offset = self.skip_field(data, offset, wire_type)?;
                    }
                }
                // field 3: producer_version (string)
                (3, 2) => {
                    offset = self.skip_field(data, offset, wire_type)?;
                }
                // field 4: domain (string)
                (4, 2) => {
                    offset = self.skip_field(data, offset, wire_type)?;
                }
                // field 5: model_version (varint)
                (5, 0) => {
                    offset = self.skip_field(data, offset, wire_type)?;
                }
                // field 6: doc_string (string)
                (6, 2) => {
                    offset = self.skip_field(data, offset, wire_type)?;
                }
                // field 7: graph (length-delimited message)
                (7, 2) => {
                    let (graph_len, after_len_offset) = self.read_varint(data, offset)?;
                    let graph_end = after_len_offset + graph_len as usize;

                    // Parse nodes from graph content
                    let (nodes, graph_inputs, graph_outputs, graph_variables) = self.parse_nodes_from_graph(data, after_len_offset, graph_end)?;
                    graph.nodes = nodes;
                    graph.inputs = graph_inputs;
                    graph.outputs = graph_outputs;
                    graph.variables = graph_variables;

                    // Move offset past the entire graph field
                    offset = graph_end;
                }
                // field 8: opset_import (message, skip)
                (8, 2) => {
                    offset = self.skip_field(data, offset, wire_type)?;
                }
                // Unknown field - skip
                _ => {
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
    /// Returns (nodes, inputs, outputs, variables)
    fn parse_nodes_from_graph(&self, data: &[u8], mut offset: usize, end: usize) -> StdResult<(Vec<Node>, Vec<GraphIO>, Vec<GraphIO>, std::collections::HashMap<String, Arc<crate::ir::Tensor>>), ModelLoaderError> {
        let mut nodes = Vec::new();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut variables: std::collections::HashMap<String, Arc<crate::ir::Tensor>> = std::collections::HashMap::new();
        let mut node_id = 0;

        // Look for node entries (field 1 in GraphProto per ONNX spec)
        while offset < end {
            let (field_number, wire_type, new_offset) = match self.read_tag(data, offset) {
                Some(v) => v,
                None => break,
            };

            match (field_number, wire_type) {
                (1, 2) => {
                    // NodeProto - length-delimited
                    let (content_len, len_bytes) = {
                        let (len, pos) = self.read_varint(data, new_offset)?;
                        (len as usize, pos - new_offset)
                    };
                    let content_start = new_offset + len_bytes;
                    let content_end = content_start + content_len;
                    let (node, _) = self.parse_node(data, content_start, content_end, node_id)?;
                    nodes.push(node);
                    offset = content_end;
                    node_id += 1;
                }
                (5, 2) => {
                    // initializer (TensorProto) - length-delimited
                    let (content_len, len_bytes) = {
                        let (len, pos) = self.read_varint(data, new_offset)?;
                        (len as usize, pos - new_offset)
                    };
                    let content_start = new_offset + len_bytes;
                    let content_end = content_start + content_len;
                    if let Some(tensor) = self.parse_tensor_proto(data, content_start, content_end)? {
                        let name = tensor.name.clone();
                        variables.insert(name, Arc::new(tensor));
                    }
                    offset = content_end;
                }
                (11, 2) => {
                    // ValueInfoProto for input
                    let (content_len, len_bytes) = {
                        let (len, pos) = self.read_varint(data, new_offset)?;
                        (len as usize, pos - new_offset)
                    };
                    let content_start = new_offset + len_bytes;
                    let content_end = content_start + content_len;
                    if let Some(graph_io) = self.parse_value_info_proto(data, content_start, content_end, true)? {
                        inputs.push(graph_io);
                    }
                    offset = content_end;
                }
                (12, 2) => {
                    // ValueInfoProto for output
                    let (content_len, len_bytes) = {
                        let (len, pos) = self.read_varint(data, new_offset)?;
                        (len as usize, pos - new_offset)
                    };
                    let content_start = new_offset + len_bytes;
                    let content_end = content_start + content_len;
                    if let Some(graph_io) = self.parse_value_info_proto(data, content_start, content_end, false)? {
                        outputs.push(graph_io);
                    }
                    offset = content_end;
                }
                _ => {
                    // Skip any other fields
                    offset = self.skip_field(data, new_offset, wire_type)?;
                }
            }
        }

        Ok((nodes, inputs, outputs, variables))
    }

    /// Parse TensorProto (initializer)
    fn parse_tensor_proto(&self, data: &[u8], offset: usize, end: usize) -> StdResult<Option<crate::ir::Tensor>, ModelLoaderError> {
        let mut name = String::new();
        let mut dims: Vec<usize> = Vec::new();
        let mut data_type: i32 = 1;  // Default to F32
        let mut float_data: Option<Vec<f32>> = None;
        let mut raw_data: Option<Vec<u8>> = None;

        let mut pos = offset;
        while pos < end && pos < data.len() {
            let (field_number, wire_type, new_offset) = match self.read_tag(data, pos) {
                Some(v) => v,
                None => break,
            };
            pos = new_offset;

            match (field_number, wire_type) {
                (1, 0) => {
                    // dims (repeated int64) - each dim is a varint
                    if let Some((val, new_pos)) = self.read_varint_raw(data, pos) {
                        dims.push(val as usize);
                        pos = new_pos;
                    }
                }
                (1, 2) => {
                    // dims as packed bytes (protobuf 2 packed encoding)
                    if let Ok((dim_bytes, new_pos)) = self.read_bytes(data, pos) {
                        // Parse dimensions from raw bytes (packed int64)
                        let mut byte_pos = 0;
                        while byte_pos + 8 <= dim_bytes.len() {
                            let dim = i64::from_le_bytes([
                                dim_bytes[byte_pos], dim_bytes[byte_pos+1],
                                dim_bytes[byte_pos+2], dim_bytes[byte_pos+3],
                                dim_bytes[byte_pos+4], dim_bytes[byte_pos+5],
                                dim_bytes[byte_pos+6], dim_bytes[byte_pos+7],
                            ]);
                            dims.push(dim as usize);
                            byte_pos += 8;
                        }
                        pos = new_pos;
                    }
                }
                (2, 0) => {
                    // data_type (int)
                    if let Ok((val, new_pos)) = self.read_varint(data, pos) {
                        data_type = val as i32;
                        pos = new_pos;
                    }
                }
                (4, 2) => {
                    // float_data (repeated float) - length-delimited, packed bytes
                    if let Ok((float_bytes, new_pos)) = self.read_bytes(data, pos) {
                        let count = float_bytes.len() / 4;
                        let mut floats = Vec::with_capacity(count);
                        for i in 0..count {
                            let f = f32::from_le_bytes([
                                float_bytes[i*4], float_bytes[i*4+1],
                                float_bytes[i*4+2], float_bytes[i*4+3],
                            ]);
                            floats.push(f);
                        }
                        float_data = Some(floats);
                        pos = new_pos;
                    }
                }
                (8, 2) => {
                    // name (string)
                    if let Ok((str_val, new_pos)) = self.read_string(data, pos) {
                        name = str_val;
                        pos = new_pos;
                    }
                }
                (9, 2) => {
                    // raw_data (bytes)
                    if let Ok((bytes, new_pos)) = self.read_bytes(data, pos) {
                        raw_data = Some(bytes);
                        pos = new_pos;
                    }
                }
                _ => {
                    pos = self.skip_field(data, pos, wire_type)?;
                }
            }
        }

        if name.is_empty() {
            return Ok(None);
        }

        // Convert data to tensor
        let dtype = match data_type {
            1 => crate::common::DataType::F32,  // FLOAT
            7 => crate::common::DataType::I32,  // INT32
            _ => crate::common::DataType::F32,
        };

        // Use float_data if available, otherwise use raw_data
        let tensor_data = if let Some(ref floats) = float_data {
            let byte_size = floats.len() * 4;
            let mut bytes = Vec::with_capacity(byte_size);
            for f in floats {
                bytes.extend_from_slice(&f.to_le_bytes());
            }
            bytes
        } else if let Some(bytes) = raw_data {
            bytes
        } else {
            return Ok(None);
        };

        let tensor = crate::ir::Tensor {
            name: name.clone(),
            shape: dims,
            data_type: dtype,
            layout: crate::common::StorageLayout::Default,
            data: crate::ir::TensorData::Owned(tensor_data),
            quantization: None,
            lifetime: crate::common::TensorLifetime::Static,
        };

        Ok(Some(tensor))
    }

    /// Parse ValueInfoProto (input/output definition)
    /// Returns GraphIO if parsing succeeds
    fn parse_value_info_proto(&self, data: &[u8], offset: usize, end: usize, is_input: bool) -> StdResult<Option<GraphIO>, ModelLoaderError> {
        let mut name = String::new();
        let mut dtype = crate::common::DataType::F32;

        let mut pos = offset;
        while pos < end && pos < data.len() {
            let (field_number, wire_type, new_offset) = match self.read_tag(data, pos) {
                Some(v) => v,
                None => break,
            };
            pos = new_offset;

            match (field_number, wire_type) {
                (1, 2) => {
                    // name (string)
                    if let Ok((str_val, new_pos)) = self.read_string(data, pos) {
                        name = str_val;
                        pos = new_pos;
                    }
                }
                (2, 2) => {
                    // type (TypeProto) - length-delimited
                    let (content_len, len_bytes) = {
                        let (len, p) = self.read_varint(data, pos)?;
                        (len as usize, p - pos)
                    };
                    let type_end = pos + len_bytes + content_len;
                    // Parse embedded TypeProto to get tensor_type.elem_type
                    let mut type_pos = pos + len_bytes;
                    while type_pos < type_end && type_pos < data.len() {
                        let (type_field, type_wire, type_new) = match self.read_tag(data, type_pos) {
                            Some(v) => v,
                            None => break,
                        };
                        type_pos = type_new;

                        if type_field == 1 && type_wire == 2 {
                            // tensor_type
                            let (tensor_len, tensor_len_bytes) = {
                                let (len, p) = self.read_varint(data, type_pos)?;
                                (len as usize, p - type_pos)
                            };
                            let tensor_end = type_pos + tensor_len_bytes + tensor_len;
                            let mut elem_type_pos = type_pos + tensor_len_bytes;
                            while elem_type_pos < tensor_end && elem_type_pos < data.len() {
                                let (et_field, et_wire, et_new) = match self.read_tag(data, elem_type_pos) {
                                    Some(v) => v,
                                    None => break,
                                };
                                elem_type_pos = et_new;
                                if et_field == 1 && et_wire == 0 {
                                    // elem_type
                                    if let Some((val, _)) = self.read_varint_raw(data, elem_type_pos) {
                                        dtype = match val as i32 {
                                            1 => crate::common::DataType::F32,
                                            7 => crate::common::DataType::I32,
                                            _ => crate::common::DataType::F32,
                                        };
                                    }
                                } else if et_field == 2 {
                                    // shape
                                    // Could parse dimensions here if needed
                                }
                                if et_wire == 2 || et_new >= tensor_end {
                                    break;
                                }
                                elem_type_pos = self.skip_field(data, elem_type_pos, et_wire).unwrap_or(elem_type_pos);
                            }
                        }
                        if type_wire == 2 || type_new >= type_end {
                            break;
                        }
                        type_pos = self.skip_field(data, type_pos, type_wire).unwrap_or(type_pos);
                    }
                    pos = type_end;
                }
                _ => {
                    pos = self.skip_field(data, pos, wire_type)?;
                }
            }
        }

        if name.is_empty() {
            return Ok(None);
        }

        Ok(Some(GraphIO {
            name: name.clone(),
            io: crate::ir::NodeIO {
                tensor_name: name,
                data_type: dtype,
            },
            is_model_input: is_input,
            is_model_output: !is_input,
        }))
    }

    /// Parse a single node (NodeProto)
    fn parse_node(&self, data: &[u8], offset: usize, end: usize, node_id: u32) -> StdResult<(Node, usize), ModelLoaderError> {
        let mut name = format!("node_{}", node_id);
        let mut op_type = String::new();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut current_offset = offset;

        while current_offset < end {
            let (field_number, wire_type, new_offset) = match self.read_tag(data, current_offset) {
                Some(v) => v,
                None => break,
            };

            match field_number {
                1 => {
                    // input (repeated string) - field 1 in actual ONNX bytes is input tensor name
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
                2 => {
                    // output (repeated string) - field 2 in actual ONNX bytes is output tensor name
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
                3 => {
                    // name (string) - according to ONNX spec
                    if let Ok((str_val, read_offset)) = self.read_string(data, new_offset) {
                        name = str_val;
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
                5 | 6 | 7 => {
                    // attribute, doc_string, domain - skip
                    current_offset = self.skip_field(data, new_offset, wire_type)?;
                }
                _ => {
                    // Unknown field - skip
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
            fusion: None,
        }, current_offset))
    }

    /// Parse ValueInfoProto (input/output definition)
    #[allow(dead_code)]
    fn parse_value_info(&self, data: &[u8], offset: usize) -> StdResult<Option<GraphIO>, ModelLoaderError> {
        let mut name = String::new();
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
        let (varint, new_offset) = self.read_varint_raw(data, offset)?;
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

    /// Read raw bytes (for binary data like packed int64s or raw tensor data)
    fn read_bytes(&self, data: &[u8], offset: usize) -> Result<(Vec<u8>, usize), ModelLoaderError> {
        let (len, mut pos) = self.read_varint(data, offset)?;
        let len = len as usize;
        if pos + len > data.len() {
            return Err(ModelLoaderError::ParseError {
                location: format!("offset {}", offset),
                message: format!("Bytes extend beyond buffer: {} + {}", pos, len),
            });
        }
        let bytes = data[pos..pos + len].to_vec();
        pos += len;
        Ok((bytes, pos))
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
        let errors = Vec::new();
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
}
