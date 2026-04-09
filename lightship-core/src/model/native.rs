//! LightShip Native Model Format
//!
//! Serialization and deserialization for LightShip native model format.

use crate::common::types::{DataType, StorageLayout, TensorLifetime};
use crate::ir::graph::{Graph, GraphIO, Node, NodeIO};
use crate::ir::operator::OperatorType;
use crate::ir::tensor::Tensor;
use crate::model::error::ModelLoaderError;
use crate::model::loader::ModelFile;
use crate::model::metadata::ModelMetadata;
use crate::common::ModelFormat;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::Path;
use std::result::Result as StdResult;

/// LightShip native model magic number
const LIGHTSHIP_MAGIC: &[u8] = b"LIGHTSHIP";
const LIGHTSHIP_VERSION: u32 = 1;

/// Native model serializer
pub struct NativeSerializer;

impl NativeSerializer {
    /// Serialize a ModelFile to bytes
    pub fn serialize(model: &ModelFile) -> Result<Vec<u8>, ModelLoaderError> {
        let mut bytes = Vec::new();

        // Write header
        bytes.extend_from_slice(LIGHTSHIP_MAGIC);
        bytes.extend_from_slice(&LIGHTSHIP_VERSION.to_le_bytes());

        // Write metadata
        Self::write_string(&mut bytes, &model.metadata.name);
        Self::write_string(&mut bytes, &model.metadata.version);

        // Write graph name
        Self::write_string(&mut bytes, &model.graph.name);

        // Write inputs
        Self::write_u32(&mut bytes, model.graph.inputs.len() as u32);
        for input in &model.graph.inputs {
            Self::write_graph_io(&mut bytes, input);
        }

        // Write outputs
        Self::write_u32(&mut bytes, model.graph.outputs.len() as u32);
        for output in &model.graph.outputs {
            Self::write_graph_io(&mut bytes, output);
        }

        // Write nodes
        Self::write_u32(&mut bytes, model.graph.nodes.len() as u32);
        for node in &model.graph.nodes {
            Self::write_node(&mut bytes, node);
        }

        // Write variables (static tensors)
        Self::write_u32(&mut bytes, model.graph.variables.len() as u32);
        for (name, tensor) in &model.graph.variables {
            Self::write_string(&mut bytes, name);
            Self::write_tensor(&mut bytes, tensor);
        }

        Ok(bytes)
    }

    /// Deserialize bytes to ModelFile
    pub fn deserialize(bytes: &[u8]) -> Result<ModelFile, ModelLoaderError> {
        let mut offset = 0;

        // Read and verify header
        if bytes.len() < LIGHTSHIP_MAGIC.len() + 4 {
            return Err(ModelLoaderError::InvalidFormat("File too short".into()));
        }

        let magic = &bytes[0..LIGHTSHIP_MAGIC.len()];
        if magic != LIGHTSHIP_MAGIC {
            return Err(ModelLoaderError::InvalidFormat("Invalid magic number".into()));
        }
        offset += LIGHTSHIP_MAGIC.len();

        let version = u32::from_le_bytes([
            bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]
        ]);
        offset += 4;

        if version > LIGHTSHIP_VERSION {
            return Err(ModelLoaderError::UnsupportedVersion(format!("Version {} not supported", version)));
        }

        // Read metadata
        let (name, new_offset) = Self::read_string(bytes, offset)?;
        offset = new_offset;
        let (version_str, new_offset) = Self::read_string(bytes, offset)?;
        offset = new_offset;

        let metadata = ModelMetadata {
            name,
            version: version_str,
            ..Default::default()
        };

        // Read graph name
        let (graph_name, new_offset) = Self::read_string(bytes, offset)?;
        offset = new_offset;
        let mut graph = Graph::new(graph_name);

        // Read inputs
        let (num_inputs, new_offset) = Self::read_u32(bytes, offset)?;
        offset = new_offset;
        for _ in 0..num_inputs {
            let (input, new_offset) = Self::read_graph_io(bytes, offset)?;
            graph.inputs.push(input);
            offset = new_offset;
        }

        // Read outputs
        let (num_outputs, new_offset) = Self::read_u32(bytes, offset)?;
        offset = new_offset;
        for _ in 0..num_outputs {
            let (output, new_offset) = Self::read_graph_io(bytes, offset)?;
            graph.outputs.push(output);
            offset = new_offset;
        }

        // Read nodes
        let (num_nodes, new_offset) = Self::read_u32(bytes, offset)?;
        offset = new_offset;
        for i in 0..num_nodes {
            let (node, new_offset) = Self::read_node(bytes, offset, i)?;
            graph.nodes.push(node);
            offset = new_offset;
        }

        // Read variables
        let mut variables = HashMap::new();
        let (num_vars, new_offset) = Self::read_u32(bytes, offset)?;
        offset = new_offset;
        for _ in 0..num_vars {
            let (name, new_offset) = Self::read_string(bytes, offset)?;
            offset = new_offset;
            let (tensor, new_offset) = Self::read_tensor(bytes, offset)?;
            variables.insert(name, std::sync::Arc::new(tensor));
            offset = new_offset;
        }
        graph.variables = variables;

        Ok(ModelFile {
            format: ModelFormat::Native,
            metadata,
            ir_version: version,
            graph,
            extra_data: HashMap::new(),
        })
    }

    /// Save model to file
    pub fn save_to_file(model: &ModelFile, path: &Path) -> Result<(), ModelLoaderError> {
        let bytes = Self::serialize(model)?;
        std::fs::write(path, bytes)
            .map_err(|e| ModelLoaderError::IoError(e.to_string()))?;
        Ok(())
    }

    /// Load model from file
    pub fn load_from_file(path: &Path) -> Result<ModelFile, ModelLoaderError> {
        let bytes = std::fs::read(path)
            .map_err(|e| ModelLoaderError::IoError(e.to_string()))?;
        Self::deserialize(&bytes)
    }

    // === Writing helpers ===

    fn write_string(bytes: &mut Vec<u8>, s: &str) {
        let s_bytes = s.as_bytes();
        Self::write_u32(bytes, s_bytes.len() as u32);
        bytes.extend_from_slice(s_bytes);
    }

    fn write_u32(bytes: &mut Vec<u8>, v: u32) {
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    fn write_u64(bytes: &mut Vec<u8>, v: u64) {
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    fn write_u8(bytes: &mut Vec<u8>, v: u8) {
        bytes.push(v);
    }

    fn write_f32(bytes: &mut Vec<u8>, v: f32) {
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    fn write_bytes(bytes: &mut Vec<u8>, data: &[u8]) {
        Self::write_u32(bytes, data.len() as u32);
        bytes.extend_from_slice(data);
    }

    fn write_graph_io(bytes: &mut Vec<u8>, io: &GraphIO) {
        Self::write_string(bytes, &io.name);
        Self::write_node_io(bytes, &io.io);
        Self::write_u8(bytes, io.is_model_input as u8);
        Self::write_u8(bytes, io.is_model_output as u8);
    }

    fn write_node_io(bytes: &mut Vec<u8>, io: &NodeIO) {
        Self::write_string(bytes, &io.tensor_name);
        Self::write_u8(bytes, io.data_type as u8);
    }

    fn write_node(bytes: &mut Vec<u8>, node: &Node) {
        Self::write_u32(bytes, node.id);
        Self::write_string(bytes, &node.name);
        Self::write_operator_type(bytes, &node.operator_type);
        Self::write_u32(bytes, node.inputs.len() as u32);
        for input in &node.inputs {
            Self::write_node_io(bytes, input);
        }
        Self::write_u32(bytes, node.outputs.len() as u32);
        for output in &node.outputs {
            Self::write_node_io(bytes, output);
        }
    }

    fn write_operator_type(bytes: &mut Vec<u8>, op: &OperatorType) {
        let val = match op {
            OperatorType::Conv2d => 1,
            OperatorType::ConvTranspose2d => 2,
            OperatorType::MaxPool2d => 3,
            OperatorType::AvgPool2d => 4,
            OperatorType::GlobalAvgPool2d => 5,
            OperatorType::GlobalMaxPool2d => 6,
            OperatorType::FullyConnected => 7,
            OperatorType::ReLU => 8,
            OperatorType::ReLU6 => 9,
            OperatorType::Sigmoid => 10,
            OperatorType::Tanh => 11,
            OperatorType::Softmax => 12,
            OperatorType::GELU => 13,
            OperatorType::SiLU => 14,
            OperatorType::BatchNorm => 15,
            OperatorType::LayerNorm => 16,
            OperatorType::InstanceNorm => 17,
            OperatorType::SelfAttention => 18,
            OperatorType::MultiHeadAttention => 19,
            OperatorType::Add => 20,
            OperatorType::Sub => 21,
            OperatorType::Mul => 22,
            OperatorType::Div => 23,
            OperatorType::MatMul => 24,
            OperatorType::Broadcast => 25,
            OperatorType::Reshape => 26,
            OperatorType::Transpose => 27,
            OperatorType::Concat => 28,
            OperatorType::Split => 29,
            OperatorType::Slice => 30,
            OperatorType::Tile => 31,
            OperatorType::Pad => 32,
            OperatorType::Crop => 33,
            OperatorType::Expand => 34,
            OperatorType::Flatten => 35,
            OperatorType::Squeeze => 36,
            OperatorType::Unsqueeze => 37,
            OperatorType::Gather => 38,
            OperatorType::Reduce => 39,
            OperatorType::Resize => 40,
            OperatorType::Normalize => 41,
            OperatorType::LSTM => 42,
            OperatorType::GRU => 43,
            OperatorType::Print => 44,
            OperatorType::Assert => 45,
            OperatorType::Custom => 0,
        };
        Self::write_u8(bytes, val);
    }

    fn write_tensor(bytes: &mut Vec<u8>, tensor: &Tensor) {
        Self::write_string(bytes, &tensor.name);
        Self::write_u32(bytes, tensor.shape.len() as u32);
        for dim in &tensor.shape {
            Self::write_u64(bytes, *dim as u64);
        }
        Self::write_u8(bytes, tensor.data_type as u8);
        Self::write_u8(bytes, tensor.layout as u8);

        // Write data
        match &tensor.data {
            crate::ir::tensor::TensorData::Empty => {
                Self::write_u32(bytes, 0);
            }
            crate::ir::tensor::TensorData::Owned(data) => {
                Self::write_bytes(bytes, data);
            }
            crate::ir::tensor::TensorData::Shared(data) => {
                Self::write_bytes(bytes, data.as_slice());
            }
        }

        // Write lifetime
        Self::write_u8(bytes, tensor.lifetime as u8);
    }

    // === Reading helpers ===

    fn read_string(bytes: &[u8], offset: usize) -> Result<(String, usize), ModelLoaderError> {
        let (len, mut pos) = Self::read_u32(bytes, offset)?;
        let len = len as usize;
        if pos + len > bytes.len() {
            return Err(ModelLoaderError::InvalidFormat("Unexpected end of buffer".into()));
        }
        let s = String::from_utf8_lossy(&bytes[pos..pos + len]).to_string();
        pos += len;
        Ok((s, pos))
    }

    fn read_u32(bytes: &[u8], offset: usize) -> Result<(u32, usize), ModelLoaderError> {
        if offset + 4 > bytes.len() {
            return Err(ModelLoaderError::InvalidFormat("Unexpected end of buffer".into()));
        }
        let v = u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]);
        Ok((v, offset + 4))
    }

    fn read_u64(bytes: &[u8], offset: usize) -> Result<(u64, usize), ModelLoaderError> {
        if offset + 8 > bytes.len() {
            return Err(ModelLoaderError::InvalidFormat("Unexpected end of buffer".into()));
        }
        let v = u64::from_le_bytes([
            bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3],
            bytes[offset+4], bytes[offset+5], bytes[offset+6], bytes[offset+7]
        ]);
        Ok((v, offset + 8))
    }

    fn read_u8(bytes: &[u8], offset: usize) -> Result<(u8, usize), ModelLoaderError> {
        if offset >= bytes.len() {
            return Err(ModelLoaderError::InvalidFormat("Unexpected end of buffer".into()));
        }
        Ok((bytes[offset], offset + 1))
    }

    fn read_bytes(bytes: &[u8], offset: usize) -> Result<(Vec<u8>, usize), ModelLoaderError> {
        let (len, mut pos) = Self::read_u32(bytes, offset)?;
        let len = len as usize;
        if pos + len > bytes.len() {
            return Err(ModelLoaderError::InvalidFormat("Unexpected end of buffer".into()));
        }
        let data = bytes[pos..pos + len].to_vec();
        pos += len;
        Ok((data, pos))
    }

    fn read_graph_io(bytes: &[u8], offset: usize) -> Result<(GraphIO, usize), ModelLoaderError> {
        let (name, mut pos) = Self::read_string(bytes, offset)?;
        let (io, new_pos) = Self::read_node_io(bytes, pos)?;
        pos = new_pos;
        let (is_input, new_pos) = Self::read_u8(bytes, pos)?;
        pos = new_pos;
        let (is_output, new_pos) = Self::read_u8(bytes, pos)?;
        pos = new_pos;
        Ok((GraphIO {
            name: name.clone(),
            io: NodeIO {
                tensor_name: name,
                data_type: io.data_type,
            },
            is_model_input: is_input != 0,
            is_model_output: is_output != 0,
        }, pos))
    }

    fn read_node_io(bytes: &[u8], offset: usize) -> Result<(NodeIO, usize), ModelLoaderError> {
        let (name, mut pos) = Self::read_string(bytes, offset)?;
        let (dtype, new_pos) = Self::read_u8(bytes, pos)?;
        pos = new_pos;
        Ok((NodeIO {
            tensor_name: name,
            data_type: Self::u8_to_datatype(dtype),
        }, pos))
    }

    fn read_node(bytes: &[u8], offset: usize, index: u32) -> Result<(Node, usize), ModelLoaderError> {
        let mut pos = offset;

        let (id, new_pos) = Self::read_u32(bytes, pos)?;
        pos = new_pos;

        let (name, new_pos) = Self::read_string(bytes, pos)?;
        pos = new_pos;

        let (op_type, new_pos) = Self::read_operator_type(bytes, pos)?;
        pos = new_pos;

        let (num_inputs, new_pos) = Self::read_u32(bytes, pos)?;
        pos = new_pos;
        let mut inputs = Vec::new();
        for _ in 0..num_inputs {
            let (input, new_pos) = Self::read_node_io(bytes, pos)?;
            inputs.push(input);
            pos = new_pos;
        }

        let (num_outputs, new_pos) = Self::read_u32(bytes, pos)?;
        pos = new_pos;
        let mut outputs = Vec::new();
        for _ in 0..num_outputs {
            let (output, new_pos) = Self::read_node_io(bytes, pos)?;
            outputs.push(output);
            pos = new_pos;
        }

        Ok((Node {
            id: if id == 0 { index } else { id },
            name,
            operator_type: op_type,
            inputs,
            outputs,
            fusion: None,
        }, pos))
    }

    fn read_operator_type(bytes: &[u8], offset: usize) -> Result<(OperatorType, usize), ModelLoaderError> {
        let (val, pos) = Self::read_u8(bytes, offset)?;
        // val 0 = Custom, val 1+ = known types
        // Note: This is a simplified mapping. In production, you'd want a proper mapping table.
        let op = match val {
            1 => OperatorType::Conv2d,
            2 => OperatorType::ConvTranspose2d,
            3 => OperatorType::MaxPool2d,
            4 => OperatorType::AvgPool2d,
            5 => OperatorType::GlobalAvgPool2d,
            6 => OperatorType::GlobalMaxPool2d,
            7 => OperatorType::FullyConnected,
            8 => OperatorType::ReLU,
            9 => OperatorType::ReLU6,
            10 => OperatorType::Sigmoid,
            11 => OperatorType::Tanh,
            12 => OperatorType::Softmax,
            13 => OperatorType::GELU,
            14 => OperatorType::SiLU,
            15 => OperatorType::BatchNorm,
            16 => OperatorType::LayerNorm,
            17 => OperatorType::InstanceNorm,
            18 => OperatorType::SelfAttention,
            19 => OperatorType::MultiHeadAttention,
            20 => OperatorType::Add,
            21 => OperatorType::Sub,
            22 => OperatorType::Mul,
            23 => OperatorType::Div,
            24 => OperatorType::MatMul,
            25 => OperatorType::Broadcast,
            26 => OperatorType::Reshape,
            27 => OperatorType::Transpose,
            28 => OperatorType::Concat,
            29 => OperatorType::Split,
            30 => OperatorType::Slice,
            31 => OperatorType::Tile,
            32 => OperatorType::Pad,
            33 => OperatorType::Crop,
            34 => OperatorType::Expand,
            35 => OperatorType::Flatten,
            36 => OperatorType::Squeeze,
            37 => OperatorType::Unsqueeze,
            38 => OperatorType::Gather,
            39 => OperatorType::Reduce,
            40 => OperatorType::Resize,
            41 => OperatorType::Normalize,
            42 => OperatorType::LSTM,
            43 => OperatorType::GRU,
            44 => OperatorType::Print,
            45 => OperatorType::Assert,
            _ => OperatorType::Custom,
        };
        Ok((op, pos))
    }

    fn read_tensor(bytes: &[u8], offset: usize) -> Result<(Tensor, usize), ModelLoaderError> {
        let mut pos = offset;

        let (name, new_pos) = Self::read_string(bytes, pos)?;
        pos = new_pos;

        let (num_dims, new_pos) = Self::read_u32(bytes, pos)?;
        pos = new_pos;
        let mut shape = Vec::new();
        for _ in 0..num_dims {
            let (dim, new_pos) = Self::read_u64(bytes, pos)?;
            shape.push(dim as usize);
            pos = new_pos;
        }

        let (dtype, new_pos) = Self::read_u8(bytes, pos)?;
        pos = new_pos;
        let (layout, new_pos) = Self::read_u8(bytes, pos)?;
        pos = new_pos;

        let (data, new_pos) = Self::read_bytes(bytes, pos)?;
        pos = new_pos;

        let (lifetime, new_pos) = Self::read_u8(bytes, pos)?;
        pos = new_pos;

        Ok((Tensor {
            name,
            shape,
            data_type: Self::u8_to_datatype(dtype),
            layout: Self::u8_to_layout(layout),
            data: crate::ir::tensor::TensorData::Owned(data),
            quantization: None,
            lifetime: Self::u8_to_lifetime(lifetime),
        }, pos))
    }

    fn u8_to_datatype(v: u8) -> DataType {
        match v {
            0 => DataType::F32,
            1 => DataType::F16,
            2 => DataType::F64,
            3 => DataType::I8,
            4 => DataType::I16,
            5 => DataType::I32,
            6 => DataType::I64,
            7 => DataType::U8,
            8 => DataType::U16,
            9 => DataType::U32,
            10 => DataType::U64,
            11 => DataType::Bool,
            12 => DataType::QUInt8,
            13 => DataType::QInt8,
            14 => DataType::QInt32,
            _ => DataType::F32,
        }
    }

    fn u8_to_layout(v: u8) -> StorageLayout {
        match v {
            0 => StorageLayout::NCHW,
            1 => StorageLayout::NHWC,
            2 => StorageLayout::NCHWc,
            3 => StorageLayout::OIHW,
            4 => StorageLayout::GOIHW,
            5 => StorageLayout::Constant,
            6 => StorageLayout::Default,
            _ => StorageLayout::NCHW,
        }
    }

    fn u8_to_lifetime(v: u8) -> TensorLifetime {
        match v {
            0 => TensorLifetime::Static,
            1 => TensorLifetime::Temporary,
            2 => TensorLifetime::Input,
            3 => TensorLifetime::Output,
            _ => TensorLifetime::Static,
        }
    }
}

impl Debug for NativeSerializer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeSerializer").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let model = ModelFile {
            format: ModelFormat::Native,
            metadata: ModelMetadata {
                name: "test_model".to_string(),
                version: "1.0".to_string(),
                ..Default::default()
            },
            ir_version: 1,
            graph: Graph::new("test_graph".to_string()),
            extra_data: HashMap::new(),
        };

        let bytes = NativeSerializer::serialize(&model).unwrap();
        let deserialized = NativeSerializer::deserialize(&bytes).unwrap();

        assert_eq!(deserialized.metadata.name, "test_model");
        assert_eq!(deserialized.graph.name, "test_graph");
        assert_eq!(deserialized.format, ModelFormat::Native);
    }

    #[test]
    fn test_magic_number() {
        let model = ModelFile::new(ModelFormat::Native, Graph::new("test".to_string()));
        let bytes = NativeSerializer::serialize(&model).unwrap();

        assert_eq!(&bytes[0..LIGHTSHIP_MAGIC.len()], LIGHTSHIP_MAGIC);
    }
}
