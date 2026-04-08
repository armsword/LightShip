//! Operator definitions for LightShip IR

use super::graph::NodeIO;
use super::attribute::AttributeMap;
use std::fmt;

/// Operator type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatorType {
    // === Convolution operators ===
    /// 2D Convolution
    Conv2d,
    /// 2D Transposed Convolution
    ConvTranspose2d,

    // === Pooling operators ===
    /// Max Pooling 2D
    MaxPool2d,
    /// Average Pooling 2D
    AvgPool2d,
    /// Global Average Pooling 2D
    GlobalAvgPool2d,
    /// Global Max Pooling 2D
    GlobalMaxPool2d,

    // === Fully connected ===
    /// Fully Connected / GEMM
    FullyConnected,

    // === Activation functions ===
    /// ReLU activation
    ReLU,
    /// ReLU6 activation
    ReLU6,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Softmax activation
    Softmax,
    /// GELU activation
    GELU,
    /// SiLU (Swish) activation
    SiLU,

    // === Normalization operators ===
    /// Batch Normalization
    BatchNorm,
    /// Layer Normalization
    LayerNorm,
    /// Instance Normalization
    InstanceNorm,

    // === Attention operators ===
    /// Self Attention
    SelfAttention,
    /// Multi-Head Attention
    MultiHeadAttention,

    // === Broadcast operators ===
    /// Add (element-wise)
    Add,
    /// Subtract (element-wise)
    Sub,
    /// Multiply (element-wise)
    Mul,
    /// Divide (element-wise)
    Div,
    /// Matrix Multiply
    MatMul,
    /// Broadcast
    Broadcast,

    // === Element-wise reshape operators ===
    /// Reshape
    Reshape,
    /// Transpose
    Transpose,
    /// Concat
    Concat,
    /// Split
    Split,
    /// Slice
    Slice,
    /// Tile
    Tile,
    /// Pad
    Pad,
    /// Crop
    Crop,
    /// Expand
    Expand,
    /// Flatten
    Flatten,
    /// Squeeze
    Squeeze,
    /// Unsqueeze
    Unsqueeze,
    /// Gather
    Gather,
    /// Reduce
    Reduce,
    /// Resize / Interpolate
    Resize,
    /// Normalize
    Normalize,

    // === RNN operators ===
    /// LSTM
    LSTM,
    /// GRU
    GRU,

    // === Debug operators ===
    /// Print
    Print,
    /// Assert
    Assert,

    // === Custom operator ===
    /// Custom operator
    Custom,
}

impl OperatorType {
    /// Get operator category
    pub fn category(&self) -> &'static str {
        match self {
            OperatorType::Conv2d | OperatorType::ConvTranspose2d => "convolution",
            OperatorType::MaxPool2d | OperatorType::AvgPool2d | OperatorType::GlobalAvgPool2d | OperatorType::GlobalMaxPool2d => "pooling",
            OperatorType::FullyConnected => "fully_connected",
            OperatorType::ReLU | OperatorType::ReLU6 | OperatorType::Sigmoid | OperatorType::Tanh | OperatorType::Softmax | OperatorType::GELU | OperatorType::SiLU => "activation",
            OperatorType::BatchNorm | OperatorType::LayerNorm | OperatorType::InstanceNorm => "normalization",
            OperatorType::SelfAttention | OperatorType::MultiHeadAttention => "attention",
            OperatorType::Add | OperatorType::Sub | OperatorType::Mul | OperatorType::Div | OperatorType::MatMul | OperatorType::Broadcast => "broadcast",
            OperatorType::Reshape | OperatorType::Transpose | OperatorType::Concat | OperatorType::Split | OperatorType::Slice | OperatorType::Tile | OperatorType::Pad | OperatorType::Crop | OperatorType::Expand | OperatorType::Flatten | OperatorType::Squeeze | OperatorType::Unsqueeze | OperatorType::Gather | OperatorType::Reduce | OperatorType::Resize | OperatorType::Normalize => "reshape",
            OperatorType::LSTM | OperatorType::GRU => "rnn",
            OperatorType::Print | OperatorType::Assert => "debug",
            OperatorType::Custom => "custom",
        }
    }
}

impl fmt::Display for OperatorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            OperatorType::Conv2d => "Conv2d",
            OperatorType::ConvTranspose2d => "ConvTranspose2d",
            OperatorType::MaxPool2d => "MaxPool2d",
            OperatorType::AvgPool2d => "AvgPool2d",
            OperatorType::GlobalAvgPool2d => "GlobalAvgPool2d",
            OperatorType::GlobalMaxPool2d => "GlobalMaxPool2d",
            OperatorType::FullyConnected => "FullyConnected",
            OperatorType::ReLU => "ReLU",
            OperatorType::ReLU6 => "ReLU6",
            OperatorType::Sigmoid => "Sigmoid",
            OperatorType::Tanh => "Tanh",
            OperatorType::Softmax => "Softmax",
            OperatorType::GELU => "GELU",
            OperatorType::SiLU => "SiLU",
            OperatorType::BatchNorm => "BatchNorm",
            OperatorType::LayerNorm => "LayerNorm",
            OperatorType::InstanceNorm => "InstanceNorm",
            OperatorType::SelfAttention => "SelfAttention",
            OperatorType::MultiHeadAttention => "MultiHeadAttention",
            OperatorType::Add => "Add",
            OperatorType::Sub => "Sub",
            OperatorType::Mul => "Mul",
            OperatorType::Div => "Div",
            OperatorType::MatMul => "MatMul",
            OperatorType::Broadcast => "Broadcast",
            OperatorType::Reshape => "Reshape",
            OperatorType::Transpose => "Transpose",
            OperatorType::Concat => "Concat",
            OperatorType::Split => "Split",
            OperatorType::Slice => "Slice",
            OperatorType::Tile => "Tile",
            OperatorType::Pad => "Pad",
            OperatorType::Crop => "Crop",
            OperatorType::Expand => "Expand",
            OperatorType::Flatten => "Flatten",
            OperatorType::Squeeze => "Squeeze",
            OperatorType::Unsqueeze => "Unsqueeze",
            OperatorType::Gather => "Gather",
            OperatorType::Reduce => "Reduce",
            OperatorType::Resize => "Resize",
            OperatorType::Normalize => "Normalize",
            OperatorType::LSTM => "LSTM",
            OperatorType::GRU => "GRU",
            OperatorType::Print => "Print",
            OperatorType::Assert => "Assert",
            OperatorType::Custom => "Custom",
        };
        write!(f, "{}", s)
    }
}

/// Operator definition (metadata)
#[derive(Debug, Clone)]
pub struct OperatorDef {
    /// Operator name
    pub name: String,
    /// Operator type
    pub operator_type: OperatorType,
    /// Inputs
    pub inputs: Vec<NodeIO>,
    /// Outputs
    pub outputs: Vec<NodeIO>,
    /// Attributes
    pub attributes: AttributeMap,
}

impl OperatorDef {
    /// Create a new operator definition
    pub fn new(name: String, operator_type: OperatorType) -> Self {
        Self {
            name,
            operator_type,
            inputs: Vec::new(),
            outputs: Vec::new(),
            attributes: AttributeMap::new(),
        }
    }

    /// Add an input
    pub fn add_input(&mut self, input: NodeIO) {
        self.inputs.push(input);
    }

    /// Add an output
    pub fn add_output(&mut self, output: NodeIO) {
        self.outputs.push(output);
    }
}
