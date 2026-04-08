//! Common type definitions for LightShip

use std::fmt;

/// Data type enumeration for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    // Floating point types
    /// 32-bit floating point
    F32,
    /// 16-bit floating point
    F16,
    /// 64-bit floating point
    F64,

    // Signed integer types
    /// 8-bit signed integer
    I8,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,

    // Unsigned integer types
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer
    U16,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,

    /// Boolean type
    Bool,

    // Quantized types
    /// Quantized unsigned 8-bit integer
    QUInt8,
    /// Quantized signed 8-bit integer
    QInt8,
    /// Quantized signed 32-bit integer
    QInt32,
}

impl DataType {
    /// Returns the byte size of this data type
    pub fn byte_size(&self) -> usize {
        match self {
            DataType::F32 | DataType::I32 | DataType::U32 => 4,
            DataType::F16 | DataType::I16 | DataType::U16 => 2,
            DataType::F64 | DataType::I64 | DataType::U64 => 8,
            DataType::I8 | DataType::U8 | DataType::QUInt8 | DataType::QInt8 => 1,
            DataType::Bool => 1,
            DataType::QInt32 => 4,
        }
    }

    /// Returns true if this is a quantized type
    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            DataType::QUInt8 | DataType::QInt8 | DataType::QInt32
        )
    }

    /// Returns true if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, DataType::F32 | DataType::F16 | DataType::F64)
    }

    /// Returns true if this is an integer type
    pub fn is_int(&self) -> bool {
        matches!(
            self,
            DataType::I8
                | DataType::I16
                | DataType::I32
                | DataType::I64
                | DataType::U8
                | DataType::U16
                | DataType::U32
                | DataType::U64
        )
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DataType::F32 => "f32",
            DataType::F16 => "f16",
            DataType::F64 => "f64",
            DataType::I8 => "i8",
            DataType::I16 => "i16",
            DataType::I32 => "i32",
            DataType::I64 => "i64",
            DataType::U8 => "u8",
            DataType::U16 => "u16",
            DataType::U32 => "u32",
            DataType::U64 => "u64",
            DataType::Bool => "bool",
            DataType::QUInt8 => "quint8",
            DataType::QInt8 => "qint8",
            DataType::QInt32 => "qint32",
        };
        write!(f, "{}", s)
    }
}

/// Storage layout enumeration for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageLayout {
    /// Channels first (NCHW)
    NCHW,
    /// Channels last (NHWC)
    NHWC,
    /// Channels last with channel division (NCHWc)
    NCHWc,
    /// OIHW format for conv weights
    OIHW,
    /// Grouped OIHW format for grouped conv weights
    GOIHW,
    /// Constant/Scalar
    Constant,
    /// Default layout (usually NCHW for 4D tensors)
    Default,
}

impl Default for StorageLayout {
    fn default() -> Self {
        StorageLayout::NCHW
    }
}

impl fmt::Display for StorageLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            StorageLayout::NCHW => "NCHW",
            StorageLayout::NHWC => "NHWC",
            StorageLayout::NCHWc => "NCHWc",
            StorageLayout::OIHW => "OIHW",
            StorageLayout::GOIHW => "GOIHW",
            StorageLayout::Constant => "Constant",
            StorageLayout::Default => "Default",
        };
        write!(f, "{}", s)
    }
}

/// Tensor lifetime enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorLifetime {
    /// Static tensor (model weights)
    Static,
    /// Temporary tensor (operator output)
    Temporary,
    /// User input tensor
    Input,
    /// User output tensor
    Output,
}

/// Backend type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BackendType {
    /// CPU backend
    CPU,
    /// GPU backend
    GPU,
    /// NPU backend
    NPU,
    /// Vulkan compute backend
    Vulkan,
    /// Metal backend (Apple devices)
    Metal,
    /// DSP backend
    DSP,
}

impl fmt::Display for BackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            BackendType::CPU => "CPU",
            BackendType::GPU => "GPU",
            BackendType::NPU => "NPU",
            BackendType::Vulkan => "Vulkan",
            BackendType::Metal => "Metal",
            BackendType::DSP => "DSP",
        };
        write!(f, "{}", s)
    }
}

/// Model format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    /// LightShip native format
    Native,
    /// ONNX format
    ONNX,
    /// TensorFlow SavedModel format
    TensorFlow,
    /// TensorFlow Lite format
    TFLite,
    /// Caffe format
    Caffe,
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ModelFormat::Native => "Native",
            ModelFormat::ONNX => "ONNX",
            ModelFormat::TensorFlow => "TensorFlow",
            ModelFormat::TFLite => "TFLite",
            ModelFormat::Caffe => "Caffe",
        };
        write!(f, "{}", s)
    }
}

/// Inference mode enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InferenceMode {
    /// Synchronous inference
    Synchronous,
    /// Asynchronous inference
    Asynchronous,
}

impl Default for InferenceMode {
    fn default() -> Self {
        InferenceMode::Synchronous
    }
}

/// Inference execution stage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Loading stage
    Loading,
    /// Compiling stage
    Compiling,
    /// Executing stage
    Executing,
    /// Cleanup stage
    Cleanup,
}
