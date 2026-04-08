//! Tensor data structure for LightShip IR

use crate::common::{DataType, StorageLayout, TensorLifetime};
use std::sync::Arc;

/// Tensor shape type
pub type TensorShape = Vec<usize>;

/// Quantization parameters
#[derive(Debug, Clone)]
pub struct Quantization {
    /// Quantization scale(s)
    pub scale: Vec<f32>,
    /// Zero point(s)
    pub zero_point: Vec<i32>,
    /// Bit width
    pub bit_width: u8,
}

/// Tensor data storage
#[derive(Debug, Clone)]
pub enum TensorData {
    /// Empty tensor (no data)
    Empty,
    /// Owned data
    Owned(Vec<u8>),
    /// Shared data reference
    Shared(Arc<Vec<u8>>),
}

impl Default for TensorData {
    fn default() -> Self {
        TensorData::Empty
    }
}

/// Tensor in the IR
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor name
    pub name: String,
    /// Tensor shape
    pub shape: TensorShape,
    /// Data type
    pub data_type: DataType,
    /// Storage layout
    pub layout: StorageLayout,
    /// Tensor data
    pub data: TensorData,
    /// Quantization parameters
    pub quantization: Option<Quantization>,
    /// Lifetime
    pub lifetime: TensorLifetime,
}

impl Tensor {
    /// Create a new tensor with shape and data type
    pub fn new(name: String, shape: TensorShape, data_type: DataType) -> Self {
        Self {
            name,
            shape,
            data_type,
            layout: StorageLayout::Default,
            data: TensorData::Empty,
            quantization: None,
            lifetime: TensorLifetime::Temporary,
        }
    }

    /// Get the number of elements
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the byte size of the tensor data
    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.data_type.byte_size()
    }

    /// Get the rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Check if tensor is static (weight)
    pub fn is_static(&self) -> bool {
        self.lifetime == TensorLifetime::Static
    }

    /// Check if tensor is quantized
    pub fn is_quantized(&self) -> bool {
        self.quantization.is_some()
    }
}
