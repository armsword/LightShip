//! Quantization scheme definitions
//!
//! Defines quantization schemes for converting between
//! full-precision and quantized tensor representations.

use crate::common::DataType;
use std::fmt;

/// Quantization type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizationType {
    /// Symmetric quantization (zero-point = 0)
    Symmetric,
    /// Asymmetric quantization (zero-point != 0)
    Asymmetric,
    /// Per-tensor quantization (single scale for entire tensor)
    PerTensor,
    /// Per-channel quantization (scale per channel)
    PerChannel,
}

impl Default for QuantizationType {
    fn default() -> Self {
        QuantizationType::PerTensor
    }
}

impl fmt::Display for QuantizationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantizationType::Symmetric => write!(f, "Symmetric"),
            QuantizationType::Asymmetric => write!(f, "Asymmetric"),
            QuantizationType::PerTensor => write!(f, "PerTensor"),
            QuantizationType::PerChannel => write!(f, "PerChannel"),
        }
    }
}

/// Quantization axis for per-channel quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizationAxis {
    /// Quantize along channel dimension (typically axis=1 for NCHW)
    Channel,
    /// Quantize along spatial dimensions
    Spatial,
    /// Quantize along batch dimension
    Batch,
}

impl Default for QuantizationAxis {
    fn default() -> Self {
        QuantizationAxis::Channel
    }
}

/// Quantization parameters
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationParameters {
    /// Scale values (one per channel for per-channel, one for per-tensor)
    pub scales: Vec<f32>,
    /// Zero-point offset (0 for symmetric quantization)
    pub zero_points: Vec<i32>,
    /// Bit width (typically 8, but can be 4, 2, etc.)
    pub bit_width: u8,
    /// Minimum value in quantized range
    pub quantized_min: i32,
    /// Maximum value in quantized range
    pub quantized_max: i32,
}

impl QuantizationParameters {
    /// Create new quantization parameters for per-tensor
    pub fn new_per_tensor(scale: f32, zero_point: i32, bit_width: u8) -> Self {
        let (qmin, qmax) = Self::range_for_bits(bit_width);
        Self {
            scales: vec![scale],
            zero_points: vec![zero_point],
            bit_width,
            quantized_min: qmin,
            quantized_max: qmax,
        }
    }

    /// Create new quantization parameters for per-channel
    pub fn new_per_channel(scales: Vec<f32>, zero_points: Vec<i32>, _axis: QuantizationAxis, bit_width: u8) -> Self {
        let (qmin, qmax) = Self::range_for_bits(bit_width);
        Self {
            scales,
            zero_points,
            bit_width,
            quantized_min: qmin,
            quantized_max: qmax,
        }
    }

    /// Get quantized range based on bit width
    pub fn range_for_bits(bits: u8) -> (i32, i32) {
        match bits {
            8 => (i32::MIN, i32::MAX),
            7 => (-64, 63),
            4 => (-8, 7),
            2 => (-2, 1),
            _ => (i32::MIN, i32::MAX),
        }
    }

    /// Check if quantization is symmetric
    pub fn is_symmetric(&self) -> bool {
        self.zero_points.iter().all(|&zp| zp == 0)
    }

    /// Get scale at a specific channel
    pub fn scale_at(&self, channel: usize) -> f32 {
        self.scales.get(channel).copied().unwrap_or(1.0)
    }

    /// Get zero-point at a specific channel
    pub fn zero_point_at(&self, channel: usize) -> i32 {
        self.zero_points.get(channel).copied().unwrap_or(0)
    }
}

impl Default for QuantizationParameters {
    fn default() -> Self {
        Self::new_per_tensor(1.0, 0, 8)
    }
}

/// Quantization scheme combining type and parameters
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationScheme {
    /// Quantization type
    pub quant_type: QuantizationType,
    /// Axis for per-channel quantization
    pub axis: QuantizationAxis,
    /// Source data type before quantization
    pub source_dtype: DataType,
    /// Target quantized data type
    pub target_dtype: DataType,
    /// Quantization parameters
    pub parameters: QuantizationParameters,
}

impl QuantizationScheme {
    /// Create a symmetric per-tensor quantization scheme
    pub fn symmetric(bit_width: u8) -> Self {
        Self {
            quant_type: QuantizationType::Symmetric,
            axis: QuantizationAxis::Channel,
            source_dtype: DataType::F32,
            target_dtype: DataType::QInt8,
            parameters: QuantizationParameters::new_per_tensor(1.0, 0, bit_width),
        }
    }

    /// Create an asymmetric per-tensor quantization scheme
    pub fn asymmetric(bit_width: u8) -> Self {
        Self {
            quant_type: QuantizationType::Asymmetric,
            axis: QuantizationAxis::Channel,
            source_dtype: DataType::F32,
            target_dtype: DataType::QUInt8,
            parameters: QuantizationParameters::new_per_tensor(1.0, 128, bit_width),
        }
    }

    /// Create a per-channel quantization scheme
    pub fn per_channel(scales: Vec<f32>, zero_points: Vec<i32>, axis: QuantizationAxis, bit_width: u8) -> Self {
        Self {
            quant_type: QuantizationType::PerChannel,
            axis,
            source_dtype: DataType::F32,
            target_dtype: DataType::QInt8,
            parameters: QuantizationParameters::new_per_channel(scales, zero_points, axis, bit_width),
        }
    }

    /// Create int8 symmetric quantization scheme
    pub fn int8_symmetric() -> Self {
        Self::symmetric(8)
    }

    /// Create int8 asymmetric quantization scheme
    pub fn int8_asymmetric() -> Self {
        Self::asymmetric(8)
    }

    /// Create uint8 asymmetric quantization scheme
    pub fn uint8_asymmetric() -> Self {
        let mut scheme = Self::asymmetric(8);
        scheme.target_dtype = DataType::QUInt8;
        scheme
    }

    /// Check if this scheme uses per-channel quantization
    pub fn is_per_channel(&self) -> bool {
        matches!(self.quant_type, QuantizationType::PerChannel)
    }
}

impl Default for QuantizationScheme {
    fn default() -> Self {
        Self::int8_symmetric()
    }
}
