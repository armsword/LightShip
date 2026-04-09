//! Quantization helper functions
//!
//! Provides utilities for quantizing and dequantizing tensors.

use crate::common::DataType;
use crate::ir::{Tensor, TensorData};

use super::scheme::{QuantizationParameters, QuantizationScheme};

/// Quantize a tensor according to the given scheme
pub fn quantize_tensor(tensor: &Tensor, scheme: &QuantizationScheme) -> Tensor {
    let params = &scheme.parameters;
    let scale = params.scales.first().copied().unwrap_or(1.0);
    let zero_point = params.zero_points.first().copied().unwrap_or(0);
    let dtype = scheme.target_dtype;

    // Get F32 data from tensor
    let bytes = tensor.data_as_bytes();
    let f32_data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let quantized: Vec<i32> = f32_data
        .iter()
        .map(|&v| quantize_value(v, scale, zero_point, dtype))
        .collect();

    // Convert to bytes based on target type
    let bytes: Vec<u8> = match dtype {
        DataType::QUInt8 => quantized.iter().map(|&v| v as u8).collect(),
        DataType::QInt8 => quantized.iter().map(|&v| v as i8 as u8).collect(),
        DataType::QInt32 => {
            let mut b = Vec::with_capacity(quantized.len() * 4);
            for &v in &quantized {
                b.extend_from_slice(&v.to_le_bytes());
            }
            b
        }
        _ => return tensor.clone(),
    };

    let mut result = Tensor::new(
        format!("{}_quantized", tensor.name),
        tensor.shape.clone(),
        dtype,
    );
    result.data = TensorData::Owned(bytes);
    result
}

/// Dequantize a tensor according to the given scheme
pub fn dequantize_tensor(tensor: &Tensor, scheme: &QuantizationScheme) -> Tensor {
    let params = &scheme.parameters;
    let scale = params.scales.first().copied().unwrap_or(1.0);
    let zero_point = params.zero_points.first().copied().unwrap_or(0);

    let dtype = tensor.data_type;

    // Get quantized bytes
    let bytes = tensor.data_as_bytes();

    // Dequantize based on type
    let float_values: Vec<f32> = match dtype {
        DataType::QUInt8 => bytes.iter().map(|&v| dequantize_value(v as i32, scale, zero_point)).collect(),
        DataType::QInt8 => bytes.iter().map(|&v| dequantize_value(v as i8 as i32, scale, zero_point)).collect(),
        DataType::QInt32 => {
            let mut values = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks(4) {
                if chunk.len() == 4 {
                    let v = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    values.push(dequantize_value(v, scale, zero_point));
                }
            }
            values
        }
        _ => return tensor.clone(),
    };

    // Convert f32 values to bytes
    let mut result_bytes = Vec::with_capacity(float_values.len() * 4);
    for &v in &float_values {
        result_bytes.extend_from_slice(&v.to_le_bytes());
    }

    let mut result = Tensor::new(
        format!("{}_dequantized", tensor.name),
        tensor.shape.clone(),
        DataType::F32,
    );
    result.data = TensorData::Owned(result_bytes);

    result
}

/// Find optimal scale and zero-point for a tensor
///
/// Uses the Min-Max quantization method:
/// - scale = (max - min) / (qmax - qmin)
/// - zero_point = qmin - min / scale
pub fn find_scale_zp(data: &[f32], dtype: DataType, is_symmetric: bool) -> QuantizationParameters {
    if data.is_empty() {
        return QuantizationParameters::default();
    }

    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let (qmin, qmax) = match dtype {
        DataType::QUInt8 => (0i32, 255i32),
        DataType::QInt8 => (-128i32, 127i32),
        DataType::QInt32 => (i32::MIN, i32::MAX),
        _ => (0i32, 255i32),
    };

    if is_symmetric {
        // Symmetric: zero_point = 0
        let abs_max = max_val.abs().max(min_val.abs());
        let scale = if abs_max > 0.0 {
            abs_max / (qmax as f32)
        } else {
            1.0
        };
        QuantizationParameters::new_per_tensor(scale, 0, 8)
    } else {
        // Asymmetric
        let scale = if max_val > min_val {
            (max_val - min_val) / ((qmax - qmin) as f32)
        } else {
            1.0
        };
        let zero_point = qmin as f32 - min_val / scale;
        let zero_point_i32 = zero_point.round() as i32;
        QuantizationParameters::new_per_tensor(scale, zero_point_i32, 8)
    }
}

/// Quantize a single value
pub fn quantize_value(value: f32, scale: f32, zero_point: i32, dtype: DataType) -> i32 {
    let qmin = match dtype {
        DataType::QUInt8 => 0i32,
        DataType::QInt8 => -128i32,
        DataType::QInt32 => i32::MIN,
        _ => 0i32,
    };
    let qmax = match dtype {
        DataType::QUInt8 => 255i32,
        DataType::QInt8 => 127i32,
        DataType::QInt32 => i32::MAX,
        _ => 255i32,
    };

    let quantized = (value / scale).round() as i32 + zero_point as i32;
    quantized.clamp(qmin, qmax)
}

/// Dequantize a single value
pub fn dequantize_value(quantized: i32, scale: f32, zero_point: i32) -> f32 {
    (quantized as f32 - zero_point as f32) * scale
}

/// Compute MSE between original and quantized-dequantized values
pub fn compute_quantization_error(original: &[f32], reconstructed: &[f32]) -> f32 {
    if original.len() != reconstructed.len() {
        return f32::INFINITY;
    }

    let sum_sq_error: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(o, r)| (o - r).powi(2))
        .sum();

    let mse = sum_sq_error / original.len() as f32;
    mse.sqrt()
}
