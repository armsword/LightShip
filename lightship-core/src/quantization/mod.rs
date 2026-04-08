//! Quantization support for LightShip
//!
//! This module provides quantization schemes and utilities for
//! converting between full-precision and quantized representations.

mod scheme;
mod scale;
mod helpers;

pub use scheme::{QuantizationScheme, QuantizationType, QuantizationAxis, QuantizationParameters};
pub use scale::ScaleEncoding;
pub use helpers::{quantize_tensor, dequantize_tensor, find_scale_zp, quantize_value, dequantize_value, compute_quantization_error};
