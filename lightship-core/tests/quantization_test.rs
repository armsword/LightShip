//! Unit tests for quantization module

use lightship_core::common::DataType;
use lightship_core::quantization::{
    find_scale_zp, quantize_value, dequantize_value, compute_quantization_error,
    QuantizationScheme, QuantizationType, QuantizationAxis, QuantizationParameters,
    ScaleEncoding,
};

#[test]
fn test_quantization_parameters_per_tensor() {
    let params = QuantizationParameters::new_per_tensor(0.1, 0, 8);

    assert_eq!(params.scales.len(), 1);
    assert_eq!(params.zero_points.len(), 1);
    assert_eq!(params.bit_width, 8);
    assert!(params.is_symmetric());
}

#[test]
fn test_quantization_parameters_per_channel() {
    let scales = vec![0.1, 0.2, 0.3, 0.4];
    let zero_points = vec![0, 0, 0, 0];
    let params =
        QuantizationParameters::new_per_channel(scales.clone(), zero_points, QuantizationAxis::Channel, 8);

    assert_eq!(params.scales.len(), 4);
    assert_eq!(params.scale_at(0), 0.1);
    assert_eq!(params.scale_at(2), 0.3);
    assert!(params.is_symmetric());
}

#[test]
fn test_quantization_parameters_asymmetric() {
    let params = QuantizationParameters::new_per_tensor(0.1, 128, 8);

    assert!(!params.is_symmetric());
    assert_eq!(params.zero_point_at(0), 128);
}

#[test]
fn test_quantization_scheme_symmetric() {
    let scheme = QuantizationScheme::symmetric(8);

    assert_eq!(scheme.quant_type, QuantizationType::Symmetric);
    assert_eq!(scheme.source_dtype, DataType::F32);
    assert_eq!(scheme.target_dtype, DataType::QInt8);
}

#[test]
fn test_quantization_scheme_asymmetric() {
    let scheme = QuantizationScheme::asymmetric(8);

    assert_eq!(scheme.quant_type, QuantizationType::Asymmetric);
    assert_eq!(scheme.target_dtype, DataType::QUInt8);
}

#[test]
fn test_quantization_scheme_uint8() {
    let scheme = QuantizationScheme::uint8_asymmetric();

    assert_eq!(scheme.target_dtype, DataType::QUInt8);
    assert!(!scheme.parameters.is_symmetric());
}

#[test]
fn test_quantization_scheme_int8() {
    let scheme = QuantizationScheme::int8_symmetric();

    assert_eq!(scheme.target_dtype, DataType::QInt8);
    assert!(scheme.parameters.is_symmetric());
}

#[test]
fn test_quantization_scheme_per_channel() {
    let scales = vec![0.1, 0.2, 0.3];
    let zero_points = vec![0, 0, 0];
    let scheme = QuantizationScheme::per_channel(scales, zero_points, QuantizationAxis::Channel, 8);

    assert!(scheme.is_per_channel());
    assert_eq!(scheme.parameters.scales.len(), 3);
}

#[test]
fn test_find_scale_zp_symmetric() {
    let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    let params = find_scale_zp(&data, DataType::QInt8, true);

    assert!(params.is_symmetric());
    assert_eq!(params.zero_points[0], 0);
}

#[test]
fn test_find_scale_zp_asymmetric() {
    let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    let params = find_scale_zp(&data, DataType::QUInt8, false);

    assert!(!params.is_symmetric());
}

#[test]
fn test_quantize_dequantize_roundtrip() {
    let scale = 0.01;
    let zero_point = 0i32;
    let original = 0.5f32;

    let quantized = quantize_value(original, scale, zero_point, DataType::QInt8);
    let dequantized = dequantize_value(quantized, scale, zero_point);

    assert!((dequantized - original).abs() < 0.1);
}

#[test]
fn test_scale_encoding() {
    let float32 = ScaleEncoding::Float32;
    assert!(!format!("{}", float32).is_empty());

    let blockwise = ScaleEncoding::BlockWise { block_size: 32 };
    assert!(format!("{}", blockwise).contains("32"));

    let lut = ScaleEncoding::LookupTable { num_entries: 256 };
    assert!(format!("{}", lut).contains("256"));
}

#[test]
fn test_compute_quantization_error() {
    let original = vec![1.0, 2.0, 3.0];
    let reconstructed = vec![1.0, 2.0, 3.0];

    let error = compute_quantization_error(&original, &reconstructed);
    assert_eq!(error, 0.0);
}

#[test]
fn test_compute_quantization_error_mismatch() {
    let original = vec![1.0, 2.0, 3.0];
    let reconstructed = vec![1.0, 2.1, 2.9];

    let error = compute_quantization_error(&original, &reconstructed);
    assert!(error > 0.0);
}

#[test]
fn test_compute_quantization_error_different_lengths() {
    let original = vec![1.0, 2.0, 3.0];
    let reconstructed = vec![1.0, 2.0];

    let error = compute_quantization_error(&original, &reconstructed);
    assert_eq!(error, f32::INFINITY);
}
