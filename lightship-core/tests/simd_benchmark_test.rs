//! SIMD performance benchmark tests
//!
//! These tests measure the performance of SIMD-accelerated operators
//! and verify that optimizations are working correctly.

use lightship_core::backend::{Backend, CpuBackend};
use lightship_core::common::DataType;
use lightship_core::ir::{NodeIO, OperatorDef, OperatorType, Tensor};
use std::time::Instant;

#[test]
fn test_relu_performance_small() {
    // Benchmark ReLU with small tensor (1024 elements)
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("relu".into(), OperatorType::ReLU);
    op_def.inputs.push(NodeIO {
        tensor_name: "input".into(),
        data_type: DataType::F32,
    });
    op_def.outputs.push(NodeIO {
        tensor_name: "output".into(),
        data_type: DataType::F32,
    });

    // Create input with random-ish values including negatives
    let input_data: Vec<f32> = (0..1024)
        .map(|i| if i % 3 == 0 { -1.0 } else { i as f32 })
        .collect();
    let input = Tensor::from_data("input".into(), vec![1024], DataType::F32, input_data);
    let mut output = Tensor::new("output".into(), vec![1024], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, None, &[&input], &[&output])
        .unwrap();

    let start = Instant::now();
    for _ in 0..100 {
        backend.execute(&compiled, &[&input], &mut [&mut output]).unwrap();
    }
    let elapsed = start.elapsed();

    // 100 iterations of 1024 elements should be very fast (< 100ms even without SIMD)
    println!("ReLU (1024 elements, 100 iterations): {:?}", elapsed);
    assert!(elapsed.as_millis() < 1000); // Should be much faster, but this is a generous upper bound
}

#[test]
fn test_relu_performance_large() {
    // Benchmark ReLU with large tensor (1024*1024 elements)
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("relu".into(), OperatorType::ReLU);
    op_def.inputs.push(NodeIO {
        tensor_name: "input".into(),
        data_type: DataType::F32,
    });
    op_def.outputs.push(NodeIO {
        tensor_name: "output".into(),
        data_type: DataType::F32,
    });

    // Create large input
    let input_data: Vec<f32> = (0..1024 * 1024)
        .map(|i| if i % 5 == 0 { -1.0 } else { (i as f32) % 1000.0 })
        .collect();
    let input = Tensor::from_data("input".into(), vec![1024 * 1024], DataType::F32, input_data);
    let mut output = Tensor::new("output".into(), vec![1024 * 1024], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, None, &[&input], &[&output])
        .unwrap();

    let start = Instant::now();
    backend.execute(&compiled, &[&input], &mut [&mut output]).unwrap();
    let elapsed = start.elapsed();

    println!("ReLU (1M elements, 1 iteration): {:?}", elapsed);
    // Verify correctness
    let output_bytes = output.data_as_bytes();
    let output_data: Vec<f32> = output_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // All values should be non-negative
    for (i, val) in output_data.iter().enumerate() {
        assert!(*val >= 0.0, "ReLU output should be non-negative at index {}", i);
    }
}

#[test]
fn test_matmul_performance() {
    // Benchmark MatMul: [128, 256] @ [256, 128] -> [128, 128]
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("matmul".into(), OperatorType::MatMul);
    op_def.inputs.push(NodeIO {
        tensor_name: "a".into(),
        data_type: DataType::F32,
    });
    op_def.inputs.push(NodeIO {
        tensor_name: "b".into(),
        data_type: DataType::F32,
    });
    op_def.outputs.push(NodeIO {
        tensor_name: "c".into(),
        data_type: DataType::F32,
    });

    let m = 128;
    let k = 256;
    let n = 128;

    // Create input matrices
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();

    let a = Tensor::from_data("a".into(), vec![m, k], DataType::F32, a_data);
    let b = Tensor::from_data("b".into(), vec![k, n], DataType::F32, b_data);
    let mut c = Tensor::new("c".into(), vec![m, n], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, None, &[&a, &b], &[&c])
        .unwrap();

    let start = Instant::now();
    backend.execute(&compiled, &[&a, &b], &mut [&mut c]).unwrap();
    let elapsed = start.elapsed();

    println!("MatMul (128x256 @ 256x128): {:?}", elapsed);

    // Verify output shape
    assert_eq!(c.shape, vec![m, n]);
}

#[test]
fn test_softmax_performance() {
    // Benchmark Softmax with 1000 elements
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("softmax".into(), OperatorType::Softmax);
    op_def.inputs.push(NodeIO {
        tensor_name: "input".into(),
        data_type: DataType::F32,
    });
    op_def.outputs.push(NodeIO {
        tensor_name: "output".into(),
        data_type: DataType::F32,
    });

    let input_data: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.1).collect();
    let input = Tensor::from_data("input".into(), vec![1000], DataType::F32, input_data);
    let mut output = Tensor::new("output".into(), vec![1000], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, None, &[&input], &[&output])
        .unwrap();

    let start = Instant::now();
    backend.execute(&compiled, &[&input], &mut [&mut output]).unwrap();
    let elapsed = start.elapsed();

    println!("Softmax (1000 elements): {:?}", elapsed);

    // Verify softmax properties: all outputs should be positive and sum to ~1.0
    let output_bytes = output.data_as_bytes();
    let output_data: Vec<f32> = output_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let sum: f32 = output_data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "Softmax output should sum to ~1.0, got {}",
        sum
    );

    for val in &output_data {
        assert!(*val > 0.0, "Softmax output should be positive");
        assert!(*val <= 1.0, "Softmax output should be <= 1.0");
    }
}

#[test]
fn test_conv2d_performance() {
    // Benchmark Conv2d with typical image size
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("conv".into(), OperatorType::Conv2d);
    op_def.inputs.push(NodeIO {
        tensor_name: "input".into(),
        data_type: DataType::F32,
    });
    op_def.inputs.push(NodeIO {
        tensor_name: "filter".into(),
        data_type: DataType::F32,
    });
    op_def.outputs.push(NodeIO {
        tensor_name: "output".into(),
        data_type: DataType::F32,
    });

    // Input: [1, 3, 32, 32] (small image batch)
    let input_data: Vec<f32> = (0..1 * 3 * 32 * 32).map(|i| (i as f32) * 0.01).collect();
    let input = Tensor::from_data("input".into(), vec![1, 3, 32, 32], DataType::F32, input_data);

    // Filter: [16, 3, 3, 3] (16 output channels, 3 input channels, 3x3 kernel)
    let filter_data: Vec<f32> = (0..16 * 3 * 3 * 3).map(|_i| 0.1).collect();
    let filter = Tensor::from_data("filter".into(), vec![16, 3, 3, 3], DataType::F32, filter_data);

    let mut output = Tensor::new("output".into(), vec![1, 16, 30, 30], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, None, &[&input, &filter], &[&output])
        .unwrap();

    let start = Instant::now();
    backend.execute(&compiled, &[&input, &filter], &mut [&mut output]).unwrap();
    let elapsed = start.elapsed();

    println!("Conv2d (1x3x32x32 @ 16x3x3x3): {:?}", elapsed);
    assert_eq!(output.shape, vec![1, 16, 30, 30]);
}
