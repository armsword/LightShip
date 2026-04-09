//! Unit tests for CPU Backend

use lightship_core::backend::{Backend, CpuBackend, CpuBackendConfig};
use lightship_core::common::{BackendType, DataType};
use lightship_core::ir::{NodeIO, OperatorDef, OperatorType, Tensor};

#[test]
fn test_cpu_backend_creation() {
    let backend = CpuBackend::new();
    assert!(backend.is_available());
    assert_eq!(backend.backend_type(), BackendType::CPU);
}

#[test]
fn test_cpu_backend_capabilities() {
    let backend = CpuBackend::new();
    let caps = backend.capabilities();

    assert_eq!(caps.backend_type, BackendType::CPU);
    assert!(caps.supported_operators.contains(&OperatorType::Conv2d));
    assert!(caps.supported_operators.contains(&OperatorType::ReLU));
    assert!(caps.supported_operators.contains(&OperatorType::Add));
    assert!(caps.supported_data_types.contains(&DataType::F32));
    assert!(caps.max_threads >= 1);
}

#[test]
fn test_cpu_backend_config() {
    let config = CpuBackendConfig {
        num_threads: 4,
        use_simd: true,
        ..Default::default()
    };
    let backend = CpuBackend::with_config(config);
    assert_eq!(backend.num_threads(), 4);
}

#[test]
fn test_cpu_backend_compile_operator() {
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("test_relu".into(), OperatorType::ReLU);
    op_def.inputs.push(NodeIO {
        tensor_name: "input".into(),
        data_type: DataType::F32,
    });
    op_def.outputs.push(NodeIO {
        tensor_name: "output".into(),
        data_type: DataType::F32,
    });

    let input = Tensor::new("input".into(), vec![1, 3, 224, 224], DataType::F32);
    let output = Tensor::new("output".into(), vec![1, 3, 224, 224], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, &[&input], &[&output])
        .unwrap();

    assert_eq!(compiled.operator_type, OperatorType::ReLU);
}

#[test]
fn test_cpu_backend_allocate_deallocate() {
    let backend = CpuBackend::new();

    let block = backend.allocate(1024, 64).unwrap();
    assert_eq!(block.size, 1024);
    assert_eq!(block.alignment, 64);
    assert!(block.is_aligned());

    backend.deallocate(block);
}

#[test]
fn test_cpu_backend_allocate_invalid_alignment() {
    let backend = CpuBackend::new();

    let result = backend.allocate(1024, 3); // 3 is not power of 2
    assert!(result.is_err());
}

#[test]
fn test_cpu_backend_execute_relu() {
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

    // Create input tensor with known values: [-1.0, 0.0, 1.0, 2.0]
    let input = Tensor::from_data(
        "input".into(),
        vec![4],
        DataType::F32,
        vec![-1.0f32, 0.0, 1.0, 2.0],
    );
    let mut output = Tensor::new("output".into(), vec![4], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, &[&input], &[&output])
        .unwrap();

    let result = backend.execute(&compiled, &[&input], &mut [&mut output]);
    assert!(result.is_ok());

    // Verify ReLU output: max(x, 0) = [0.0, 0.0, 1.0, 2.0]
    let bytes = output.data_as_bytes();
    let output_data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(output_data, &[0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_cpu_backend_execute_add() {
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("add".into(), OperatorType::Add);
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

    // Create input tensors with known values
    let input_a = Tensor::from_data(
        "a".into(),
        vec![3],
        DataType::F32,
        vec![1.0f32, 2.0, 3.0],
    );
    let input_b = Tensor::from_data(
        "b".into(),
        vec![3],
        DataType::F32,
        vec![4.0f32, 5.0, 6.0],
    );
    let mut output = Tensor::new("c".into(), vec![3], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, &[&input_a, &input_b], &[&output])
        .unwrap();

    let result = backend.execute(&compiled, &[&input_a, &input_b], &mut [&mut output]);
    assert!(result.is_ok());

    // Verify Add output: a + b = [5.0, 7.0, 9.0]
    let bytes = output.data_as_bytes();
    let output_data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(output_data, &[5.0, 7.0, 9.0]);
}

#[test]
fn test_cpu_backend_synchronize() {
    let backend = CpuBackend::new();
    let result = backend.synchronize();
    assert!(result.is_ok());
}
