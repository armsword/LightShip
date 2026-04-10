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

#[test]
fn test_cpu_backend_execute_mul() {
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("mul".into(), OperatorType::Mul);
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
        vec![2.0f32, 3.0, 4.0],
    );
    let input_b = Tensor::from_data(
        "b".into(),
        vec![3],
        DataType::F32,
        vec![5.0f32, 6.0, 7.0],
    );
    let mut output = Tensor::new("c".into(), vec![3], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, &[&input_a, &input_b], &[&output])
        .unwrap();

    let result = backend.execute(&compiled, &[&input_a, &input_b], &mut [&mut output]);
    assert!(result.is_ok());

    // Verify Mul output: a * b = [10.0, 18.0, 28.0]
    let bytes = output.data_as_bytes();
    let output_data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(output_data, &[10.0, 18.0, 28.0]);
}

#[test]
fn test_cpu_backend_execute_sigmoid() {
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("sigmoid".into(), OperatorType::Sigmoid);
    op_def.inputs.push(NodeIO {
        tensor_name: "input".into(),
        data_type: DataType::F32,
    });
    op_def.outputs.push(NodeIO {
        tensor_name: "output".into(),
        data_type: DataType::F32,
    });

    // Create input: [0.0, 1.0, -1.0]
    // sigmoid(x) = 1 / (1 + exp(-x))
    // sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731, sigmoid(-1) ≈ 0.269
    let input = Tensor::from_data(
        "input".into(),
        vec![3],
        DataType::F32,
        vec![0.0f32, 1.0, -1.0],
    );
    let mut output = Tensor::new("output".into(), vec![3], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, &[&input], &[&output])
        .unwrap();

    let result = backend.execute(&compiled, &[&input], &mut [&mut output]);
    assert!(result.is_ok());

    // Verify sigmoid output
    let bytes = output.data_as_bytes();
    let output_data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Allow small floating point tolerance
    let tolerance = 0.01;
    assert!((output_data[0] - 0.5).abs() < tolerance);
    assert!((output_data[1] - 0.73105858).abs() < tolerance);
    assert!((output_data[2] - 0.26894142).abs() < tolerance);
}

#[test]
fn test_cpu_backend_execute_tanh() {
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("tanh".into(), OperatorType::Tanh);
    op_def.inputs.push(NodeIO {
        tensor_name: "input".into(),
        data_type: DataType::F32,
    });
    op_def.outputs.push(NodeIO {
        tensor_name: "output".into(),
        data_type: DataType::F32,
    });

    // Create input: [0.0, 1.0, -1.0]
    // tanh(0) = 0, tanh(1) ≈ 0.761594, tanh(-1) ≈ -0.761594
    let input = Tensor::from_data(
        "input".into(),
        vec![3],
        DataType::F32,
        vec![0.0f32, 1.0, -1.0],
    );
    let mut output = Tensor::new("output".into(), vec![3], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, &[&input], &[&output])
        .unwrap();

    let result = backend.execute(&compiled, &[&input], &mut [&mut output]);
    assert!(result.is_ok());

    // Verify tanh output
    let bytes = output.data_as_bytes();
    let output_data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let tolerance = 0.001;
    assert!((output_data[0] - 0.0).abs() < tolerance);
    assert!((output_data[1] - 0.761594).abs() < tolerance);
    assert!((output_data[2] - (-0.761594)).abs() < tolerance);
}

#[test]
fn test_cpu_backend_execute_relu6() {
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("relu6".into(), OperatorType::ReLU6);
    op_def.inputs.push(NodeIO {
        tensor_name: "input".into(),
        data_type: DataType::F32,
    });
    op_def.outputs.push(NodeIO {
        tensor_name: "output".into(),
        data_type: DataType::F32,
    });

    // Create input: [-2.0, -1.0, 0.0, 1.0, 3.0, 6.0, 8.0]
    // ReLU6: min(max(x, 0), 6) = [0, 0, 0, 1, 3, 6, 6]
    let input = Tensor::from_data(
        "input".into(),
        vec![7],
        DataType::F32,
        vec![-2.0f32, -1.0, 0.0, 1.0, 3.0, 6.0, 8.0],
    );
    let mut output = Tensor::new("output".into(), vec![7], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, &[&input], &[&output])
        .unwrap();

    let result = backend.execute(&compiled, &[&input], &mut [&mut output]);
    assert!(result.is_ok());

    // Verify ReLU6 output
    let bytes = output.data_as_bytes();
    let output_data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(output_data, &[0.0, 0.0, 0.0, 1.0, 3.0, 6.0, 6.0]);
}

#[test]
fn test_cpu_backend_execute_maxpool2d() {
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("maxpool".into(), OperatorType::MaxPool2d);
    op_def.inputs.push(NodeIO {
        tensor_name: "input".into(),
        data_type: DataType::F32,
    });
    op_def.outputs.push(NodeIO {
        tensor_name: "output".into(),
        data_type: DataType::F32,
    });

    // Create 2x2 input:
    // [1.0, 2.0]
    // [3.0, 4.0]
    // MaxPool with kernel=2, stride=2 should give [4.0]
    let input = Tensor::from_data(
        "input".into(),
        vec![1, 1, 2, 2],  // NCHW: batch=1, channels=1, height=2, width=2
        DataType::F32,
        vec![1.0f32, 2.0, 3.0, 4.0],
    );
    let mut output = Tensor::new("output".into(), vec![1, 1, 1, 1], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, &[&input], &[&output])
        .unwrap();

    let result = backend.execute(&compiled, &[&input], &mut [&mut output]);
    assert!(result.is_ok());

    // Verify MaxPool2d output: max of all elements = 4.0
    let bytes = output.data_as_bytes();
    let output_data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(output_data, &[4.0]);
}

#[test]
fn test_cpu_backend_execute_avgpool2d() {
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("avgpool".into(), OperatorType::AvgPool2d);
    op_def.inputs.push(NodeIO {
        tensor_name: "input".into(),
        data_type: DataType::F32,
    });
    op_def.outputs.push(NodeIO {
        tensor_name: "output".into(),
        data_type: DataType::F32,
    });

    // Create 2x2 input:
    // [1.0, 2.0]
    // [3.0, 4.0]
    // AvgPool with kernel=2, stride=2 should give [(1+2+3+4)/4] = [2.5]
    let input = Tensor::from_data(
        "input".into(),
        vec![1, 1, 2, 2],
        DataType::F32,
        vec![1.0f32, 2.0, 3.0, 4.0],
    );
    let mut output = Tensor::new("output".into(), vec![1, 1, 1, 1], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, &[&input], &[&output])
        .unwrap();

    let result = backend.execute(&compiled, &[&input], &mut [&mut output]);
    assert!(result.is_ok());

    // Verify AvgPool2d output: average = 2.5
    let bytes = output.data_as_bytes();
    let output_data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let tolerance = 0.001;
    assert!((output_data[0] - 2.5).abs() < tolerance);
}

#[test]
fn test_cpu_backend_execute_softmax() {
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

    // Softmax: exp(x_i) / sum(exp(x_j))
    // Input: [1.0, 2.0, 3.0]
    // exp([1,2,3]) = [e, e^2, e^3] ≈ [2.718, 7.389, 20.086]
    // sum ≈ 30.193
    // output ≈ [0.090, 0.245, 0.665]
    let input = Tensor::from_data(
        "input".into(),
        vec![3],
        DataType::F32,
        vec![1.0f32, 2.0, 3.0],
    );
    let mut output = Tensor::new("output".into(), vec![3], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, &[&input], &[&output])
        .unwrap();

    let result = backend.execute(&compiled, &[&input], &mut [&mut output]);
    assert!(result.is_ok());

    // Verify Softmax output
    let bytes = output.data_as_bytes();
    let output_data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Check sum = 1.0
    let sum: f32 = output_data.iter().sum();
    assert!((sum - 1.0).abs() < 0.001);

    // Check individual values
    let tolerance = 0.01;
    assert!((output_data[0] - 0.090).abs() < tolerance);
    assert!((output_data[1] - 0.245).abs() < tolerance);
    assert!((output_data[2] - 0.665).abs() < tolerance);
}

#[test]
fn test_cpu_backend_execute_div() {
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("div".into(), OperatorType::Div);
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

    // Input a: [6.0, 8.0, 10.0]
    // Input b: [2.0, 4.0, 5.0]
    // Output: [3.0, 2.0, 2.0]
    let input_a = Tensor::from_data(
        "a".into(),
        vec![3],
        DataType::F32,
        vec![6.0f32, 8.0, 10.0],
    );
    let input_b = Tensor::from_data(
        "b".into(),
        vec![3],
        DataType::F32,
        vec![2.0f32, 4.0, 5.0],
    );
    let mut output = Tensor::new("c".into(), vec![3], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, &[&input_a], &[&output])
        .unwrap();

    let result = backend.execute(&compiled, &[&input_a, &input_b], &mut [&mut output]);
    assert!(result.is_ok());

    let bytes = output.data_as_bytes();
    let output_data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let tolerance = 0.001;
    assert!((output_data[0] - 3.0).abs() < tolerance);
    assert!((output_data[1] - 2.0).abs() < tolerance);
    assert!((output_data[2] - 2.0).abs() < tolerance);
}

#[test]
fn test_cpu_backend_execute_sub() {
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("sub".into(), OperatorType::Sub);
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

    // Input a: [5.0, 10.0, 15.0]
    // Input b: [2.0, 3.0, 4.0]
    // Output: [3.0, 7.0, 11.0]
    let input_a = Tensor::from_data(
        "a".into(),
        vec![3],
        DataType::F32,
        vec![5.0f32, 10.0, 15.0],
    );
    let input_b = Tensor::from_data(
        "b".into(),
        vec![3],
        DataType::F32,
        vec![2.0f32, 3.0, 4.0],
    );
    let mut output = Tensor::new("c".into(), vec![3], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, &[&input_a], &[&output])
        .unwrap();

    let result = backend.execute(&compiled, &[&input_a, &input_b], &mut [&mut output]);
    assert!(result.is_ok());

    let bytes = output.data_as_bytes();
    let output_data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    assert_eq!(output_data, &[3.0, 7.0, 11.0]);
}

#[test]
fn test_cpu_backend_execute_matmul() {
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

    // 2x3 matrix A:
    // [1, 2, 3]
    // [4, 5, 6]
    //
    // 3x2 matrix B:
    // [7, 8]
    // [9, 10]
    // [11, 12]
    //
    // Result 2x2 matrix C:
    // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    let input_a = Tensor::from_data(
        "a".into(),
        vec![2, 3],  // 2 rows, 3 cols
        DataType::F32,
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    let input_b = Tensor::from_data(
        "b".into(),
        vec![3, 2],  // 3 rows, 2 cols
        DataType::F32,
        vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0],
    );
    let mut output = Tensor::new("c".into(), vec![2, 2], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, &[&input_a], &[&output])
        .unwrap();

    let result = backend.execute(&compiled, &[&input_a, &input_b], &mut [&mut output]);
    assert!(result.is_ok());

    let bytes = output.data_as_bytes();
    let output_data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // C[0,0]=58, C[0,1]=64, C[1,0]=139, C[1,1]=154
    let tolerance = 0.001;
    assert!((output_data[0] - 58.0).abs() < tolerance);  // Row 0, Col 0
    assert!((output_data[1] - 64.0).abs() < tolerance);  // Row 0, Col 1
    assert!((output_data[2] - 139.0).abs() < tolerance); // Row 1, Col 0
    assert!((output_data[3] - 154.0).abs() < tolerance); // Row 1, Col 1
}

#[test]
fn test_cpu_backend_execute_reshape() {
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("reshape".into(), OperatorType::Reshape);
    op_def.inputs.push(NodeIO {
        tensor_name: "input".into(),
        data_type: DataType::F32,
    });
    op_def.outputs.push(NodeIO {
        tensor_name: "output".into(),
        data_type: DataType::F32,
    });

    // Input: [1, 2, 3, 4] with shape [4]
    // Reshape to [2, 2]
    let input = Tensor::from_data(
        "input".into(),
        vec![4],
        DataType::F32,
        vec![1.0f32, 2.0, 3.0, 4.0],
    );
    let mut output = Tensor::new("output".into(), vec![2, 2], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, &[&input], &[&output])
        .unwrap();

    let result = backend.execute(&compiled, &[&input], &mut [&mut output]);
    assert!(result.is_ok());

    // Reshape should preserve data, just change shape
    let bytes = output.data_as_bytes();
    let output_data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    assert_eq!(output_data, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(output.shape, vec![2, 2]);
}

#[test]
fn test_cpu_backend_execute_transpose() {
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("transpose".into(), OperatorType::Transpose);
    op_def.inputs.push(NodeIO {
        tensor_name: "input".into(),
        data_type: DataType::F32,
    });
    op_def.outputs.push(NodeIO {
        tensor_name: "output".into(),
        data_type: DataType::F32,
    });

    // Input 2x3 matrix:
    // [1, 2, 3]
    // [4, 5, 6]
    // Transpose -> 3x2 matrix:
    // [1, 4]
    // [2, 5]
    // [3, 6]
    let input = Tensor::from_data(
        "input".into(),
        vec![2, 3],
        DataType::F32,
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    let mut output = Tensor::new("output".into(), vec![3, 2], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, &[&input], &[&output])
        .unwrap();

    let result = backend.execute(&compiled, &[&input], &mut [&mut output]);
    assert!(result.is_ok());

    // Verify transpose
    let bytes = output.data_as_bytes();
    let output_data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Row 0: [1, 4], Row 1: [2, 5], Row 2: [3, 6]
    assert_eq!(output_data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_cpu_backend_execute_conv2d() {
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

    // Input: 1x1x4x4, all ones
    // Filter: 1x1x3x3, all ones
    // Output: 1x1x2x2 with stride=1, pad=0
    // Convolution of all-ones with all-ones 3x3 filter = 9.0 at each position
    let input = Tensor::from_data(
        "input".into(),
        vec![1, 1, 4, 4],
        DataType::F32,
        vec![1.0f32; 16],
    );
    let filter = Tensor::from_data(
        "filter".into(),
        vec![1, 1, 3, 3],
        DataType::F32,
        vec![1.0f32; 9],
    );
    let mut output = Tensor::new("output".into(), vec![1, 1, 2, 2], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, &[&input, &filter], &[&output])
        .unwrap();

    let result = backend.execute(&compiled, &[&input, &filter], &mut [&mut output]);
    assert!(result.is_ok());

    // Each output element should be sum of 9 input elements = 9.0
    let bytes = output.data_as_bytes();
    let output_data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Output shape should be 1x1x2x2
    assert_eq!(output.shape, vec![1, 1, 2, 2]);

    // Each output value should be 9.0 (3x3 filter of all ones)
    let tolerance = 0.001;
    for val in &output_data {
        assert!((*val - 9.0).abs() < tolerance, "Expected 9.0, got {}", val);
    }
}

#[test]
fn test_cpu_backend_execute_batchnorm() {
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("batchnorm".into(), OperatorType::BatchNorm);
    op_def.inputs.push(NodeIO {
        tensor_name: "input".into(),
        data_type: DataType::F32,
    });
    op_def.outputs.push(NodeIO {
        tensor_name: "output".into(),
        data_type: DataType::F32,
    });

    // Input: 1x2x2x2 (N=1, C=2, H=2, W=2)
    // All values = 4.0
    // BatchNorm with default gamma=1, beta=0, running_mean=0, running_var=1
    // Should output (4 - 0) / sqrt(1 + eps) * 1 + 0 = 4.0
    let input = Tensor::from_data(
        "input".into(),
        vec![1, 2, 2, 2],
        DataType::F32,
        vec![4.0f32; 8],
    );
    let mut output = Tensor::new("output".into(), vec![1, 2, 2, 2], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, &[&input], &[&output])
        .unwrap();

    let result = backend.execute(&compiled, &[&input], &mut [&mut output]);
    assert!(result.is_ok());

    // Verify output shape
    assert_eq!(output.shape, vec![1, 2, 2, 2]);

    // Output values should be same as input (gamma=1, beta=0, var=1)
    let bytes = output.data_as_bytes();
    let output_data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let tolerance = 0.001;
    for val in &output_data {
        assert!((*val - 4.0).abs() < tolerance, "Expected 4.0, got {}", val);
    }
}

#[test]
fn test_cpu_backend_execute_fullyconnected() {
    let backend = CpuBackend::new();

    let mut op_def = OperatorDef::new("fc".into(), OperatorType::FullyConnected);
    op_def.inputs.push(NodeIO {
        tensor_name: "input".into(),
        data_type: DataType::F32,
    });
    op_def.inputs.push(NodeIO {
        tensor_name: "weight".into(),
        data_type: DataType::F32,
    });
    op_def.outputs.push(NodeIO {
        tensor_name: "output".into(),
        data_type: DataType::F32,
    });

    // Input: [1, 4] (batch=1, features=4)
    // Weight: [3, 4] (out_features=3, in_features=4)
    // Output: [1, 3] (batch=1, out_features=3)
    //
    // input = [1, 2, 3, 4]
    // weight = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
    // output[0] = 1*1 + 2*1 + 3*1 + 4*1 = 10
    // output[1] = 1*2 + 2*2 + 3*2 + 4*2 = 20
    // output[2] = 1*3 + 2*3 + 3*3 + 4*3 = 30
    let input = Tensor::from_data(
        "input".into(),
        vec![1, 4],
        DataType::F32,
        vec![1.0f32, 2.0, 3.0, 4.0],
    );
    let weight = Tensor::from_data(
        "weight".into(),
        vec![3, 4],
        DataType::F32,
        vec![1.0f32, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0],
    );
    let mut output = Tensor::new("output".into(), vec![1, 3], DataType::F32);

    let compiled = backend
        .compile_operator(&op_def, &[&input, &weight], &[&output])
        .unwrap();

    let result = backend.execute(&compiled, &[&input, &weight], &mut [&mut output]);
    assert!(result.is_ok());

    // Verify output
    let bytes = output.data_as_bytes();
    let output_data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // FC: output = input @ weight.T
    // [1,3] = [1,4] @ [4,3]
    // Row 0: [1*1+2*1+3*1+4*1, 1*2+2*2+3*2+4*2, 1*3+2*3+3*3+4*3]
    //       = [10, 20, 30]
    let tolerance = 0.001;
    assert!((output_data[0] - 10.0).abs() < tolerance);
    assert!((output_data[1] - 20.0).abs() < tolerance);
    assert!((output_data[2] - 30.0).abs() < tolerance);
}

#[test]
fn test_cpu_backend_conv2d_relu_chain() {
    // Test chain: Conv2d -> ReLU
    let backend = CpuBackend::new();

    // First: Conv2d
    let mut conv_def = OperatorDef::new("conv".into(), OperatorType::Conv2d);
    conv_def.inputs.push(NodeIO {
        tensor_name: "input".into(),
        data_type: DataType::F32,
    });
    conv_def.inputs.push(NodeIO {
        tensor_name: "filter".into(),
        data_type: DataType::F32,
    });
    conv_def.outputs.push(NodeIO {
        tensor_name: "conv_out".into(),
        data_type: DataType::F32,
    });

    // Input: [1, 1, 4, 4] - single channel 4x4
    let input = Tensor::from_data(
        "input".into(),
        vec![1, 1, 4, 4],
        DataType::F32,
        vec![
            // Row 0
            1.0f32, 2.0, 3.0, 4.0,
            // Row 1
            5.0f32, 6.0, 7.0, 8.0,
            // Row 2
            1.0f32, 2.0, 3.0, 4.0,
            // Row 3
            5.0f32, 6.0, 7.0, 8.0,
        ],
    );

    // Filter: [1, 1, 2, 2] - single 2x2 kernel, output 1 channel
    let filter = Tensor::from_data(
        "filter".into(),
        vec![1, 1, 2, 2],
        DataType::F32,
        vec![
            1.0f32, 0.0,
            0.0f32, 1.0f32,
        ],
    );

    let mut conv_output = Tensor::new("conv_out".into(), vec![1, 1, 3, 3], DataType::F32);

    let conv_compiled = backend
        .compile_operator(&conv_def, &[&input, &filter], &[&conv_output])
        .unwrap();

    let result = backend.execute(&conv_compiled, &[&input, &filter], &mut [&mut conv_output]);
    assert!(result.is_ok());

    // Verify conv output exists and has correct shape
    let conv_bytes = conv_output.data_as_bytes();
    assert_eq!(conv_bytes.len(), 1 * 1 * 3 * 3 * 4); // 36 bytes for 9 f32 values

    // Second: ReLU on conv output
    let mut relu_def = OperatorDef::new("relu".into(), OperatorType::ReLU);
    relu_def.inputs.push(NodeIO {
        tensor_name: "conv_out".into(),
        data_type: DataType::F32,
    });
    relu_def.outputs.push(NodeIO {
        tensor_name: "relu_out".into(),
        data_type: DataType::F32,
    });

    let mut relu_output = Tensor::new("relu_out".into(), vec![1, 1, 3, 3], DataType::F32);

    let relu_compiled = backend
        .compile_operator(&relu_def, &[&conv_output], &[&relu_output])
        .unwrap();

    let result = backend.execute(&relu_compiled, &[&conv_output], &mut [&mut relu_output]);
    assert!(result.is_ok());

    // Verify ReLU output: max(conv_out, 0)
    let relu_bytes = relu_output.data_as_bytes();
    let relu_data: Vec<f32> = relu_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // All values should be non-negative after ReLU
    for val in &relu_data {
        assert!(*val >= 0.0, "ReLU output should be non-negative, got {}", val);
    }
}
