//! CPU Backend implementation for LightShip

use std::fmt::Debug;
use crate::backend::{
    Backend, BackendCapabilities, BackendSpecificData, CompiledOperator, CpuBackendConfig,
    MemoryBlock, SimdFlags,
};
use crate::backend::memory::StorageLocation;
use crate::common::{BackendType, DataType, LightShipError, Result, StorageLayout};
use crate::common::error::BackendError;
use crate::ir::{OperatorDef, OperatorType, Tensor};
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

static BLOCK_ID: AtomicU64 = AtomicU64::new(0);

/// CPU Backend implementation
pub struct CpuBackend {
    config: CpuBackendConfig,
    capabilities: BackendCapabilities,
}

impl Debug for CpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuBackend")
            .field("config", &self.config)
            .field("capabilities", &self.capabilities)
            .finish()
    }
}

impl CpuBackend {
    /// Create a new CPU backend with default configuration
    pub fn new() -> Self {
        Self::with_config(CpuBackendConfig::default())
    }

    /// Create a new CPU backend with custom configuration
    pub fn with_config(config: CpuBackendConfig) -> Self {
        let capabilities = Self::build_capabilities(&config);
        Self {
            config,
            capabilities,
        }
    }

    fn build_capabilities(config: &CpuBackendConfig) -> BackendCapabilities {
        let num_threads = if config.num_threads == 0 {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1)
        } else {
            config.num_threads
        };

        BackendCapabilities {
            backend_type: BackendType::CPU,
            supported_operators: vec![
                OperatorType::Conv2d,
                OperatorType::MaxPool2d,
                OperatorType::AvgPool2d,
                OperatorType::GlobalAvgPool2d,
                OperatorType::GlobalMaxPool2d,
                OperatorType::FullyConnected,
                OperatorType::ReLU,
                OperatorType::ReLU6,
                OperatorType::Sigmoid,
                OperatorType::Tanh,
                OperatorType::Softmax,
                OperatorType::BatchNorm,
                OperatorType::LayerNorm,
                OperatorType::Add,
                OperatorType::Sub,
                OperatorType::Mul,
                OperatorType::Div,
                OperatorType::MatMul,
                OperatorType::Reshape,
                OperatorType::Transpose,
                OperatorType::Concat,
                OperatorType::Flatten,
            ],
            supported_data_types: vec![
                DataType::F32,
                DataType::F16,
                DataType::I32,
                DataType::I8,
                DataType::U8,
            ],
            supported_layouts: vec![
                StorageLayout::NCHW,
                StorageLayout::NHWC,
                StorageLayout::OIHW,
            ],
            max_threads: num_threads,
            has_simd: config.use_simd,
            simd_flags: SimdFlags::default(),
            memory_alignment: 64,
            register_size: 256,
            feature_flags: Default::default(),
        }
    }

    /// Get the number of threads
    pub fn num_threads(&self) -> usize {
        self.capabilities.max_threads
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::CPU
    }

    fn is_available(&self) -> bool {
        true
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities.clone()
    }

    fn compile_operator(
        &self,
        def: &OperatorDef,
        _inputs: &[&Tensor],
        _outputs: &[&Tensor],
    ) -> Result<CompiledOperator> {
        Ok(CompiledOperator {
            operator_type: def.operator_type,
            backend_data: BackendSpecificData::Cpu(Vec::new()),
            workgroup_size: None,
        })
    }

    fn execute(
        &self,
        op: &CompiledOperator,
        inputs: &[&Tensor],
        outputs: &mut [&mut Tensor],
    ) -> Result<()> {
        match op.operator_type {
            OperatorType::ReLU => self.execute_relu(inputs, outputs),
            OperatorType::ReLU6 => self.execute_relu6(inputs, outputs),
            OperatorType::Add => self.execute_add(inputs, outputs),
            OperatorType::Mul => self.execute_mul(inputs, outputs),
            OperatorType::Sigmoid => self.execute_sigmoid(inputs, outputs),
            OperatorType::Tanh => self.execute_tanh(inputs, outputs),
            OperatorType::MaxPool2d => self.execute_maxpool2d(inputs, outputs),
            OperatorType::AvgPool2d => self.execute_avgpool2d(inputs, outputs),
            OperatorType::Softmax => self.execute_softmax(inputs, outputs),
            OperatorType::Div => self.execute_div(inputs, outputs),
            OperatorType::Sub => self.execute_sub(inputs, outputs),
            OperatorType::MatMul => self.execute_matmul(inputs, outputs),
            OperatorType::Reshape => self.execute_reshape(inputs, outputs),
            OperatorType::Transpose => self.execute_transpose(inputs, outputs),
            _ => {
                tracing::debug!(
                    "CPU backend: operator {:?} execution not yet implemented",
                    op.operator_type
                );
                Ok(())
            }
        }
    }

    fn allocate(&self, size: usize, alignment: usize) -> Result<MemoryBlock> {
        if !alignment.is_power_of_two() || alignment == 0 {
            return Err(LightShipError::InvalidParam(format!(
                "Invalid alignment: {}",
                alignment
            )));
        }

        let layout = unsafe { Layout::from_size_align_unchecked(size, alignment) };
        let ptr = unsafe { alloc(layout) };

        if ptr.is_null() {
            return Err(LightShipError::Backend(
                BackendError::OutOfMemory,
            ));
        }

        let id = BLOCK_ID.fetch_add(1, Ordering::Relaxed);
        Ok(MemoryBlock::new(
            id,
            NonNull::new(ptr).unwrap(),
            size,
            alignment,
            StorageLocation::Heap,
        ))
    }

    fn deallocate(&self, block: MemoryBlock) {
        unsafe {
            dealloc(
                block.ptr.as_ptr(),
                Layout::from_size_align_unchecked(block.size, block.alignment),
            );
        }
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

impl CpuBackend {
    fn execute_relu(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
        if inputs.is_empty() || outputs.is_empty() {
            return Err(LightShipError::InvalidParam("Missing input or output".into()));
        }

        let input = inputs[0];
        let output = &mut outputs[0];

        if input.data_type != DataType::F32 {
            return Err(LightShipError::Backend(
                BackendError::UnsupportedDataType(format!("{:?}", input.data_type)),
            ));
        }

        // Get input data as bytes and compute ReLU
        let input_bytes = input.data_as_bytes();
        let num_elements = input_bytes.len() / 4;

        // ReLU: max(x, 0)
        let mut output_bytes = Vec::with_capacity(input_bytes.len());
        for chunk in input_bytes.chunks_exact(4) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let result = if value > 0.0 { value } else { 0.0 };
            output_bytes.extend_from_slice(&result.to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU ReLU: input shape={:?}, elements={}", input.shape, num_elements);
        Ok(())
    }

    fn execute_add(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
        if inputs.len() < 2 || outputs.is_empty() {
            return Err(LightShipError::InvalidParam("Add requires 2 inputs and 1 output".into()));
        }

        let input_a = inputs[0];
        let input_b = inputs[1];
        let output = &mut outputs[0];

        if input_a.data_type != DataType::F32 || input_b.data_type != DataType::F32 {
            return Err(LightShipError::Backend(
                BackendError::UnsupportedDataType("Add requires F32 inputs".into()),
            ));
        }

        let input_bytes_a = input_a.data_as_bytes();
        let input_bytes_b = input_b.data_as_bytes();

        if input_bytes_a.len() != input_bytes_b.len() {
            return Err(LightShipError::InvalidParam("Add inputs must have same size".into()));
        }

        // Element-wise addition: a + b
        let mut output_bytes = Vec::with_capacity(input_bytes_a.len());
        for (chunk_a, chunk_b) in input_bytes_a.chunks_exact(4).zip(input_bytes_b.chunks_exact(4)) {
            let a = f32::from_le_bytes([chunk_a[0], chunk_a[1], chunk_a[2], chunk_a[3]]);
            let b = f32::from_le_bytes([chunk_b[0], chunk_b[1], chunk_b[2], chunk_b[3]]);
            output_bytes.extend_from_slice(&(a + b).to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU Add: executed {} elements", input_bytes_a.len() / 4);
        Ok(())
    }

    fn execute_mul(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
        if inputs.len() < 2 || outputs.is_empty() {
            return Err(LightShipError::InvalidParam("Mul requires 2 inputs and 1 output".into()));
        }

        let input_a = inputs[0];
        let input_b = inputs[1];
        let output = &mut outputs[0];

        if input_a.data_type != DataType::F32 || input_b.data_type != DataType::F32 {
            return Err(LightShipError::Backend(
                BackendError::UnsupportedDataType("Mul requires F32 inputs".into()),
            ));
        }

        let input_bytes_a = input_a.data_as_bytes();
        let input_bytes_b = input_b.data_as_bytes();

        if input_bytes_a.len() != input_bytes_b.len() {
            return Err(LightShipError::InvalidParam("Mul inputs must have same size".into()));
        }

        // Element-wise multiplication: a * b
        let mut output_bytes = Vec::with_capacity(input_bytes_a.len());
        for (chunk_a, chunk_b) in input_bytes_a.chunks_exact(4).zip(input_bytes_b.chunks_exact(4)) {
            let a = f32::from_le_bytes([chunk_a[0], chunk_a[1], chunk_a[2], chunk_a[3]]);
            let b = f32::from_le_bytes([chunk_b[0], chunk_b[1], chunk_b[2], chunk_b[3]]);
            output_bytes.extend_from_slice(&(a * b).to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU Mul: executed {} elements", input_bytes_a.len() / 4);
        Ok(())
    }

    fn execute_sigmoid(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
        if inputs.is_empty() || outputs.is_empty() {
            return Err(LightShipError::InvalidParam("Missing input or output".into()));
        }

        let input = inputs[0];
        let output = &mut outputs[0];

        if input.data_type != DataType::F32 {
            return Err(LightShipError::Backend(
                BackendError::UnsupportedDataType(format!("{:?}", input.data_type)),
            ));
        }

        let input_bytes = input.data_as_bytes();

        // sigmoid(x) = 1 / (1 + exp(-x))
        let mut output_bytes = Vec::with_capacity(input_bytes.len());
        for chunk in input_bytes.chunks_exact(4) {
            let x = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let result = 1.0 / (1.0 + (-x).exp());
            output_bytes.extend_from_slice(&result.to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU Sigmoid: executed {} elements", input_bytes.len() / 4);
        Ok(())
    }

    fn execute_tanh(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
        if inputs.is_empty() || outputs.is_empty() {
            return Err(LightShipError::InvalidParam("Missing input or output".into()));
        }

        let input = inputs[0];
        let output = &mut outputs[0];

        if input.data_type != DataType::F32 {
            return Err(LightShipError::Backend(
                BackendError::UnsupportedDataType(format!("{:?}", input.data_type)),
            ));
        }

        let input_bytes = input.data_as_bytes();

        // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        // Rust std library provides tanh directly
        let mut output_bytes = Vec::with_capacity(input_bytes.len());
        for chunk in input_bytes.chunks_exact(4) {
            let x = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let result = x.tanh();
            output_bytes.extend_from_slice(&result.to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU Tanh: executed {} elements", input_bytes.len() / 4);
        Ok(())
    }

    fn execute_relu6(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
        if inputs.is_empty() || outputs.is_empty() {
            return Err(LightShipError::InvalidParam("Missing input or output".into()));
        }

        let input = inputs[0];
        let output = &mut outputs[0];

        if input.data_type != DataType::F32 {
            return Err(LightShipError::Backend(
                BackendError::UnsupportedDataType(format!("{:?}", input.data_type)),
            ));
        }

        let input_bytes = input.data_as_bytes();

        // ReLU6: min(max(x, 0), 6)
        let mut output_bytes = Vec::with_capacity(input_bytes.len());
        for chunk in input_bytes.chunks_exact(4) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let result = value.max(0.0).min(6.0);
            output_bytes.extend_from_slice(&result.to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU ReLU6: executed {} elements", input_bytes.len() / 4);
        Ok(())
    }

    fn execute_maxpool2d(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
        if inputs.is_empty() || outputs.is_empty() {
            return Err(LightShipError::InvalidParam("Missing input or output".into()));
        }

        let input = inputs[0];
        let output = &mut outputs[0];

        if input.data_type != DataType::F32 {
            return Err(LightShipError::Backend(
                BackendError::UnsupportedDataType(format!("{:?}", input.data_type)),
            ));
        }

        // Assume NCHW format: [batch, channels, height, width]
        if input.shape.len() != 4 {
            return Err(LightShipError::InvalidParam("MaxPool2d requires 4D NCHW input".into()));
        }

        let [batch, channels, height, width] = &input.shape[..4] else {
            return Err(LightShipError::InvalidParam("Invalid input shape".into()));
        };

        // Assume kernel=2, stride=2 for simplicity
        let kernel = 2;
        let stride = 2;
        let out_h = (height + 2 * 0 - kernel) / stride + 1;
        let out_w = (width + 2 * 0 - kernel) / stride + 1;

        let input_bytes = input.data_as_bytes();

        // For each output element, find max in kernel window
        let mut output_bytes = Vec::with_capacity(batch * channels * out_h * out_w * 4);

        for b in 0..*batch {
            for c in 0..*channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;

                        for kh in 0..kernel {
                            for kw in 0..kernel {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;

                                if ih < *height && iw < *width {
                                    let idx = ((b * channels + c) * height + ih) * width + iw;
                                    let chunk_start = idx * 4;
                                    let val = f32::from_le_bytes([
                                        input_bytes[chunk_start],
                                        input_bytes[chunk_start + 1],
                                        input_bytes[chunk_start + 2],
                                        input_bytes[chunk_start + 3],
                                    ]);
                                    max_val = max_val.max(val);
                                }
                            }
                        }

                        output_bytes.extend_from_slice(&max_val.to_le_bytes());
                    }
                }
            }
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU MaxPool2d: output shape=[{}, {}, {}, {}]", batch, channels, out_h, out_w);
        Ok(())
    }

    fn execute_avgpool2d(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
        if inputs.is_empty() || outputs.is_empty() {
            return Err(LightShipError::InvalidParam("Missing input or output".into()));
        }

        let input = inputs[0];
        let output = &mut outputs[0];

        if input.data_type != DataType::F32 {
            return Err(LightShipError::Backend(
                BackendError::UnsupportedDataType(format!("{:?}", input.data_type)),
            ));
        }

        // Assume NCHW format: [batch, channels, height, width]
        if input.shape.len() != 4 {
            return Err(LightShipError::InvalidParam("AvgPool2d requires 4D NCHW input".into()));
        }

        let [batch, channels, height, width] = &input.shape[..4] else {
            return Err(LightShipError::InvalidParam("Invalid input shape".into()));
        };

        // Assume kernel=2, stride=2 for simplicity
        let kernel = 2;
        let stride = 2;
        let out_h = (height + 2 * 0 - kernel) / stride + 1;
        let out_w = (width + 2 * 0 - kernel) / stride + 1;

        let input_bytes = input.data_as_bytes();

        // For each output element, compute average in kernel window
        let mut output_bytes = Vec::with_capacity(batch * channels * out_h * out_w * 4);

        for b in 0..*batch {
            for c in 0..*channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        let mut count = 0.0f32;

                        for kh in 0..kernel {
                            for kw in 0..kernel {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;

                                if ih < *height && iw < *width {
                                    let idx = ((b * channels + c) * height + ih) * width + iw;
                                    let chunk_start = idx * 4;
                                    let val = f32::from_le_bytes([
                                        input_bytes[chunk_start],
                                        input_bytes[chunk_start + 1],
                                        input_bytes[chunk_start + 2],
                                        input_bytes[chunk_start + 3],
                                    ]);
                                    sum += val;
                                    count += 1.0;
                                }
                            }
                        }

                        let avg = if count > 0.0 { sum / count } else { 0.0 };
                        output_bytes.extend_from_slice(&avg.to_le_bytes());
                    }
                }
            }
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU AvgPool2d: output shape=[{}, {}, {}, {}]", batch, channels, out_h, out_w);
        Ok(())
    }

    fn execute_softmax(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
        if inputs.is_empty() || outputs.is_empty() {
            return Err(LightShipError::InvalidParam("Missing input or output".into()));
        }

        let input = inputs[0];
        let output = &mut outputs[0];

        if input.data_type != DataType::F32 {
            return Err(LightShipError::Backend(
                BackendError::UnsupportedDataType(format!("{:?}", input.data_type)),
            ));
        }

        let input_bytes = input.data_as_bytes();
        let num_elements = input_bytes.len() / 4;

        // Read input values
        let values: Vec<f32> = input_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Compute softmax: exp(x_i) / sum(exp(x_j))
        // Subtract max for numerical stability: softmax(x) = exp(x - max) / sum(exp(x - max))
        let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let exp_values: Vec<f32> = values
            .iter()
            .map(|&v| (v - max_val).exp())
            .collect();

        let sum_exp: f32 = exp_values.iter().sum();

        // Compute output
        let mut output_bytes = Vec::with_capacity(input_bytes.len());
        for exp_v in exp_values {
            let softmax = exp_v / sum_exp;
            output_bytes.extend_from_slice(&softmax.to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU Softmax: {} elements, sum_exp={}", num_elements, sum_exp);
        Ok(())
    }

    fn execute_div(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
        if inputs.len() < 2 || outputs.is_empty() {
            return Err(LightShipError::InvalidParam("Div requires 2 inputs and 1 output".into()));
        }

        let input_a = inputs[0];
        let input_b = inputs[1];
        let output = &mut outputs[0];

        if input_a.data_type != DataType::F32 || input_b.data_type != DataType::F32 {
            return Err(LightShipError::Backend(
                BackendError::UnsupportedDataType("Div requires F32 inputs".into()),
            ));
        }

        let input_bytes_a = input_a.data_as_bytes();
        let input_bytes_b = input_b.data_as_bytes();

        if input_bytes_a.len() != input_bytes_b.len() {
            return Err(LightShipError::InvalidParam("Div inputs must have same size".into()));
        }

        // Element-wise division: a / b
        let mut output_bytes = Vec::with_capacity(input_bytes_a.len());
        for (chunk_a, chunk_b) in input_bytes_a.chunks_exact(4).zip(input_bytes_b.chunks_exact(4)) {
            let a = f32::from_le_bytes([chunk_a[0], chunk_a[1], chunk_a[2], chunk_a[3]]);
            let b = f32::from_le_bytes([chunk_b[0], chunk_b[1], chunk_b[2], chunk_b[3]]);
            output_bytes.extend_from_slice(&(a / b).to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU Div: executed {} elements", input_bytes_a.len() / 4);
        Ok(())
    }

    fn execute_sub(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
        if inputs.len() < 2 || outputs.is_empty() {
            return Err(LightShipError::InvalidParam("Sub requires 2 inputs and 1 output".into()));
        }

        let input_a = inputs[0];
        let input_b = inputs[1];
        let output = &mut outputs[0];

        if input_a.data_type != DataType::F32 || input_b.data_type != DataType::F32 {
            return Err(LightShipError::Backend(
                BackendError::UnsupportedDataType("Sub requires F32 inputs".into()),
            ));
        }

        let input_bytes_a = input_a.data_as_bytes();
        let input_bytes_b = input_b.data_as_bytes();

        if input_bytes_a.len() != input_bytes_b.len() {
            return Err(LightShipError::InvalidParam("Sub inputs must have same size".into()));
        }

        // Element-wise subtraction: a - b
        let mut output_bytes = Vec::with_capacity(input_bytes_a.len());
        for (chunk_a, chunk_b) in input_bytes_a.chunks_exact(4).zip(input_bytes_b.chunks_exact(4)) {
            let a = f32::from_le_bytes([chunk_a[0], chunk_a[1], chunk_a[2], chunk_a[3]]);
            let b = f32::from_le_bytes([chunk_b[0], chunk_b[1], chunk_b[2], chunk_b[3]]);
            output_bytes.extend_from_slice(&(a - b).to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU Sub: executed {} elements", input_bytes_a.len() / 4);
        Ok(())
    }

    fn execute_matmul(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
        if inputs.len() < 2 || outputs.is_empty() {
            return Err(LightShipError::InvalidParam("MatMul requires 2 inputs and 1 output".into()));
        }

        let input_a = inputs[0];
        let input_b = inputs[1];
        let output = &mut outputs[0];

        if input_a.data_type != DataType::F32 || input_b.data_type != DataType::F32 {
            return Err(LightShipError::Backend(
                BackendError::UnsupportedDataType("MatMul requires F32 inputs".into()),
            ));
        }

        // Assume 2D matrices: [M, K] and [K, N] -> [M, N]
        if input_a.shape.len() != 2 || input_b.shape.len() != 2 {
            return Err(LightShipError::InvalidParam("MatMul requires 2D matrices".into()));
        }

        let [m, k_a] = &input_a.shape[..2] else {
            return Err(LightShipError::InvalidParam("Invalid matrix A shape".into()));
        };

        let [k_b, n] = &input_b.shape[..2] else {
            return Err(LightShipError::InvalidParam("Invalid matrix B shape".into()));
        };

        if k_a != k_b {
            return Err(LightShipError::InvalidParam(
                format!("Matrix dimension mismatch: A_cols={} != B_rows={}", k_a, k_b)
            ));
        }

        let k = *k_a;

        let input_bytes_a = input_a.data_as_bytes();
        let input_bytes_b = input_b.data_as_bytes();

        // Compute C = A * B
        // C[i,j] = sum_k A[i,k] * B[k,j]
        let mut output_bytes = Vec::with_capacity(m * n * 4);

        for i in 0..*m {
            for j in 0..*n {
                let mut sum = 0.0f32;

                for k_idx in 0..k {
                    // A[i,k_idx] at index i * k + k_idx
                    let a_idx = (i * k + k_idx) * 4;
                    let a_val = f32::from_le_bytes([
                        input_bytes_a[a_idx],
                        input_bytes_a[a_idx + 1],
                        input_bytes_a[a_idx + 2],
                        input_bytes_a[a_idx + 3],
                    ]);

                    // B[k_idx,j] at index k_idx * n + j
                    let b_idx = (k_idx * n + j) * 4;
                    let b_val = f32::from_le_bytes([
                        input_bytes_b[b_idx],
                        input_bytes_b[b_idx + 1],
                        input_bytes_b[b_idx + 2],
                        input_bytes_b[b_idx + 3],
                    ]);

                    sum += a_val * b_val;
                }

                output_bytes.extend_from_slice(&sum.to_le_bytes());
            }
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU MatMul: {}x{} @ {}x{} -> {}x{}", m, k, k, n, m, n);
        Ok(())
    }

    fn execute_reshape(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
        if inputs.is_empty() || outputs.is_empty() {
            return Err(LightShipError::InvalidParam("Missing input or output".into()));
        }

        let input = inputs[0];
        let output = &mut outputs[0];

        // Reshape just copies the data, shape is already set in output tensor
        let input_bytes = input.data_as_bytes();
        output.data = crate::ir::TensorData::Owned(input_bytes.to_vec());

        tracing::debug!("CPU Reshape: {:?} -> {:?}", input.shape, output.shape);
        Ok(())
    }

    fn execute_transpose(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
        if inputs.is_empty() || outputs.is_empty() {
            return Err(LightShipError::InvalidParam("Missing input or output".into()));
        }

        let input = inputs[0];
        let output = &mut outputs[0];

        if input.data_type != DataType::F32 {
            return Err(LightShipError::Backend(
                BackendError::UnsupportedDataType(format!("{:?}", input.data_type)),
            ));
        }

        // Assume 2D matrix transpose
        if input.shape.len() != 2 || output.shape.len() != 2 {
            return Err(LightShipError::InvalidParam("Transpose requires 2D input/output".into()));
        }

        let [rows, cols] = &input.shape[..2] else {
            return Err(LightShipError::InvalidParam("Invalid input shape".into()));
        };

        let [out_rows, out_cols] = &output.shape[..2] else {
            return Err(LightShipError::InvalidParam("Invalid output shape".into()));
        };

        if *out_rows != *cols || *out_cols != *rows {
            return Err(LightShipError::InvalidParam(
                format!("Transpose shape mismatch: input {}x{} -> output {}x{}", rows, cols, out_rows, out_cols)
            ));
        }

        let input_bytes = input.data_as_bytes();

        // Transpose: output[j,i] = input[i,j]
        let mut output_bytes = Vec::with_capacity(input_bytes.len());

        for j in 0..*cols {
            for i in 0..*rows {
                let idx = (i * cols + j) * 4;
                output_bytes.extend_from_slice(&[
                    input_bytes[idx],
                    input_bytes[idx + 1],
                    input_bytes[idx + 2],
                    input_bytes[idx + 3],
                ]);
            }
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU Transpose: {:?} -> {:?}", input.shape, output.shape);
        Ok(())
    }
}

impl Drop for CpuBackend {
    fn drop(&mut self) {}
}
