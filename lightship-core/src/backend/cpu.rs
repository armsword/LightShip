//! CPU Backend implementation for LightShip

use std::fmt::Debug;
use crate::backend::{
    Backend, BackendCapabilities, BackendSpecificData, CompiledOperator, CpuBackendConfig,
    MemoryBlock, SimdFlags,
};
use crate::operator::{BatchNorm, Conv2d, Conv2dConfig, Pool2d, Pool2dConfig};
use crate::backend::memory::StorageLocation;
use crate::common::{BackendType, DataType, LightShipError, Result, StorageLayout};
use crate::common::error::BackendError;
use crate::ir::{OperatorDef, OperatorType, Tensor};
use crate::platform::{add_simd, detect_simd_level, div_scalar_simd, div_simd, exp_simd, gemm_simd, horizontal_sum, mul_simd, relu_simd, relu6_simd, sub_simd, tanh_simd, SimdLevel};
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

static BLOCK_ID: AtomicU64 = AtomicU64::new(0);

/// CPU Backend implementation
#[derive(Clone)]
pub struct CpuBackend {
    config: CpuBackendConfig,
    capabilities: BackendCapabilities,
    /// Scratch buffer for intermediate f32 data (reused across calls)
    scratch_f32: Vec<f32>,
    /// Scratch buffer for output bytes (reused across calls)
    scratch_bytes: Vec<u8>,
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
            // Pre-allocate scratch buffers for reuse (8MB each)
            scratch_f32: Vec::with_capacity(8 * 1024 * 1024 / 4),
            scratch_bytes: Vec::with_capacity(8 * 1024 * 1024),
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
            OperatorType::Conv2d => self.execute_conv2d(inputs, outputs),
            OperatorType::BatchNorm => self.execute_batchnorm(inputs, outputs),
            OperatorType::FullyConnected => self.execute_fullyconnected(inputs, outputs),
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

        let input_bytes = input.data_as_bytes();
        let num_elements = input_bytes.len() / 4;

        // Convert bytes to f32 slices for SIMD processing
        let input_f32: Vec<f32> = input_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut output_f32 = vec![0.0f32; num_elements];

        // Use SIMD acceleration if available
        let simd_level = detect_simd_level();
        relu_simd(&input_f32, &mut output_f32, simd_level);

        // Convert back to bytes
        let mut output_bytes = Vec::with_capacity(input_bytes.len());
        for &val in &output_f32 {
            output_bytes.extend_from_slice(&val.to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU ReLU: input shape={:?}, elements={}, simd={:?}", input.shape, num_elements, simd_level);
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

        let num_elements = input_bytes_a.len() / 4;

        // Convert bytes to f32 slices
        let a_f32: Vec<f32> = input_bytes_a
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let b_f32: Vec<f32> = input_bytes_b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut c_f32 = vec![0.0f32; num_elements];

        // Use SIMD acceleration
        let simd_level = detect_simd_level();
        add_simd(&a_f32, &b_f32, &mut c_f32, simd_level);

        // Convert back to bytes
        let mut output_bytes = Vec::with_capacity(input_bytes_a.len());
        for &val in &c_f32 {
            output_bytes.extend_from_slice(&val.to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU Add: executed {} elements, simd={:?}", num_elements, simd_level);
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

        let num_elements = input_bytes_a.len() / 4;

        // Convert bytes to f32 slices
        let a_f32: Vec<f32> = input_bytes_a
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let b_f32: Vec<f32> = input_bytes_b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut c_f32 = vec![0.0f32; num_elements];

        // Use SIMD acceleration
        let simd_level = detect_simd_level();
        mul_simd(&a_f32, &b_f32, &mut c_f32, simd_level);

        // Convert back to bytes
        let mut output_bytes = Vec::with_capacity(input_bytes_a.len());
        for &val in &c_f32 {
            output_bytes.extend_from_slice(&val.to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU Mul: executed {} elements, simd={:?}", num_elements, simd_level);
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
        let num_elements = input_bytes.len() / 4;

        // Convert to f32 slices
        let x_f32: Vec<f32> = input_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut neg_x = vec![0.0f32; num_elements];
        let mut exp_neg_x = vec![0.0f32; num_elements];
        let mut one_plus_exp = vec![0.0f32; num_elements];

        // neg_x = -x
        let simd_level = detect_simd_level();
        for i in 0..num_elements {
            neg_x[i] = -x_f32[i];
        }

        // exp_neg_x = exp(-x) using SIMD
        exp_simd(&neg_x, &mut exp_neg_x, simd_level);

        // sigmoid = 1 / (1 + exp(-x))
        let mut sigmoid = vec![0.0f32; num_elements];
        for i in 0..num_elements {
            one_plus_exp[i] = 1.0 + exp_neg_x[i];
            sigmoid[i] = 1.0 / one_plus_exp[i];
        }

        // Convert back to bytes
        let mut output_bytes = Vec::with_capacity(input_bytes.len());
        for &val in &sigmoid {
            output_bytes.extend_from_slice(&val.to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU Sigmoid: executed {} elements, simd={:?}", num_elements, simd_level);
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
        let num_elements = input_bytes.len() / 4;

        // Convert bytes to f32 slices
        let x_f32: Vec<f32> = input_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut output_f32 = vec![0.0f32; num_elements];

        // Use SIMD acceleration
        let simd_level = detect_simd_level();
        tanh_simd(&x_f32, &mut output_f32, simd_level);

        // Convert back to bytes
        let mut output_bytes = Vec::with_capacity(input_bytes.len());
        for &val in &output_f32 {
            output_bytes.extend_from_slice(&val.to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU Tanh: executed {} elements, simd={:?}", num_elements, simd_level);
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
        let num_elements = input_bytes.len() / 4;

        // Convert bytes to f32 slices for SIMD processing
        let input_f32: Vec<f32> = input_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut output_f32 = vec![0.0f32; num_elements];

        // Use SIMD acceleration if available
        let simd_level = detect_simd_level();
        relu6_simd(&input_f32, &mut output_f32, simd_level);

        // Convert back to bytes
        let mut output_bytes = Vec::with_capacity(input_bytes.len());
        for &val in &output_f32 {
            output_bytes.extend_from_slice(&val.to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU ReLU6: input shape={:?}, elements={}, simd={:?}", input.shape, num_elements, simd_level);
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

        // Use Pool2d SIMD implementation
        let config = Pool2dConfig {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            count_include_pad: false,
        };

        let pool = Pool2d::max_pool(config);
        let result = pool.max_pool2d_simd(input)?;

        output.data = result.data;

        tracing::debug!("CPU MaxPool2d: output shape={:?}", output.shape);
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

        // Use Pool2d implementation
        let config = Pool2dConfig {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            count_include_pad: false,
        };

        let pool = Pool2d::avg_pool(config);
        let result = pool.avg_pool2d(input)?;

        output.data = result.data;

        tracing::debug!("CPU AvgPool2d: output shape={:?}", output.shape);
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
        let mut values: Vec<f32> = input_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let simd_level = detect_simd_level();

        // Find max value for numerical stability (scalar reduction)
        let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Subtract max: values = values - max_val (scalar subtraction)
        for v in &mut values {
            *v = *v - max_val;
        }

        // Compute exp(values - max) using SIMD
        let mut exp_values = vec![0.0f32; num_elements];
        exp_simd(&values, &mut exp_values, simd_level);

        // Sum all exp values using SIMD horizontal sum
        let sum_exp = horizontal_sum(&exp_values, simd_level);

        // Divide each exp value by sum using SIMD scalar division
        let mut softmax_values = vec![0.0f32; num_elements];
        div_scalar_simd(&exp_values, &mut softmax_values, sum_exp, simd_level);

        // Convert to bytes
        let mut output_bytes = Vec::with_capacity(input_bytes.len());
        for &val in &softmax_values {
            output_bytes.extend_from_slice(&val.to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU Softmax: {} elements, sum_exp={}, simd={:?}", num_elements, sum_exp, simd_level);
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
        let num_elements = input_bytes_a.len() / 4;

        // Convert bytes to f32 slices
        let a_f32: Vec<f32> = input_bytes_a
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let b_f32: Vec<f32> = input_bytes_b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut c_f32 = vec![0.0f32; num_elements];

        // Use SIMD acceleration
        let simd_level = detect_simd_level();
        div_simd(&a_f32, &b_f32, &mut c_f32, simd_level);

        // Convert back to bytes
        let mut output_bytes = Vec::with_capacity(input_bytes_a.len());
        for &val in &c_f32 {
            output_bytes.extend_from_slice(&val.to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU Div: executed {} elements, simd={:?}", num_elements, simd_level);
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
        let num_elements = input_bytes_a.len() / 4;

        // Convert bytes to f32 slices
        let a_f32: Vec<f32> = input_bytes_a
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let b_f32: Vec<f32> = input_bytes_b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut c_f32 = vec![0.0f32; num_elements];

        // Use SIMD acceleration
        let simd_level = detect_simd_level();
        sub_simd(&a_f32, &b_f32, &mut c_f32, simd_level);

        // Convert back to bytes
        let mut output_bytes = Vec::with_capacity(input_bytes_a.len());
        for &val in &c_f32 {
            output_bytes.extend_from_slice(&val.to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU Sub: executed {} elements, simd={:?}", num_elements, simd_level);
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

        // Convert bytes to f32 slices
        let a_f32: Vec<f32> = input_bytes_a
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let b_f32: Vec<f32> = input_bytes_b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut c_f32 = vec![0.0f32; m * n];

        // Use SIMD acceleration for GEMM
        let simd_level = detect_simd_level();
        gemm_simd(&a_f32, &b_f32, &mut c_f32, *m, *n, k, simd_level);

        // Convert result back to bytes
        let mut output_bytes = Vec::with_capacity(m * n * 4);
        for &val in &c_f32 {
            output_bytes.extend_from_slice(&val.to_le_bytes());
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU MatMul: {}x{} @ {}x{} -> {}x{}, simd={:?}", m, k, k, n, m, n, simd_level);
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

    fn execute_conv2d(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
        if inputs.len() < 2 || outputs.is_empty() {
            return Err(LightShipError::InvalidParam("Conv2d requires 2 inputs (input, filter) and 1 output".into()));
        }

        let input = inputs[0];
        let filter = inputs[1];
        let output = &mut outputs[0];

        if input.data_type != DataType::F32 || filter.data_type != DataType::F32 {
            return Err(LightShipError::Backend(
                BackendError::UnsupportedDataType("Conv2d requires F32 inputs".into()),
            ));
        }

        // Create Conv2d with default config
        // Note: In production, config should come from OperatorDef.attributes
        let config = Conv2dConfig {
            out_channels: filter.shape[0],
            kernel_h: filter.shape[2],
            kernel_w: filter.shape[3],
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
        };

        let conv = Conv2d::new(config);
        let result = conv.forward(input, filter)?;

        // Copy output data
        output.data = result.data;

        tracing::debug!("CPU Conv2d: input shape={:?}, filter shape={:?}", input.shape, filter.shape);
        Ok(())
    }

    fn execute_batchnorm(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
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

        // Determine number of channels from shape
        let num_features = if input.shape.len() >= 2 {
            input.shape[1]
        } else {
            return Err(LightShipError::InvalidParam("BatchNorm requires at least 2D input".into()));
        };

        // Create BatchNorm with default parameters (gamma=1, beta=0)
        let bn = BatchNorm::new(num_features);
        let result = bn.forward_inference(input)?;

        // Copy output data
        output.data = result.data;

        tracing::debug!("CPU BatchNorm: input shape={:?}, channels={}", input.shape, num_features);
        Ok(())
    }

    fn execute_fullyconnected(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
        if inputs.len() < 2 || outputs.is_empty() {
            return Err(LightShipError::InvalidParam("FullyConnected requires 2 inputs (input, weight) and 1 output".into()));
        }

        let input = inputs[0];
        let weight = inputs[1];
        let output = &mut outputs[0];

        if input.data_type != DataType::F32 || weight.data_type != DataType::F32 {
            return Err(LightShipError::Backend(
                BackendError::UnsupportedDataType("FullyConnected requires F32 inputs".into()),
            ));
        }

        // Input: [batch, in_features]
        // Weight: [out_features, in_features]
        // Output: [batch, out_features]
        //
        // FC: output = input @ weight.T
        // where weight.T has shape [in_features, out_features]

        if input.shape.len() != 2 || weight.shape.len() != 2 {
            return Err(LightShipError::InvalidParam("FullyConnected requires 2D input and weight".into()));
        }

        let [batch, in_features] = &input.shape[..2] else {
            return Err(LightShipError::InvalidParam("Invalid input shape".into()));
        };

        let [out_features, weight_in_features] = &weight.shape[..2] else {
            return Err(LightShipError::InvalidParam("Invalid weight shape".into()));
        };

        if *weight_in_features != *in_features {
            return Err(LightShipError::InvalidParam(
                format!("Weight in_features {} != input in_features {}", weight_in_features, in_features)
            ));
        }

        let input_bytes = input.data_as_bytes();
        let weight_bytes = weight.data_as_bytes();

        // FC: output[b, out_f] = sum_j (input[b, j] * weight[out_f, j])
        // Note: weight is stored as [out_features, in_features]
        let mut output_bytes = Vec::with_capacity(batch * out_features * 4);

        for b in 0..*batch {
            for out_f in 0..*out_features {
                let mut sum = 0.0f32;

                for in_f in 0..*in_features {
                    // input[b, in_f]
                    let input_idx = (b * in_features + in_f) * 4;
                    let input_val = f32::from_le_bytes([
                        input_bytes[input_idx],
                        input_bytes[input_idx + 1],
                        input_bytes[input_idx + 2],
                        input_bytes[input_idx + 3],
                    ]);

                    // weight[out_f, in_f]
                    let weight_idx = (out_f * in_features + in_f) * 4;
                    let weight_val = f32::from_le_bytes([
                        weight_bytes[weight_idx],
                        weight_bytes[weight_idx + 1],
                        weight_bytes[weight_idx + 2],
                        weight_bytes[weight_idx + 3],
                    ]);

                    sum += input_val * weight_val;
                }

                output_bytes.extend_from_slice(&sum.to_le_bytes());
            }
        }

        output.data = crate::ir::TensorData::Owned(output_bytes);

        tracing::debug!("CPU FullyConnected: {}x{} @ {}x{} -> {}x{}", batch, in_features, out_features, in_features, batch, out_features);
        Ok(())
    }
}

impl Drop for CpuBackend {
    fn drop(&mut self) {}
}
