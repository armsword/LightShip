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
            OperatorType::Add => self.execute_add(inputs, outputs),
            OperatorType::Mul => self.execute_mul(inputs, outputs),
            OperatorType::Sigmoid => self.execute_sigmoid(inputs, outputs),
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
}

impl Drop for CpuBackend {
    fn drop(&mut self) {}
}
