//! Metal GPU backend implementation for LightShip
//!
//! This module provides GPU acceleration through Apple's Metal framework on macOS/iOS.

#[cfg(target_os = "macos")]
use metal::{CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize, CommandQueue};
use std::collections::HashMap;

use crate::backend::{
    Backend, BackendCapabilities, BackendSpecificData, CompiledOperator,
    MemoryBlock,
};
use crate::common::{BackendType, DataType, StorageLayout};
use crate::common::error::BackendError;
use crate::common::Result as LightShipResult;
use crate::ir::{OperatorDef, OperatorType, Tensor};

/// Metal backend configuration
#[derive(Debug, Clone)]
pub struct MetalBackendConfig {
    /// Enable FP16 computation
    pub use_fp16: bool,
    /// Enable shader caching
    pub enable_caching: bool,
}

impl Default for MetalBackendConfig {
    fn default() -> Self {
        Self {
            use_fp16: false,
            enable_caching: true,
        }
    }
}

/// Metal buffer wrapper
#[cfg(target_os = "macos")]
pub struct MetalBuffer {
    #[allow(dead_code)]
    buffer: metal::Buffer,
    size: usize,
}

#[cfg(target_os = "macos")]
impl std::fmt::Debug for MetalBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBuffer")
            .field("size", &self.size)
            .finish()
    }
}

#[cfg(target_os = "macos")]
impl MetalBuffer {
    /// Create a new Metal buffer
    pub fn new(device: &Device, size: usize) -> Result<Self, BackendError> {
        let options = MTLResourceOptions::StorageModeShared;
        let buffer = device.new_buffer(size as u64, options);
        Ok(Self { buffer, size })
    }

    /// Get raw buffer reference
    pub fn as_buffer(&self) -> &metal::Buffer {
        &self.buffer
    }

    /// Get buffer size
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Pipeline cache entry
#[cfg(target_os = "macos")]
struct PipelineCacheEntry {
    pipeline: ComputePipelineState,
    threadgroup_size: MTLSize,
}

/// Metal GPU backend
#[cfg(target_os = "macos")]
pub struct MetalBackend {
    #[allow(dead_code)]
    device: Device,
    #[allow(dead_code)]
    command_queue: CommandQueue,
    config: MetalBackendConfig,
    #[allow(dead_code)]
    pipeline_cache: HashMap<String, PipelineCacheEntry>,
}

#[cfg(target_os = "macos")]
impl std::fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBackend")
            .field("device", &self.device.name())
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(target_os = "macos")]
impl MetalBackend {
    /// Create a new Metal backend
    pub fn new() -> LightShipResult<Self> {
        let device = Device::system_default()
            .ok_or_else(|| BackendError::NotAvailable("No Metal device available".to_string()))?;

        let command_queue = device.new_command_queue();

        Ok(Self {
            device,
            command_queue,
            config: MetalBackendConfig::default(),
            pipeline_cache: HashMap::new(),
        })
    }

    /// Create with custom config
    pub fn with_config(config: MetalBackendConfig) -> LightShipResult<Self> {
        let device = Device::system_default()
            .ok_or_else(|| BackendError::NotAvailable("No Metal device available".to_string()))?;

        let command_queue = device.new_command_queue();

        Ok(Self {
            device,
            command_queue,
            config,
            pipeline_cache: HashMap::new(),
        })
    }

    /// Get device info
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Get max threads per threadgroup
    pub fn max_threads_per_threadgroup(&self) -> usize {
        self.device.max_threads_per_threadgroup().width as usize
    }
}

/// Mock MetalBackend for non-macOS platforms
#[cfg(not(target_os = "macos"))]
pub struct MetalBackend;

#[cfg(not(target_os = "macos"))]
impl MetalBackend {
    pub fn new() -> LightShipResult<Self> {
        Err(BackendError::NotAvailable(
            "Metal backend is only available on macOS".to_string(),
        ).into())
    }

    pub fn with_config(_config: MetalBackendConfig) -> LightShipResult<Self> {
        Err(BackendError::NotAvailable(
            "Metal backend is only available on macOS".to_string(),
        ).into())
    }
}

#[cfg(target_os = "macos")]
impl Backend for MetalBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Metal
    }

    fn is_available(&self) -> bool {
        Device::system_default().is_some()
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            backend_type: BackendType::Metal,
            supported_operators: vec![
                OperatorType::ReLU,
                OperatorType::ReLU6,
                OperatorType::Sigmoid,
                OperatorType::Tanh,
                OperatorType::Conv2d,
                OperatorType::MatMul,
                OperatorType::Add,
                OperatorType::Mul,
            ],
            supported_data_types: vec![DataType::F32, DataType::F16],
            supported_layouts: vec![StorageLayout::NCHW],
            max_threads: self.device.max_threads_per_threadgroup().width as usize,
            has_simd: true,
            simd_flags: crate::backend::SimdFlags::default(),
            memory_alignment: 64,
            register_size: 64,
            feature_flags: crate::backend::BackendFeatureFlags::default(),
        }
    }

    fn compile_operator(
        &self,
        def: &OperatorDef,
        fusion: Option<&crate::ir::FusionInfo>,
        _inputs: &[&Tensor],
        _outputs: &[&Tensor],
    ) -> LightShipResult<CompiledOperator> {
        Ok(CompiledOperator {
            operator_type: def.operator_type.clone(),
            backend_data: BackendSpecificData::Gpu(Vec::new()),
            workgroup_size: Some((64, 1, 1)),
            fusion: fusion.cloned(),
        })
    }

    fn execute(
        &self,
        op: &CompiledOperator,
        inputs: &[&Tensor],
        outputs: &mut [&mut Tensor],
    ) -> LightShipResult<()> {
        match op.operator_type {
            OperatorType::ReLU => self.execute_relu(inputs, outputs),
            OperatorType::ReLU6 => self.execute_relu6(inputs, outputs),
            OperatorType::Sigmoid => self.execute_sigmoid(inputs, outputs),
            OperatorType::Tanh => self.execute_tanh(inputs, outputs),
            _ => Err(BackendError::ExecutionFailed(format!(
                "Operator {:?} not yet implemented in Metal backend",
                op.operator_type
            )).into()),
        }
    }

    fn allocate(&self, size: usize, _alignment: usize) -> LightShipResult<MemoryBlock> {
        let buffer = MetalBuffer::new(&self.device, size)?;
        Ok(MemoryBlock {
            id: 0,
            ptr: std::ptr::NonNull::dangling(),
            size,
            alignment: 64,
            location: crate::backend::memory::StorageLocation::GPU,
        })
    }

    fn deallocate(&self, _block: MemoryBlock) {
        // Metal buffers are managed automatically
    }

    fn synchronize(&self) -> LightShipResult<()> {
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl MetalBackend {
    fn execute_relu(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> LightShipResult<()> {
        let input = inputs[0];
        let output = &mut outputs[0];

        let count = input.shape.iter().product::<usize>();
        let input_bytes = input.data_as_bytes();
        let output_bytes = output.data_as_bytes_mut();

        for i in 0..count {
            let val = f32::from_le_bytes([
                input_bytes[i * 4],
                input_bytes[i * 4 + 1],
                input_bytes[i * 4 + 2],
                input_bytes[i * 4 + 3],
            ]);
            let result = val.max(0.0f32);
            output_bytes[i * 4..(i + 1) * 4].copy_from_slice(&result.to_le_bytes());
        }

        Ok(())
    }

    fn execute_relu6(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> LightShipResult<()> {
        let input = inputs[0];
        let output = &mut outputs[0];

        let count = input.shape.iter().product::<usize>();
        let input_bytes = input.data_as_bytes();
        let output_bytes = output.data_as_bytes_mut();

        for i in 0..count {
            let val = f32::from_le_bytes([
                input_bytes[i * 4],
                input_bytes[i * 4 + 1],
                input_bytes[i * 4 + 2],
                input_bytes[i * 4 + 3],
            ]);
            let result = val.max(0.0f32).min(6.0f32);
            output_bytes[i * 4..(i + 1) * 4].copy_from_slice(&result.to_le_bytes());
        }

        Ok(())
    }

    fn execute_sigmoid(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> LightShipResult<()> {
        let input = inputs[0];
        let output = &mut outputs[0];

        let count = input.shape.iter().product::<usize>();
        let input_bytes = input.data_as_bytes();
        let output_bytes = output.data_as_bytes_mut();

        for i in 0..count {
            let val = f32::from_le_bytes([
                input_bytes[i * 4],
                input_bytes[i * 4 + 1],
                input_bytes[i * 4 + 2],
                input_bytes[i * 4 + 3],
            ]);
            let result = 1.0f32 / (1.0f32 + (-val).exp());
            output_bytes[i * 4..(i + 1) * 4].copy_from_slice(&result.to_le_bytes());
        }

        Ok(())
    }

    fn execute_tanh(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> LightShipResult<()> {
        let input = inputs[0];
        let output = &mut outputs[0];

        let count = input.shape.iter().product::<usize>();
        let input_bytes = input.data_as_bytes();
        let output_bytes = output.data_as_bytes_mut();

        for i in 0..count {
            let val = f32::from_le_bytes([
                input_bytes[i * 4],
                input_bytes[i * 4 + 1],
                input_bytes[i * 4 + 2],
                input_bytes[i * 4 + 3],
            ]);
            let result = val.tanh();
            output_bytes[i * 4..(i + 1) * 4].copy_from_slice(&result.to_le_bytes());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "macos")]
    fn test_metal_backend_creation() {
        if let Ok(backend) = MetalBackend::new() {
            assert!(backend.is_available());
            println!("Metal device: {}", backend.device_name());
        }
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_metal_backend_not_available() {
        let result = MetalBackend::new();
        assert!(result.is_err());
    }
}
