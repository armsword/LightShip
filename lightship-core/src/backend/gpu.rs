//! GPU backend trait extensions
//!
//! Provides GPU-specific backend traits and utilities.

use crate::common::{BackendType, Result};
use crate::ir::{OperatorDef, Tensor};

/// GPU backend trait - extension for GPU backends
pub trait GpuBackend: Send + Sync {
    /// Get the GPU device name
    fn device_name(&self) -> &str;

    /// Get available memory
    fn available_memory(&self) -> usize;

    /// Get used memory
    fn used_memory(&self) -> usize;

    /// Flush the GPU command buffer
    fn flush(&mut self) -> Result<()>;

    /// Synchronize with the GPU
    fn synchronize(&self) -> Result<()>;

    /// Upload data to GPU memory
    fn upload(&mut self, data: &[u8], dst: &mut [u8]) -> Result<()> {
        dst.copy_from_slice(data);
        Ok(())
    }

    /// Download data from GPU memory
    fn download(&mut self, src: &[u8], dst: &mut [u8]) -> Result<()> {
        dst.copy_from_slice(src);
        Ok(())
    }
}

/// Vulkan-specific backend extension
pub trait VulkanBackend: GpuBackend {
    /// Get Vulkan device handle
    fn vulkan_device(&self) -> &VulkanDevice;

    /// Create a Vulkan shader module
    fn create_shader_module(&self, spv: &[u32]) -> Result<VulkanShaderModule>;

    /// Create a Vulkan compute pipeline
    fn create_compute_pipeline(
        &self,
        shader: &VulkanShaderModule,
        workgroup_size: (u32, u32, u32),
    ) -> Result<VulkanPipeline>;
}

/// Vulkan device handle
#[derive(Debug, Clone)]
pub struct VulkanDevice {
    /// Device ID
    pub id: usize,
    /// Device name
    pub name: String,
    /// Vendor ID
    pub vendor_id: u32,
    /// Device type
    pub device_type: VulkanDeviceType,
    /// Supported features
    pub features: VulkanFeatures,
}

impl VulkanDevice {
    /// Create a new Vulkan device
    pub fn new(id: usize, name: String) -> Self {
        Self {
            id,
            name,
            vendor_id: 0,
            device_type: VulkanDeviceType::Unknown,
            features: VulkanFeatures::default(),
        }
    }
}

/// Vulkan device type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanDeviceType {
    /// Integrated GPU
    IntegratedGpu,
    /// Discrete GPU
    DiscreteGpu,
    /// Virtual GPU
    VirtualGpu,
    /// CPU
    Cpu,
    /// Unknown
    Unknown,
}

/// Vulkan device features
#[derive(Debug, Clone, Default)]
pub struct VulkanFeatures {
    /// Supports float16
    pub float16: bool,
    /// Supports float64
    pub float64: bool,
    /// Supports int64
    pub int64: bool,
    /// Supports storage buffers
    pub storage_buffer: bool,
    /// Supports storage images
    pub storage_image: bool,
}

/// Vulkan shader module
#[derive(Debug, Clone)]
pub struct VulkanShaderModule {
    /// Handle
    pub handle: u64,
}

/// Vulkan compute pipeline
#[derive(Debug, Clone)]
pub struct VulkanPipeline {
    /// Pipeline handle
    pub handle: u64,
    /// Workgroup size
    pub workgroup_size: (u32, u32, u32),
}

/// Metal-specific backend extension
pub trait MetalBackend: GpuBackend {
    /// Get Metal device
    fn metal_device(&self) -> &MetalDevice;

    /// Create a Metal compute pipeline
    fn create_compute_pipeline(
        &self,
        source: &str,
        function_name: &str,
    ) -> Result<MetalPipeline>;
}

/// Metal device
#[derive(Debug, Clone)]
pub struct MetalDevice {
    /// Device name
    pub name: String,
    /// Recommended working group size
    pub max_workgroup_size: usize,
}

impl MetalDevice {
    /// Create a new Metal device
    pub fn new(name: String) -> Self {
        Self {
            name,
            max_workgroup_size: 512,
        }
    }
}

/// Metal compute pipeline
#[derive(Debug, Clone)]
pub struct MetalPipeline {
    /// Pipeline handle
    pub handle: u64,
}

/// GPU memory allocation
#[derive(Debug, Clone)]
pub struct GpuMemoryBlock {
    /// Device ID
    pub device_id: usize,
    /// Memory address
    pub ptr: u64,
    /// Size in bytes
    pub size: usize,
    /// Memory type
    pub memory_type: GpuMemoryType,
}

/// GPU memory type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMemoryType {
    /// Device local memory (GPU)
    DeviceLocal,
    /// Host visible memory
    HostVisible,
    /// Host cached memory
    HostCached,
    /// Unified memory
    Unified,
}

/// NPU backend trait
pub trait NpuBackend: Send + Sync {
    /// Get the NPU device name
    fn device_name(&self) -> &str;

    /// Get the NPU vendor
    fn vendor(&self) -> &str;

    /// Get available NPU memory
    fn available_memory(&self) -> usize;

    /// Get NPU capabilities
    fn capabilities(&self) -> NpuCapabilities;
}

/// NPU capabilities
#[derive(Debug, Clone, Default)]
pub struct NpuCapabilities {
    /// Supports float16 inference
    pub float16: bool,
    /// Supports int8 inference
    pub int8: bool,
    /// Supports int4 inference
    pub int4: bool,
    /// Maximum tensor size
    pub max_tensor_size: usize,
}

/// GPU scheduling priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GpuSchedulingPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Real-time priority
    Realtime,
}

impl Default for GpuSchedulingPriority {
    fn default() -> Self {
        GpuSchedulingPriority::Normal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_device() {
        let device = VulkanDevice::new(0, "NVIDIA RTX 3080".to_string());
        assert_eq!(device.name, "NVIDIA RTX 3080");
    }

    #[test]
    fn test_vulkan_features() {
        let features = VulkanFeatures::default();
        assert!(!features.float16);
    }

    #[test]
    fn test_metal_device() {
        let device = MetalDevice::new("Apple M1".to_string());
        assert_eq!(device.name, "Apple M1");
        assert_eq!(device.max_workgroup_size, 512);
    }

    #[test]
    fn test_gpu_memory_type() {
        assert!(matches!(GpuMemoryType::DeviceLocal, GpuMemoryType::DeviceLocal));
    }

    #[test]
    fn test_gpu_scheduling_priority() {
        let low = GpuSchedulingPriority::Low;
        let high = GpuSchedulingPriority::High;
        assert!(low < high);
    }

    #[test]
    fn test_npu_capabilities() {
        let caps = NpuCapabilities::default();
        assert_eq!(caps.max_tensor_size, 0);
    }
}
