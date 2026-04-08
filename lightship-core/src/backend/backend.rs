//! Backend trait definition for LightShip

use std::fmt::Debug;
use crate::common::{BackendType, LightShipError, Result};
use crate::ir::{OperatorDef, OperatorType, Tensor};

/// Backend error code
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendErrorCode {
    /// Operator not supported by this backend
    UnsupportedOperator,
    /// Data type not supported
    UnsupportedDataType,
    /// Compilation failed
    CompilationFailed,
    /// Execution failed
    ExecutionFailed,
    /// Out of memory
    OutOfMemory,
    /// Invalid parameter
    InvalidParameter,
}

/// Backend configuration enum
#[derive(Debug, Clone)]
pub enum BackendConfig {
    /// CPU backend configuration
    Cpu(CpuBackendConfig),
    /// GPU backend configuration
    Gpu(GpuBackendConfig),
    /// Metal backend (Apple devices)
    #[cfg(target_os = "macos")]
    Metal,
}

/// CPU backend configuration
#[derive(Debug, Clone)]
pub struct CpuBackendConfig {
    /// Number of threads (0 = auto)
    pub num_threads: usize,
    /// Enable SIMD
    pub use_simd: bool,
    /// Thread affinity
    pub thread_affinity: ThreadAffinity,
}

impl Default for CpuBackendConfig {
    fn default() -> Self {
        Self {
            num_threads: 0,
            use_simd: true,
            thread_affinity: ThreadAffinity::default(),
        }
    }
}

/// Thread affinity policy
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThreadAffinity {
    /// Disable thread affinity
    Disabled,
    /// Prefer small cores
    SmallCore,
    /// Prefer big cores
    BigCore,
    /// Custom core list
    Custom(Vec<usize>),
}

impl Default for ThreadAffinity {
    fn default() -> Self {
        ThreadAffinity::Disabled
    }
}

/// GPU backend configuration
#[derive(Debug, Clone)]
pub struct GpuBackendConfig {
    /// Device ID
    pub device_id: usize,
    /// Enable FP16
    pub enable_fp16: bool,
}

impl Default for GpuBackendConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            enable_fp16: false,
        }
    }
}

/// Compiled operator
#[derive(Debug, Clone)]
pub struct CompiledOperator {
    /// Operator type
    pub operator_type: OperatorType,
    /// Backend-specific data
    pub backend_data: BackendSpecificData,
    /// Workgroup size for GPU
    pub workgroup_size: Option<(u32, u32, u32)>,
}

/// Backend-specific compiled data
#[derive(Debug, Clone)]
pub enum BackendSpecificData {
    /// CPU specific data
    Cpu(Vec<u8>),
    /// GPU specific data
    Gpu(Vec<u8>),
    /// NPU specific data
    Npu(Vec<u8>),
}

impl Default for BackendSpecificData {
    fn default() -> Self {
        BackendSpecificData::Cpu(Vec::new())
    }
}

/// Backend trait - all backends must implement this
pub trait Backend: Send + Sync {
    /// Returns the backend type
    fn backend_type(&self) -> BackendType;

    /// Check if this backend is available on the current platform
    fn is_available(&self) -> bool;

    /// Get backend capabilities
    fn capabilities(&self) -> crate::backend::BackendCapabilities;

    /// Compile an operator for this backend
    fn compile_operator(
        &self,
        def: &OperatorDef,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
    ) -> Result<CompiledOperator>;

    /// Execute a compiled operator
    fn execute(
        &self,
        op: &CompiledOperator,
        inputs: &[&Tensor],
        outputs: &mut [&Tensor],
    ) -> Result<()>;

    /// Allocate memory on this backend
    fn allocate(&self, size: usize, alignment: usize) -> Result<crate::backend::MemoryBlock>;

    /// Deallocate memory
    fn deallocate(&self, block: crate::backend::MemoryBlock);

    /// Synchronize the backend (wait for all operations to complete)
    fn synchronize(&self) -> Result<()>;
}

/// Backend manager for creating and managing backends
pub struct BackendManager {
    backends: Vec<Box<dyn Backend>>,
}

impl Debug for BackendManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendManager")
            .field("backends", &self.backends.len())
            .finish()
    }
}

impl BackendManager {
    /// Create a new backend manager
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    /// Register a backend
    pub fn register<B: Backend + 'static>(&mut self, backend: B) {
        self.backends.push(Box::new(backend));
    }

    /// Get a backend by type
    pub fn get(&self, backend_type: BackendType) -> Option<&dyn Backend> {
        self.backends
            .iter()
            .find(|b| b.backend_type() == backend_type)
            .map(|b| b.as_ref())
    }

    /// Get the first available backend
    pub fn get_available(&self) -> Option<&dyn Backend> {
        self.backends.iter().find(|b| b.is_available()).map(|b| b.as_ref())
    }

    /// List all registered backend types
    pub fn list_backends(&self) -> Vec<BackendType> {
        self.backends.iter().map(|b| b.backend_type()).collect()
    }

    /// List all available backend types
    pub fn list_available(&self) -> Vec<BackendType> {
        self.backends
            .iter()
            .filter(|b| b.is_available())
            .map(|b| b.backend_type())
            .collect()
    }
}

impl Default for BackendManager {
    fn default() -> Self {
        Self::new()
    }
}
