//! Backend abstraction for LightShip

pub mod backend;
pub mod capabilities;
pub mod cpu;
pub mod memory;

#[cfg(target_os = "macos")]
pub mod metal;

pub use backend::{
    Backend, BackendConfig, BackendManager, BackendSpecificData, CompiledOperator,
    CpuBackendConfig, GpuBackendConfig, ThreadAffinity,
};
pub use capabilities::{BackendCapabilities, BackendFeatureFlags, SimdFlags};
pub use cpu::CpuBackend;
pub use memory::MemoryBlock;

#[cfg(target_os = "macos")]
pub use metal::MetalBackend;
