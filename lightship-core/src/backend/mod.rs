//! Backend abstraction for LightShip

pub mod backend;
pub mod capabilities;
pub mod cpu;
pub mod memory;

pub use backend::{
    Backend, BackendConfig, BackendManager, BackendSpecificData, CompiledOperator,
    CpuBackendConfig, GpuBackendConfig, ThreadAffinity,
};
pub use capabilities::{BackendCapabilities, BackendFeatureFlags, SimdFlags};
pub use cpu::CpuBackend;
pub use memory::MemoryBlock;
