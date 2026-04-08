//! Backend abstraction for LightShip

pub mod capabilities;
pub mod memory;

pub use capabilities::{BackendCapabilities, BackendFeatureFlags, SimdFlags};
pub use memory::MemoryBlock;
