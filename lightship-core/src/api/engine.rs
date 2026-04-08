//! Engine API for LightShip

use crate::common::{BackendType, LightShipError, Result};

/// Engine configuration
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Preferred backend type
    pub preferred_backend: BackendType,
    /// Number of threads (0 = auto)
    pub num_threads: usize,
    /// Enable debug mode
    pub debug_mode: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            preferred_backend: BackendType::CPU,
            num_threads: 0,
            debug_mode: false,
        }
    }
}

/// LightShip Engine
#[derive(Debug)]
pub struct Engine {
    config: EngineConfig,
}

impl Engine {
    /// Create a new engine instance
    pub fn new(config: EngineConfig) -> Result<Self> {
        Ok(Self { config })
    }

    /// Get the engine version
    pub fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    /// Get available backends
    pub fn available_backends(&self) -> Vec<BackendType> {
        vec![BackendType::CPU]
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new(EngineConfig::default()).unwrap()
    }
}
