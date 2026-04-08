//! Session API for LightShip

use crate::common::{BackendType, InferenceMode, Result};

/// Session configuration
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Preferred backend type
    pub preferred_backend: BackendType,
    /// Number of threads (0 = auto)
    pub num_threads: usize,
    /// Enable low memory mode
    pub low_memory_mode: bool,
    /// Maximum memory limit in bytes
    pub memory_limit: Option<usize>,
    /// Inference mode
    pub inference_mode: InferenceMode,
    /// Enable operator fusion
    pub enable_fusion: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            preferred_backend: BackendType::CPU,
            num_threads: 0,
            low_memory_mode: false,
            memory_limit: None,
            inference_mode: InferenceMode::Synchronous,
            enable_fusion: true,
        }
    }
}

/// Session handle for inference
#[derive(Debug)]
pub struct SessionHandle {
    // Placeholder - will be expanded in later phases
}

impl SessionHandle {
    /// Create a new session handle
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for SessionHandle {
    fn default() -> Self {
        Self::new()
    }
}
