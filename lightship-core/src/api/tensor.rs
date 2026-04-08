//! Tensor API for LightShip

use crate::common::Result;

/// Tensor handle for external API
#[derive(Debug)]
pub struct TensorHandle {
    // Placeholder - will be expanded when IR tensor is implemented
}

impl TensorHandle {
    /// Create a new tensor handle
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for TensorHandle {
    fn default() -> Self {
        Self::new()
    }
}
