//! CPU-specific optimized kernels
//!
//! This module provides CPU-optimized implementations of neural network operators.

pub mod thread_scheduler;
pub mod winograd_conv2d;

pub use thread_scheduler::*;
