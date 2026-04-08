//! Platform support module
//!
//! Provides cross-platform utilities for threading and CPU feature detection.

mod thread_pool;
mod cpu_info;

pub use thread_pool::{ThreadPool, ThreadPoolConfig};
pub use cpu_info::CpuInfo;
