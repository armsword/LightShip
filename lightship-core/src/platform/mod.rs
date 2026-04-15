//! Platform support module
//!
//! Provides cross-platform utilities for threading and CPU feature detection.

mod thread_pool;
mod cpu_info;
mod simd;

pub use thread_pool::{ThreadPool, ThreadPoolConfig};
pub use cpu_info::CpuInfo;
pub use simd::{
    SimdLevel, SimdOp, detect_simd_level,
    relu_simd, relu6_simd, relu_simd_bytes, add_simd, mul_simd, div_simd, sub_simd, gemm_simd, horizontal_sum,
    exp_simd, exp_softmax_simd, sub_scalar_simd, div_scalar_simd, mul_scalar_simd, tanh_simd,
};
