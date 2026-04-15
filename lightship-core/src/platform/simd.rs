//! SIMD optimized operator kernels
//!
//! Provides vectorized implementations of common neural network operators
//! using SSE, AVX, AVX2, AVX-512, and NEON instructions.

use crate::ir::Tensor;

/// SIMD operation trait for runtime dispatch
pub trait SimdOp: Send + Sync {
    /// Execute the operation
    fn execute(&self, input: &[f32], output: &mut [f32]);
}

/// SIMD level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// No SIMD support
    None = 0,
    /// SSE (128-bit)
    Sse = 1,
    /// SSE2 (128-bit)
    Sse2 = 2,
    /// SSE3 (128-bit)
    Sse3 = 3,
    /// Supplemental SSE3 (128-bit)
    Ssse3 = 4,
    /// SSE4.1 (128-bit)
    Sse4_1 = 5,
    /// SSE4.2 (128-bit)
    Sse4_2 = 6,
    /// AVX (256-bit)
    Avx = 7,
    /// AVX2 (256-bit)
    Avx2 = 8,
    /// AVX-512 (512-bit)
    Avx512 = 9,
    /// NEON (128-bit ARM)
    Neon = 10,
    /// NEON with FP16 (ARM)
    Neonfp16 = 11,
}

impl SimdLevel {
    /// Get vector width in bytes
    pub fn vector_width(&self) -> usize {
        match self {
            SimdLevel::None => 0,
            SimdLevel::Sse | SimdLevel::Sse2 | SimdLevel::Sse3 |
            SimdLevel::Ssse3 | SimdLevel::Sse4_1 | SimdLevel::Sse4_2 |
            SimdLevel::Avx | SimdLevel::Neon | SimdLevel::Neonfp16 => 16,
            SimdLevel::Avx2 => 32,
            SimdLevel::Avx512 => 64,
        }
    }

    /// Get the number of f32 elements per vector
    pub fn f32_per_vector(&self) -> usize {
        self.vector_width() / 4
    }
}

// ============================================================================
// Platform-agnostic function that dispatches to the right implementation
// ============================================================================

/// Get the optimal SIMD level for this platform
pub fn detect_simd_level() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            SimdLevel::Avx512
        } else if is_x86_feature_detected!("avx2") {
            SimdLevel::Avx2
        } else if is_x86_feature_detected!("avx") {
            SimdLevel::Avx
        } else if is_x86_feature_detected!("sse4.2") {
            SimdLevel::Sse4_2
        } else if is_x86_feature_detected!("sse4.1") {
            SimdLevel::Sse4_1
        } else if is_x86_feature_detected!("ssse3") {
            SimdLevel::Ssse3
        } else if is_x86_feature_detected!("sse3") {
            SimdLevel::Sse3
        } else if is_x86_feature_detected!("sse2") {
            SimdLevel::Sse2
        } else {
            SimdLevel::None
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on ARM64
        SimdLevel::Neon
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        SimdLevel::None
    }
}

/// ReLU: max(0, x) - dispatches to best available implementation
pub fn relu_simd(input: &[f32], output: &mut [f32], level: SimdLevel) {
    let len = input.len().min(output.len());
    #[cfg(target_arch = "x86_64")]
    {
        match level {
            SimdLevel::Avx512 => { unsafe { relu_avx512(input, output, len) }; return; }
            SimdLevel::Avx2 => { unsafe { relu_avx2(input, output, len) }; return; }
            SimdLevel::Avx => { unsafe { relu_avx(input, output, len) }; return; }
            SimdLevel::Sse2 | SimdLevel::Neon => { unsafe { relu_sse(input, output, len) }; return; }
            _ => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if matches!(level, SimdLevel::Neon | SimdLevel::Neonfp16) {
            unsafe { relu_neon(input, output, len) };
            return;
        }
    }
    relu_scalar(input, output, len);
}

/// ReLU on bytes directly - avoids f32 conversion overhead
pub fn relu_simd_bytes(input: &[u8], output: &mut [u8], level: SimdLevel) {
    let len = input.len() / 4;
    #[cfg(target_arch = "aarch64")]
    {
        if matches!(level, SimdLevel::Neon | SimdLevel::Neonfp16) {
            unsafe { relu_bytes_neon(input, output, len) };
            return;
        }
    }
    // Fallback: convert and use f32 version
    let input_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const f32, len)
    };
    let output_f32: &mut [f32] = unsafe {
        std::slice::from_raw_parts_mut(output.as_ptr() as *mut f32, len)
    };
    relu_simd(input_f32, output_f32, level);
}

/// ReLU6: clamp(x, 0, 6) - dispatches to best available implementation
pub fn relu6_simd(input: &[f32], output: &mut [f32], level: SimdLevel) {
    let len = input.len().min(output.len());
    #[cfg(target_arch = "x86_64")]
    {
        match level {
            SimdLevel::Avx512 => { unsafe { relu6_avx512(input, output, len) }; return; }
            SimdLevel::Avx2 => { unsafe { relu6_avx2(input, output, len) }; return; }
            SimdLevel::Avx => { unsafe { relu6_avx(input, output, len) }; return; }
            SimdLevel::Sse2 | SimdLevel::Neon => { unsafe { relu6_sse(input, output, len) }; return; }
            _ => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if matches!(level, SimdLevel::Neon | SimdLevel::Neonfp16) {
            unsafe { relu6_neon(input, output, len) };
            return;
        }
    }
    relu6_scalar(input, output, len);
}

/// Element-wise addition: c = a + b
pub fn add_simd(a: &[f32], b: &[f32], c: &mut [f32], level: SimdLevel) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    let len = a.len();
    #[cfg(target_arch = "x86_64")]
    {
        match level {
            SimdLevel::Avx512 => { unsafe { add_avx512(a, b, c, len) }; return; }
            SimdLevel::Avx2 => { unsafe { add_avx2(a, b, c, len) }; return; }
            SimdLevel::Avx => { unsafe { add_avx(a, b, c, len) }; return; }
            SimdLevel::Sse2 | SimdLevel::Neon => { unsafe { add_sse(a, b, c, len) }; return; }
            _ => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if matches!(level, SimdLevel::Neon | SimdLevel::Neonfp16) {
            unsafe { add_neon(a, b, c, len) };
            return;
        }
    }
    add_scalar(a, b, c, len);
}

/// Element-wise multiplication: c = a * b
pub fn mul_simd(a: &[f32], b: &[f32], c: &mut [f32], level: SimdLevel) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    let len = a.len();
    #[cfg(target_arch = "x86_64")]
    {
        match level {
            SimdLevel::Avx512 => { unsafe { mul_avx512(a, b, c, len) }; return; }
            SimdLevel::Avx2 => { unsafe { mul_avx2(a, b, c, len) }; return; }
            SimdLevel::Avx => { unsafe { mul_avx(a, b, c, len) }; return; }
            SimdLevel::Sse2 | SimdLevel::Neon => { unsafe { mul_sse(a, b, c, len) }; return; }
            _ => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if matches!(level, SimdLevel::Neon | SimdLevel::Neonfp16) {
            unsafe { mul_neon(a, b, c, len) };
            return;
        }
    }
    mul_scalar(a, b, c, len);
}

/// Element-wise subtraction: c[i] = a[i] - b[i]
pub fn sub_simd(a: &[f32], b: &[f32], c: &mut [f32], level: SimdLevel) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    let len = a.len();
    #[cfg(target_arch = "x86_64")]
    {
        match level {
            SimdLevel::Avx512 => { unsafe { sub_avx512(a, b, c, len) }; return; }
            SimdLevel::Avx2 => { unsafe { sub_avx2(a, b, c, len) }; return; }
            SimdLevel::Avx => { unsafe { sub_avx(a, b, c, len) }; return; }
            SimdLevel::Sse2 | SimdLevel::Neon => { unsafe { sub_sse(a, b, c, len) }; return; }
            _ => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if matches!(level, SimdLevel::Neon | SimdLevel::Neonfp16) {
            unsafe { sub_neon(a, b, c, len) };
            return;
        }
    }
    sub_scalar(a, b, c, len);
}

/// Element-wise division: c[i] = a[i] / b[i]
pub fn div_simd(a: &[f32], b: &[f32], c: &mut [f32], level: SimdLevel) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    let len = a.len();
    #[cfg(target_arch = "x86_64")]
    {
        match level {
            SimdLevel::Avx512 => { unsafe { div_avx512(a, b, c, len) }; return; }
            SimdLevel::Avx2 => { unsafe { div_avx2(a, b, c, len) }; return; }
            SimdLevel::Avx => { unsafe { div_avx(a, b, c, len) }; return; }
            SimdLevel::Sse2 | SimdLevel::Neon => { unsafe { div_sse(a, b, c, len) }; return; }
            _ => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if matches!(level, SimdLevel::Neon | SimdLevel::Neonfp16) {
            unsafe { div_neon(a, b, c, len) };
            return;
        }
    }
    div_scalar(a, b, c, len);
}

/// Compute exp(x) for each element (softmax use)
pub fn exp_simd(input: &[f32], output: &mut [f32], level: SimdLevel) {
    let len = input.len().min(output.len());
    #[cfg(target_arch = "x86_64")]
    {
        match level {
            SimdLevel::Avx512 => { unsafe { exp_avx512(input, output, len) }; return; }
            SimdLevel::Avx2 => { unsafe { exp_avx2(input, output, len) }; return; }
            SimdLevel::Avx | SimdLevel::Sse2 | SimdLevel::Neon => { unsafe { exp_sse(input, output, len) }; return; }
            _ => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if matches!(level, SimdLevel::Neon | SimdLevel::Neonfp16) {
            unsafe { exp_neon(input, output, len) };
            return;
        }
    }
    exp_scalar(input, output, len);
}

// ============================================================================
// Softmax-specific fast exp using lookup table (x in [-10, 0])
// ============================================================================

const SOFTMAX_EXP_TABLE_SIZE: usize = 1024;
const SOFTMAX_EXP_MIN: f32 = -10.0;
const SOFTMAX_EXP_MAX: f32 = 0.0;
const SOFTMAX_EXP_STEP: f32 = (SOFTMAX_EXP_MAX - SOFTMAX_EXP_MIN) / (SOFTMAX_EXP_TABLE_SIZE - 1) as f32;

fn build_softmax_exp_table() -> [f32; SOFTMAX_EXP_TABLE_SIZE] {
    let mut table = [0.0f32; SOFTMAX_EXP_TABLE_SIZE];
    for i in 0..SOFTMAX_EXP_TABLE_SIZE {
        let x = SOFTMAX_EXP_MIN + i as f32 * SOFTMAX_EXP_STEP;
        table[i] = x.exp();
    }
    table
}

#[inline]
fn softmax_exp_approx(x: f32) -> f32 {
    let x = x.clamp(SOFTMAX_EXP_MIN, SOFTMAX_EXP_MAX);
    let normalized = (x - SOFTMAX_EXP_MIN) / SOFTMAX_EXP_STEP;
    let index = normalized as usize;
    let fraction = normalized - index as f32;
    let table = SOFTMAX_EXP_TABLE.get().unwrap();
    let lo = table[index.min(SOFTMAX_EXP_TABLE_SIZE - 1)];
    let hi = table[(index + 1).min(SOFTMAX_EXP_TABLE_SIZE - 1)];
    lo + (hi - lo) * fraction
}

use std::sync::OnceLock;
static SOFTMAX_EXP_TABLE: OnceLock<[f32; SOFTMAX_EXP_TABLE_SIZE]> = OnceLock::new();

/// Fast exp for softmax - uses lookup table for x in [-10, 0]
pub fn exp_softmax_simd(input: &[f32], output: &mut [f32], level: SimdLevel) {
    let len = input.len().min(output.len());
    let _ = SOFTMAX_EXP_TABLE.get_or_init(build_softmax_exp_table);
    #[cfg(target_arch = "aarch64")]
    {
        if matches!(level, SimdLevel::Neon | SimdLevel::Neonfp16) {
            unsafe { exp_softmax_neon(input, output, len) };
            return;
        }
    }
    // Fallback: use scalar approximation
    for i in 0..len {
        output[i] = softmax_exp_approx(input[i]);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn exp_softmax_neon(input: &[f32], output: &mut [f32], len: usize) {
    use std::arch::aarch64::*;
    let table = SOFTMAX_EXP_TABLE.get().unwrap();
    let step = SOFTMAX_EXP_STEP;
    let min_val = SOFTMAX_EXP_MIN;
    let mut i = 0;
    while i + 4 <= len {
        let mut vals = [0.0f32; 4];
        for j in 0..4 {
            let x = input[i + j].clamp(min_val, 0.0);
            let normalized = (x - min_val) / step;
            let index = normalized as usize;
            let fraction = normalized - index as f32;
            let lo = table[index.min(SOFTMAX_EXP_TABLE_SIZE - 1)];
            let hi = table[(index + 1).min(SOFTMAX_EXP_TABLE_SIZE - 1)];
            vals[j] = lo + (hi - lo) * fraction;
        }
        let result = vld1q_f32(vals.as_ptr());
        vst1q_f32(&mut output[i], result);
        i += 4;
    }
    while i < len {
        output[i] = softmax_exp_approx(input[i]);
        i += 1;
    }
}

/// Subtract scalar from each element: output = input - scalar
pub fn sub_scalar_simd(input: &[f32], output: &mut [f32], scalar: f32, level: SimdLevel) {
    let len = input.len().min(output.len());
    #[cfg(target_arch = "x86_64")]
    {
        match level {
            SimdLevel::Avx512 => { unsafe { sub_scalar_avx512(input, output, scalar, len) }; return; }
            SimdLevel::Avx2 => { unsafe { sub_scalar_avx2(input, output, scalar, len) }; return; }
            SimdLevel::Avx | SimdLevel::Sse2 | SimdLevel::Neon => { unsafe { sub_scalar_sse(input, output, scalar, len) }; return; }
            _ => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if matches!(level, SimdLevel::Neon | SimdLevel::Neonfp16) {
            unsafe { sub_scalar_neon(input, output, scalar, len) };
            return;
        }
    }
    sub_scalar_scalar(input, output, scalar, len);
}

/// Divide by scalar: output = input / scalar
pub fn div_scalar_simd(input: &[f32], output: &mut [f32], scalar: f32, level: SimdLevel) {
    let len = input.len().min(output.len());
    #[cfg(target_arch = "x86_64")]
    {
        match level {
            SimdLevel::Avx512 => { unsafe { div_scalar_avx512(input, output, scalar, len) }; return; }
            SimdLevel::Avx2 => { unsafe { div_scalar_avx2(input, output, scalar, len) }; return; }
            SimdLevel::Avx | SimdLevel::Sse2 | SimdLevel::Neon => { unsafe { div_scalar_sse(input, output, scalar, len) }; return; }
            _ => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if matches!(level, SimdLevel::Neon | SimdLevel::Neonfp16) {
            unsafe { div_scalar_neon(input, output, scalar, len) };
            return;
        }
    }
    div_scalar_scalar(input, output, scalar, len);
}

/// Multiply by scalar: output = input * scalar
pub fn mul_scalar_simd(input: &[f32], output: &mut [f32], scalar: f32, level: SimdLevel) {
    let len = input.len().min(output.len());
    #[cfg(target_arch = "x86_64")]
    {
        match level {
            SimdLevel::Avx512 => { unsafe { mul_scalar_avx512(input, output, scalar, len) }; return; }
            SimdLevel::Avx2 => { unsafe { mul_scalar_avx2(input, output, scalar, len) }; return; }
            SimdLevel::Avx | SimdLevel::Sse2 | SimdLevel::Neon => { unsafe { mul_scalar_sse(input, output, scalar, len) }; return; }
            _ => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if matches!(level, SimdLevel::Neon | SimdLevel::Neonfp16) {
            unsafe { mul_scalar_neon(input, output, scalar, len) };
            return;
        }
    }
    mul_scalar_scalar(input, output, scalar, len);
}

/// GEMM: C = A * B + C
/// - A: [M x K], B: [K x N], C: [M x N]
pub fn gemm_simd(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize, level: SimdLevel) {
    #[cfg(target_arch = "x86_64")]
    {
        match level {
            SimdLevel::Avx512 => { unsafe { gemm_avx512(a, b, c, m, n, k) }; return; }
            SimdLevel::Avx2 => { unsafe { gemm_avx2(a, b, c, m, n, k) }; return; }
            SimdLevel::Avx | SimdLevel::Sse2 | SimdLevel::Neon => { unsafe { gemm_sse(a, b, c, m, n, k) }; return; }
            _ => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if matches!(level, SimdLevel::Neon | SimdLevel::Neonfp16) {
            unsafe { gemm_neon(a, b, c, m, n, k) };
            return;
        }
    }
    gemm_scalar(a, b, c, m, n, k);
}

/// Compute horizontal sum of array
pub fn horizontal_sum(arr: &[f32], level: SimdLevel) -> f32 {
    let len = arr.len();
    #[cfg(target_arch = "x86_64")]
    {
        match level {
            SimdLevel::Avx512 => { return unsafe { horizontal_sum_avx512(arr, len) }; }
            SimdLevel::Avx2 => { return unsafe { horizontal_sum_avx2(arr, len) }; }
            SimdLevel::Avx | SimdLevel::Sse2 | SimdLevel::Neon => { return unsafe { horizontal_sum_sse(arr, len) }; }
            _ => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if matches!(level, SimdLevel::Neon | SimdLevel::Neonfp16) {
            return unsafe { horizontal_sum_neon(arr, len) };
        }
    }
    horizontal_sum_scalar(arr, len)
}

/// Tanh: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
pub fn tanh_simd(input: &[f32], output: &mut [f32], level: SimdLevel) {
    let len = input.len().min(output.len());
    if len == 0 {
        return;
    }

    // Allocate temporary buffers for exp(x) and exp(-x)
    let mut exp_x = vec![0.0f32; len];
    let mut exp_neg_x = vec![0.0f32; len];

    // Compute exp(x)
    exp_simd(input, &mut exp_x, level);

    // Compute exp(-x) = exp(negate(input))
    let mut neg_input = vec![0.0f32; len];
    for i in 0..len {
        neg_input[i] = -input[i];
    }
    exp_simd(&neg_input, &mut exp_neg_x, level);

    // Compute tanh = (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
    for i in 0..len {
        let exp_pos = exp_x[i];
        let exp_neg = exp_neg_x[i];
        output[i] = (exp_pos - exp_neg) / (exp_pos + exp_neg);
    }
}

// ============================================================================
// Scalar implementations (fallback)
// ============================================================================

fn relu_scalar(input: &[f32], output: &mut [f32], len: usize) {
    for i in 0..len {
        output[i] = if input[i] > 0.0 { input[i] } else { 0.0 };
    }
}

fn relu6_scalar(input: &[f32], output: &mut [f32], len: usize) {
    for i in 0..len {
        output[i] = input[i].clamp(0.0, 6.0);
    }
}

fn add_scalar(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
    for i in 0..len {
        c[i] = a[i] + b[i];
    }
}

fn mul_scalar(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
    for i in 0..len {
        c[i] = a[i] * b[i];
    }
}

fn div_scalar(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
    for i in 0..len {
        c[i] = a[i] / b[i];
    }
}

fn sub_scalar(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
    for i in 0..len {
        c[i] = a[i] - b[i];
    }
}

fn gemm_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for m_idx in 0..m {
        for n_idx in 0..n {
            let mut sum = 0.0f32;
            for k_idx in 0..k {
                sum += a[m_idx * k + k_idx] * b[k_idx * n + n_idx];
            }
            c[m_idx * n + n_idx] += sum;
        }
    }
}

fn horizontal_sum_scalar(arr: &[f32], _len: usize) -> f32 {
    arr.iter().sum()
}

fn exp_scalar(input: &[f32], output: &mut [f32], len: usize) {
    for i in 0..len {
        output[i] = input[i].exp();
    }
}

fn sub_scalar_scalar(input: &[f32], output: &mut [f32], scalar: f32, len: usize) {
    for i in 0..len {
        output[i] = input[i] - scalar;
    }
}

fn div_scalar_scalar(input: &[f32], output: &mut [f32], scalar: f32, len: usize) {
    for i in 0..len {
        output[i] = input[i] / scalar;
    }
}

fn mul_scalar_scalar(input: &[f32], output: &mut [f32], scalar: f32, len: usize) {
    for i in 0..len {
        output[i] = input[i] * scalar;
    }
}

// ============================================================================
// x86_64 SIMD implementations
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod x86_64_impls {
    use super::*;

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn relu_avx512(input: &[f32], output: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let zero = _mm512_setzero_ps();
        let mut i = 0;
        while i + 16 <= len {
            let data = _mm512_loadu_ps(&input[i]);
            let result = _mm512_max_ps(zero, data);
            _mm512_storeu_ps(&mut output[i], result);
            i += 16;
        }
        while i < len {
            output[i] = if input[i] > 0.0 { input[i] } else { 0.0 };
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn relu_avx2(input: &[f32], output: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let zero = _mm256_setzero256();
        let mut i = 0;
        while i + 8 <= len {
            let data = _mm256_loadu_ps(&input[i]);
            let result = _mm256_max_ps(zero, data);
            _mm256_storeu_ps(&mut output[i], result);
            i += 8;
        }
        while i < len {
            output[i] = if input[i] > 0.0 { input[i] } else { 0.0 };
            i += 1;
        }
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn relu_avx(input: &[f32], output: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let zero = _mm256_setzero256();
        let mut i = 0;
        while i + 8 <= len {
            let data = _mm256_loadu_ps(&input[i]);
            let result = _mm256_max_ps(zero, data);
            _mm256_storeu_ps(&mut output[i], result);
            i += 8;
        }
        while i < len {
            output[i] = if input[i] > 0.0 { input[i] } else { 0.0 };
            i += 1;
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn relu_sse(input: &[f32], output: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let zero = _mm_setzero_ps();
        let mut i = 0;
        while i + 4 <= len {
            let data = _mm_loadu_ps(&input[i]);
            let result = _mm_max_ps(zero, data);
            _mm_storeu_ps(&mut output[i], result);
            i += 4;
        }
        while i < len {
            output[i] = if input[i] > 0.0 { input[i] } else { 0.0 };
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn relu6_avx512(input: &[f32], output: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let zero = _mm512_setzero_ps();
        let six = _mm512_set1_ps(6.0);
        let mut i = 0;
        while i + 16 <= len {
            let data = _mm512_loadu_ps(&input[i]);
            let clamped = _mm512_min_ps(_mm512_max_ps(zero, data), six);
            _mm512_storeu_ps(&mut output[i], clamped);
            i += 16;
        }
        while i < len {
            output[i] = input[i].clamp(0.0, 6.0);
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn relu6_avx2(input: &[f32], output: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let zero = _mm256_setzero256();
        let six = _mm256_set1_ps(6.0);
        let mut i = 0;
        while i + 8 <= len {
            let data = _mm256_loadu_ps(&input[i]);
            let clamped = _mm256_min_ps(_mm256_max_ps(zero, data), six);
            _mm256_storeu_ps(&mut output[i], clamped);
            i += 8;
        }
        while i < len {
            output[i] = input[i].clamp(0.0, 6.0);
            i += 1;
        }
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn relu6_avx(input: &[f32], output: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let zero = _mm256_setzero256();
        let six = _mm256_set1_ps(6.0);
        let mut i = 0;
        while i + 8 <= len {
            let data = _mm256_loadu_ps(&input[i]);
            let clamped = _mm256_min_ps(_mm256_max_ps(zero, data), six);
            _mm256_storeu_ps(&mut output[i], clamped);
            i += 8;
        }
        while i < len {
            output[i] = input[i].clamp(0.0, 6.0);
            i += 1;
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn relu6_sse(input: &[f32], output: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let zero = _mm_setzero_ps();
        let six = _mm_set1_ps(6.0);
        let mut i = 0;
        while i + 4 <= len {
            let data = _mm_loadu_ps(&input[i]);
            let clamped = _mm_min_ps(_mm_max_ps(zero, data), six);
            _mm_storeu_ps(&mut output[i], clamped);
            i += 4;
        }
        while i < len {
            output[i] = input[i].clamp(0.0, 6.0);
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn add_avx512(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 16 <= len {
            let va = _mm512_loadu_ps(&a[i]);
            let vb = _mm512_loadu_ps(&b[i]);
            let vc = _mm512_add_ps(va, vb);
            _mm512_storeu_ps(&mut c[i], vc);
            i += 16;
        }
        while i < len {
            c[i] = a[i] + b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn sub_avx512(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 16 <= len {
            let va = _mm512_loadu_ps(&a[i]);
            let vb = _mm512_loadu_ps(&b[i]);
            let vc = _mm512_sub_ps(va, vb);
            _mm512_storeu_ps(&mut c[i], vc);
            i += 16;
        }
        while i < len {
            c[i] = a[i] - b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn add_avx2(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 8 <= len {
            let va = _mm256_loadu_ps(&a[i]);
            let vb = _mm256_loadu_ps(&b[i]);
            let vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&mut c[i], vc);
            i += 8;
        }
        while i < len {
            c[i] = a[i] + b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn sub_avx2(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 8 <= len {
            let va = _mm256_loadu_ps(&a[i]);
            let vb = _mm256_loadu_ps(&b[i]);
            let vc = _mm256_sub_ps(va, vb);
            _mm256_storeu_ps(&mut c[i], vc);
            i += 8;
        }
        while i < len {
            c[i] = a[i] - b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn add_avx(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 8 <= len {
            let va = _mm256_loadu_ps(&a[i]);
            let vb = _mm256_loadu_ps(&b[i]);
            let vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&mut c[i], vc);
            i += 8;
        }
        while i < len {
            c[i] = a[i] + b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn sub_avx(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 8 <= len {
            let va = _mm256_loadu_ps(&a[i]);
            let vb = _mm256_loadu_ps(&b[i]);
            let vc = _mm256_sub_ps(va, vb);
            _mm256_storeu_ps(&mut c[i], vc);
            i += 8;
        }
        while i < len {
            c[i] = a[i] - b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn add_sse(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 4 <= len {
            let va = _mm_loadu_ps(&a[i]);
            let vb = _mm_loadu_ps(&b[i]);
            let vc = _mm_add_ps(va, vb);
            _mm_storeu_ps(&mut c[i], vc);
            i += 4;
        }
        while i < len {
            c[i] = a[i] + b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn sub_sse(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 4 <= len {
            let va = _mm_loadu_ps(&a[i]);
            let vb = _mm_loadu_ps(&b[i]);
            let vc = _mm_sub_ps(va, vb);
            _mm_storeu_ps(&mut c[i], vc);
            i += 4;
        }
        while i < len {
            c[i] = a[i] - b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn mul_avx512(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 16 <= len {
            let va = _mm512_loadu_ps(&a[i]);
            let vb = _mm512_loadu_ps(&b[i]);
            let vc = _mm512_mul_ps(va, vb);
            _mm512_storeu_ps(&mut c[i], vc);
            i += 16;
        }
        while i < len {
            c[i] = a[i] * b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn div_avx512(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 16 <= len {
            let va = _mm512_loadu_ps(&a[i]);
            let vb = _mm512_loadu_ps(&b[i]);
            let vc = _mm512_div_ps(va, vb);
            _mm512_storeu_ps(&mut c[i], vc);
            i += 16;
        }
        while i < len {
            c[i] = a[i] / b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn mul_avx2(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 8 <= len {
            let va = _mm256_loadu_ps(&a[i]);
            let vb = _mm256_loadu_ps(&b[i]);
            let vc = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(&mut c[i], vc);
            i += 8;
        }
        while i < len {
            c[i] = a[i] * b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn div_avx2(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 8 <= len {
            let va = _mm256_loadu_ps(&a[i]);
            let vb = _mm256_loadu_ps(&b[i]);
            let vc = _mm256_div_ps(va, vb);
            _mm256_storeu_ps(&mut c[i], vc);
            i += 8;
        }
        while i < len {
            c[i] = a[i] / b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn mul_avx(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 8 <= len {
            let va = _mm256_loadu_ps(&a[i]);
            let vb = _mm256_loadu_ps(&b[i]);
            let vc = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(&mut c[i], vc);
            i += 8;
        }
        while i < len {
            c[i] = a[i] * b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn div_avx(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 8 <= len {
            let va = _mm256_loadu_ps(&a[i]);
            let vb = _mm256_loadu_ps(&b[i]);
            let vc = _mm256_div_ps(va, vb);
            _mm256_storeu_ps(&mut c[i], vc);
            i += 8;
        }
        while i < len {
            c[i] = a[i] / b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn mul_sse(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 4 <= len {
            let va = _mm_loadu_ps(&a[i]);
            let vb = _mm_loadu_ps(&b[i]);
            let vc = _mm_mul_ps(va, vb);
            _mm_storeu_ps(&mut c[i], vc);
            i += 4;
        }
        while i < len {
            c[i] = a[i] * b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn div_sse(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 4 <= len {
            let va = _mm_loadu_ps(&a[i]);
            let vb = _mm_loadu_ps(&b[i]);
            let vc = _mm_div_ps(va, vb);
            _mm_storeu_ps(&mut c[i], vc);
            i += 4;
        }
        while i < len {
            c[i] = a[i] / b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn horizontal_sum_avx512(arr: &[f32], len: usize) -> f32 {
        use std::arch::x86_64::*;
        let mut sum = _mm512_setzero_ps();
        let mut i = 0;
        while i + 16 <= len {
            sum = _mm512_add_ps(sum, _mm512_loadu_ps(&arr[i]));
            i += 16;
        }
        let mut result = _mm512_reduce_add_ps(sum);
        while i < len {
            result += arr[i];
            i += 1;
        }
        result
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn horizontal_sum_avx2(arr: &[f32], len: usize) -> f32 {
        use std::arch::x86_64::*;
        let mut sum = _mm256_setzero256();
        let mut i = 0;
        while i + 8 <= len {
            sum = _mm256_add_ps(sum, _mm256_loadu_ps(&arr[i]));
            i += 8;
        }
        // Horizontal add within 256-bit vector
        let temp = _mm256_hadd_ps(sum, sum);
        let temp = _mm256_hadd_ps(temp, temp);
        let result = _mm256_cvtss_f32(_mm256_hadd_ps(temp, temp));
        while i < len {
            result += arr[i];
            i += 1;
        }
        result
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn horizontal_sum_sse(arr: &[f32], len: usize) -> f32 {
        use std::arch::x86_64::*;
        let mut sum = _mm_setzero_ps();
        let mut i = 0;
        while i + 4 <= len {
            sum = _mm_add_ps(sum, _mm_loadu_ps(&arr[i]));
            i += 4;
        }
        // Horizontal add
        let temp = _mm_hadd_ps(sum, sum);
        let result = _mm_cvtss_f32(_mm_hadd_ps(temp, temp));
        while i < len {
            result += arr[i];
            i += 1;
        }
        result
    }

    // GEMM implementations with cache-blocking for better performance
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn gemm_avx512(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        use std::arch::x86_64::*;
        let nr = 16;

        // For small matrices, use the simple version to avoid overhead
        if m * n < 1024 {
            gemm_avx512_simple(a, b, c, m, n, k);
            return;
        }

        // Cache-blocking: block over m and k to improve cache hits
        let mc = 64.min(m);
        let kc = 128.min(k);

        for m_block in (0..m).step_by(mc) {
            let m_end = (m_block + mc).min(m);
            for k_block in (0..k).step_by(kc) {
                let k_end = (k_block + kc).min(k);

                for i in m_block..m_end {
                    for j in (0..n).step_by(nr) {
                        let j_end = (j + nr).min(n);
                        let mut sum = [_mm512_setzero_ps(); 16];

                        for p in k_block..k_end {
                            let a_val = _mm512_set1_ps(a[i * k + p]);
                            for jj in 0..16 {
                                let b_idx = p * n + j + jj;
                                let b_val = _mm512_set1_ps(*b.get_unchecked(b_idx));
                                sum[jj] = _mm512_fmadd_ps(a_val, b_val, sum[jj]);
                            }
                        }

                        for jj in 0..(j_end - j) {
                            let result = _mm512_reduce_add_ps(sum[jj]);
                            let c_idx = i * n + j + jj;
                            *c.get_unchecked_mut(c_idx) += result;
                        }
                    }
                }
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn gemm_avx512_simple(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        use std::arch::x86_64::*;
        let nr = 16;
        for m_idx in 0..m {
            for n_idx in (0..n).step_by(nr) {
                let n_end = (n_idx + nr).min(n);
                let mut sum = [_mm512_setzero_ps(); 16];
                let mut k_idx = 0;
                while k_idx + 16 <= k {
                    let a_val = _mm512_set1_ps(a[m_idx * k + k_idx]);
                    for j in 0..16 {
                        let b_val = _mm512_set1_ps(*b.get_unchecked((k_idx + j) * n + n_idx));
                        sum[j] = _mm512_fmadd_ps(a_val, b_val, sum[j]);
                    }
                    k_idx += 16;
                }
                for j in 0..(n_end - n_idx) {
                    let result = _mm512_reduce_add_ps(sum[j]);
                    *c.get_unchecked_mut(m_idx * n + n_idx + j) += result;
                }
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn gemm_avx2(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        use std::arch::x86_64::*;
        let nr = 8;

        if m * n < 1024 {
            gemm_avx2_simple(a, b, c, m, n, k);
            return;
        }

        let mc = 64.min(m);
        let kc = 128.min(k);

        for m_block in (0..m).step_by(mc) {
            let m_end = (m_block + mc).min(m);
            for k_block in (0..k).step_by(kc) {
                let k_end = (k_block + kc).min(k);

                for i in m_block..m_end {
                    for j in (0..n).step_by(nr) {
                        let j_end = (j + nr).min(n);
                        let mut sum = [_mm256_setzero256(); 8];

                        for p in k_block..k_end {
                            let a_val = _mm256_set1_ps(a[i * k + p]);
                            for jj in 0..8 {
                                let b_idx = p * n + j + jj;
                                let b_val = _mm256_set1_ps(*b.get_unchecked(b_idx));
                                sum[jj] = _mm256_fmadd_ps(a_val, b_val, sum[jj]);
                            }
                        }

                        for jj in 0..(j_end - j) {
                            let temp = _mm256_hadd_ps(sum[jj], sum[jj]);
                            let temp = _mm256_hadd_ps(temp, temp);
                            let result = _mm256_cvtss_f32(_mm256_hadd_ps(temp, temp));
                            let c_idx = i * n + j + jj;
                            *c.get_unchecked_mut(c_idx) += result;
                        }
                    }
                }
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn gemm_avx2_simple(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        use std::arch::x86_64::*;
        let nr = 8;
        for m_idx in 0..m {
            for n_idx in (0..n).step_by(nr) {
                let n_end = (n_idx + nr).min(n);
                let mut sum = [_mm256_setzero256(); 8];
                let mut k_idx = 0;
                while k_idx + 8 <= k {
                    let a_val = _mm256_set1_ps(a[m_idx * k + k_idx]);
                    for j in 0..8 {
                        let b_val = _mm256_set1_ps(*b.get_unchecked((k_idx + j) * n + n_idx));
                        sum[j] = _mm256_fmadd_ps(a_val, b_val, sum[j]);
                    }
                    k_idx += 8;
                }
                for j in 0..(n_end - n_idx) {
                    let temp = _mm256_hadd_ps(sum[j], sum[j]);
                    let temp = _mm256_hadd_ps(temp, temp);
                    let result = _mm256_cvtss_f32(_mm256_hadd_ps(temp, temp));
                    *c.get_unchecked_mut(m_idx * n + n_idx + j) += result;
                }
            }
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn gemm_sse(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        use std::arch::x86_64::*;
        let nr = 4;

        if m * n < 1024 {
            gemm_sse_simple(a, b, c, m, n, k);
            return;
        }

        let mc = 64.min(m);
        let kc = 128.min(k);

        for m_block in (0..m).step_by(mc) {
            let m_end = (m_block + mc).min(m);
            for k_block in (0..k).step_by(kc) {
                let k_end = (k_block + kc).min(k);

                for i in m_block..m_end {
                    for j in (0..n).step_by(nr) {
                        let j_end = (j + nr).min(n);
                        let mut sum = [_mm_setzero_ps(); 4];

                        for p in k_block..k_end {
                            let a_val = _mm_set1_ps(a[i * k + p]);
                            for jj in 0..4 {
                                let b_idx = p * n + j + jj;
                                let b_val = _mm_set1_ps(*b.get_unchecked(b_idx));
                                sum[jj] = _mm_add_ps(_mm_mul_ps(a_val, b_val), sum[jj]);
                            }
                        }

                        for jj in 0..(j_end - j) {
                            let temp = _mm_hadd_ps(sum[jj], sum[jj]);
                            let result = _mm_cvtss_f32(_mm_hadd_ps(temp, temp));
                            let c_idx = i * n + j + jj;
                            *c.get_unchecked_mut(c_idx) += result;
                        }
                    }
                }
            }
        }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn gemm_sse_simple(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        use std::arch::x86_64::*;
        let nr = 4;
        for m_idx in 0..m {
            for n_idx in (0..n).step_by(nr) {
                let n_end = (n_idx + nr).min(n);
                let mut sum = [_mm_setzero_ps(); 4];
                let mut k_idx = 0;
                while k_idx + 4 <= k {
                    let a_val = _mm_set1_ps(a[m_idx * k + k_idx]);
                    for j in 0..4 {
                        let b_val = _mm_set1_ps(*b.get_unchecked((k_idx + j) * n + n_idx));
                        sum[j] = _mm_add_ps(_mm_mul_ps(a_val, b_val), sum[j]);
                    }
                    k_idx += 4;
                }
                for j in 0..(n_end - n_idx) {
                    let temp = _mm_hadd_ps(sum[j], sum[j]);
                    let result = _mm_cvtss_f32(_mm_hadd_ps(temp, temp));
                    *c.get_unchecked_mut(m_idx * n + n_idx + j) += result;
                }
            }
        }
    }

    // exp implementation using polynomial approximation
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn exp_avx2(input: &[f32], output: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 8 <= len {
            let x0 = input[i].exp();
            let x1 = input[i + 1].exp();
            let x2 = input[i + 2].exp();
            let x3 = input[i + 3].exp();
            let x4 = input[i + 4].exp();
            let x5 = input[i + 5].exp();
            let x6 = input[i + 6].exp();
            let x7 = input[i + 7].exp();
            let vec = _mm256_set_ps(x7, x6, x5, x4, x3, x2, x1, x0);
            _mm256_storeu_ps(&mut output[i], vec);
            i += 8;
        }
        while i < len {
            output[i] = input[i].exp();
            i += 1;
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn exp_sse(input: &[f32], output: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 4 <= len {
            let x0 = input[i].exp();
            let x1 = input[i + 1].exp();
            let x2 = input[i + 2].exp();
            let x3 = input[i + 3].exp();
            let vec = _mm_set_ps(x3, x2, x1, x0);
            _mm_storeu_ps(&mut output[i], vec);
            i += 4;
        }
        while i < len {
            output[i] = input[i].exp();
            i += 1;
        }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn exp_poly_sse(x: __m128) -> __m128 {
        use std::arch::x86_64::*;
        // 8th-order polynomial for better accuracy
        let one = _mm_set1_ps(1.0);
        let x2 = _mm_mul_ps(x, x);
        let x3 = _mm_mul_ps(x2, x);
        let x4 = _mm_mul_ps(x3, x);
        let x5 = _mm_mul_ps(x4, x);
        let x6 = _mm_mul_ps(x5, x);
        let x7 = _mm_mul_ps(x6, x);
        let x8 = _mm_mul_ps(x7, x);
        let t1 = _mm_mul_ps(_mm_set1_ps(0.5), x2);
        let t2 = _mm_mul_ps(_mm_set1_ps(1.0 / 6.0), x3);
        let t3 = _mm_mul_ps(_mm_set1_ps(1.0 / 24.0), x4);
        let t4 = _mm_mul_ps(_mm_set1_ps(1.0 / 120.0), x5);
        let t5 = _mm_mul_ps(_mm_set1_ps(1.0 / 720.0), x6);
        let t6 = _mm_mul_ps(_mm_set1_ps(1.0 / 5040.0), x7);
        let t7 = _mm_mul_ps(_mm_set1_ps(1.0 / 40320.0), x8);
        let s1 = _mm_add_ps(t1, t2);
        let s2 = _mm_add_ps(t3, t4);
        let s3 = _mm_add_ps(t5, t6);
        let s4 = _mm_add_ps(s1, s2);
        let s5 = _mm_add_ps(s3, s4);
        let s6 = _mm_add_ps(s5, t7);
        let result = _mm_add_ps(_mm_add_ps(one, x), s6);
        result
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn exp_avx512(input: &[f32], output: &mut [f32], len: usize) {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 16 <= len {
            // Use scalar exp for accuracy (SSE/AVX polynomial approximation not accurate enough)
            let x0 = input[i].exp();
            let x1 = input[i + 1].exp();
            let x2 = input[i + 2].exp();
            let x3 = input[i + 3].exp();
            let x4 = input[i + 4].exp();
            let x5 = input[i + 5].exp();
            let x6 = input[i + 6].exp();
            let x7 = input[i + 7].exp();
            let x8 = input[i + 8].exp();
            let x9 = input[i + 9].exp();
            let x10 = input[i + 10].exp();
            let x11 = input[i + 11].exp();
            let x12 = input[i + 12].exp();
            let x13 = input[i + 13].exp();
            let x14 = input[i + 14].exp();
            let x15 = input[i + 15].exp();
            let vec = _mm512_set_ps(x15, x14, x13, x12, x11, x10, x9, x8,
                                    x7, x6, x5, x4, x3, x2, x1, x0);
            _mm512_storeu_ps(&mut output[i], vec);
            i += 16;
        }
        while i < len {
            output[i] = input[i].exp();
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn exp_poly_avx512(x: __m512) -> __m512 {
        use std::arch::x86_64::*;
        // 8th-order polynomial for better accuracy
        let one = _mm512_set1_ps(1.0);
        let x2 = _mm512_mul_ps(x, x);
        let x3 = _mm512_mul_ps(x2, x);
        let x4 = _mm512_mul_ps(x3, x);
        let x5 = _mm512_mul_ps(x4, x);
        let x6 = _mm512_mul_ps(x5, x);
        let x7 = _mm512_mul_ps(x6, x);
        let x8 = _mm512_mul_ps(x7, x);
        let t1 = _mm512_mul_ps(_mm512_set1_ps(0.5), x2);
        let t2 = _mm512_mul_ps(_mm512_set1_ps(1.0 / 6.0), x3);
        let t3 = _mm512_mul_ps(_mm512_set1_ps(1.0 / 24.0), x4);
        let t4 = _mm512_mul_ps(_mm512_set1_ps(1.0 / 120.0), x5);
        let t5 = _mm512_mul_ps(_mm512_set1_ps(1.0 / 720.0), x6);
        let t6 = _mm512_mul_ps(_mm512_set1_ps(1.0 / 5040.0), x7);
        let t7 = _mm512_mul_ps(_mm512_set1_ps(1.0 / 40320.0), x8);
        let s1 = _mm512_add_ps(t1, t2);
        let s2 = _mm512_add_ps(t3, t4);
        let s3 = _mm512_add_ps(t5, t6);
        let s4 = _mm512_add_ps(s1, s2);
        let s5 = _mm512_add_ps(s3, s4);
        let s6 = _mm512_add_ps(s5, t7);
        let result = _mm512_add_ps(_mm512_add_ps(one, x), s6);
        result
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn sub_scalar_avx2(input: &[f32], output: &mut [f32], scalar: f32, len: usize) {
        use std::arch::x86_64::*;
        let scalar_vec = _mm256_set1_ps(scalar);
        let mut i = 0;
        while i + 8 <= len {
            let x = _mm256_loadu_ps(&input[i]);
            let result = _mm256_sub_ps(x, scalar_vec);
            _mm256_storeu_ps(&mut output[i], result);
            i += 8;
        }
        while i < len {
            output[i] = input[i] - scalar;
            i += 1;
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn sub_scalar_sse(input: &[f32], output: &mut [f32], scalar: f32, len: usize) {
        use std::arch::x86_64::*;
        let scalar_vec = _mm_set1_ps(scalar);
        let mut i = 0;
        while i + 4 <= len {
            let x = _mm_loadu_ps(&input[i]);
            let result = _mm_sub_ps(x, scalar_vec);
            _mm_storeu_ps(&mut output[i], result);
            i += 4;
        }
        while i < len {
            output[i] = input[i] - scalar;
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn sub_scalar_avx512(input: &[f32], output: &mut [f32], scalar: f32, len: usize) {
        use std::arch::x86_64::*;
        let scalar_vec = _mm512_set1_ps(scalar);
        let mut i = 0;
        while i + 16 <= len {
            let x = _mm512_loadu_ps(&input[i]);
            let result = _mm512_sub_ps(x, scalar_vec);
            _mm512_storeu_ps(&mut output[i], result);
            i += 16;
        }
        while i < len {
            output[i] = input[i] - scalar;
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn div_scalar_avx2(input: &[f32], output: &mut [f32], scalar: f32, len: usize) {
        use std::arch::x86_64::*;
        let scalar_vec = _mm256_set1_ps(scalar);
        let mut i = 0;
        while i + 8 <= len {
            let x = _mm256_loadu_ps(&input[i]);
            let result = _mm256_div_ps(x, scalar_vec);
            _mm256_storeu_ps(&mut output[i], result);
            i += 8;
        }
        while i < len {
            output[i] = input[i] / scalar;
            i += 1;
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn div_scalar_sse(input: &[f32], output: &mut [f32], scalar: f32, len: usize) {
        use std::arch::x86_64::*;
        let scalar_vec = _mm_set1_ps(scalar);
        let mut i = 0;
        while i + 4 <= len {
            let x = _mm_loadu_ps(&input[i]);
            let result = _mm_div_ps(x, scalar_vec);
            _mm_storeu_ps(&mut output[i], result);
            i += 4;
        }
        while i < len {
            output[i] = input[i] / scalar;
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn div_scalar_avx512(input: &[f32], output: &mut [f32], scalar: f32, len: usize) {
        use std::arch::x86_64::*;
        let scalar_vec = _mm512_set1_ps(scalar);
        let mut i = 0;
        while i + 16 <= len {
            let x = _mm512_loadu_ps(&input[i]);
            let result = _mm512_div_ps(x, scalar_vec);
            _mm512_storeu_ps(&mut output[i], result);
            i += 16;
        }
        while i < len {
            output[i] = input[i] / scalar;
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn mul_scalar_avx2(input: &[f32], output: &mut [f32], scalar: f32, len: usize) {
        use std::arch::x86_64::*;
        let scalar_vec = _mm256_set1_ps(scalar);
        let mut i = 0;
        while i + 8 <= len {
            let x = _mm256_loadu_ps(&input[i]);
            let result = _mm256_mul_ps(x, scalar_vec);
            _mm256_storeu_ps(&mut output[i], result);
            i += 8;
        }
        while i < len {
            output[i] = input[i] * scalar;
            i += 1;
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn mul_scalar_sse(input: &[f32], output: &mut [f32], scalar: f32, len: usize) {
        use std::arch::x86_64::*;
        let scalar_vec = _mm_set1_ps(scalar);
        let mut i = 0;
        while i + 4 <= len {
            let x = _mm_loadu_ps(&input[i]);
            let result = _mm_mul_ps(x, scalar_vec);
            _mm_storeu_ps(&mut output[i], result);
            i += 4;
        }
        while i < len {
            output[i] = input[i] * scalar;
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn mul_scalar_avx512(input: &[f32], output: &mut [f32], scalar: f32, len: usize) {
        use std::arch::x86_64::*;
        let scalar_vec = _mm512_set1_ps(scalar);
        let mut i = 0;
        while i + 16 <= len {
            let x = _mm512_loadu_ps(&input[i]);
            let result = _mm512_mul_ps(x, scalar_vec);
            _mm512_storeu_ps(&mut output[i], result);
            i += 16;
        }
        while i < len {
            output[i] = input[i] * scalar;
            i += 1;
        }
    }
}

// ============================================================================
// ARM64 NEON implementations
// ============================================================================

#[cfg(target_arch = "aarch64")]
mod aarch64_impls {
    use super::*;

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn relu_neon(input: &[f32], output: &mut [f32], len: usize) {
        use std::arch::aarch64::*;
        let zero = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 4 <= len {
            let data = vld1q_f32(&input[i]);
            let result = vmaxq_f32(zero, data);
            vst1q_f32(&mut output[i], result);
            i += 4;
        }
        while i < len {
            output[i] = if input[i] > 0.0 { input[i] } else { 0.0 };
            i += 1;
        }
    }

    /// ReLU on bytes directly - operates on 16 bytes at a time (4 f32 values)
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn relu_bytes_neon(input: &[u8], output: &mut [u8], len: usize) {
        use std::arch::aarch64::*;
        let zero = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 16 <= len * 4 {
            let data = vld1q_f32(input.as_ptr().add(i) as *const f32);
            let result = vmaxq_f32(zero, data);
            vst1q_f32(output.as_mut_ptr().add(i) as *mut f32, result);
            i += 16;
        }
        // Handle remaining bytes
        while i < len * 4 {
            let val = f32::from_le_bytes([input[i], input[i+1], input[i+2], input[i+3]]);
            let result = if val > 0.0 { val } else { 0.0 };
            output[i..i+4].copy_from_slice(&result.to_le_bytes());
            i += 4;
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn relu6_neon(input: &[f32], output: &mut [f32], len: usize) {
        use std::arch::aarch64::*;
        let zero = vdupq_n_f32(0.0);
        let six = vdupq_n_f32(6.0);
        let mut i = 0;
        while i + 4 <= len {
            let data = vld1q_f32(&input[i]);
            let clamped = vminq_f32(vmaxq_f32(zero, data), six);
            vst1q_f32(&mut output[i], clamped);
            i += 4;
        }
        while i < len {
            output[i] = input[i].clamp(0.0, 6.0);
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn add_neon(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::aarch64::*;
        let mut i = 0;
        while i + 4 <= len {
            let va = vld1q_f32(&a[i]);
            let vb = vld1q_f32(&b[i]);
            let vc = vaddq_f32(va, vb);
            vst1q_f32(&mut c[i], vc);
            i += 4;
        }
        while i < len {
            c[i] = a[i] + b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn sub_neon(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::aarch64::*;
        let mut i = 0;
        while i + 4 <= len {
            let va = vld1q_f32(&a[i]);
            let vb = vld1q_f32(&b[i]);
            let vc = vsubq_f32(va, vb);
            vst1q_f32(&mut c[i], vc);
            i += 4;
        }
        while i < len {
            c[i] = a[i] - b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn mul_neon(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::aarch64::*;
        let mut i = 0;
        while i + 4 <= len {
            let va = vld1q_f32(&a[i]);
            let vb = vld1q_f32(&b[i]);
            let vc = vmulq_f32(va, vb);
            vst1q_f32(&mut c[i], vc);
            i += 4;
        }
        while i < len {
            c[i] = a[i] * b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn div_neon(a: &[f32], b: &[f32], c: &mut [f32], len: usize) {
        use std::arch::aarch64::*;
        let mut i = 0;
        while i + 4 <= len {
            let va = vld1q_f32(&a[i]);
            let vb = vld1q_f32(&b[i]);
            let vc = vdivq_f32(va, vb);
            vst1q_f32(&mut c[i], vc);
            i += 4;
        }
        while i < len {
            c[i] = a[i] / b[i];
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn horizontal_sum_neon(arr: &[f32], len: usize) -> f32 {
        use std::arch::aarch64::*;
        let mut sum = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 4 <= len {
            let data = vld1q_f32(&arr[i]);
            sum = vaddq_f32(sum, data);
            i += 4;
        }
        // Extract lanes and add
        let mut result = vgetq_lane_f32(sum, 0) + vgetq_lane_f32(sum, 1) +
                         vgetq_lane_f32(sum, 2) + vgetq_lane_f32(sum, 3);
        while i < len {
            result += arr[i];
            i += 1;
        }
        result
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn gemm_neon(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        use std::arch::aarch64::*;

        // For small matrices, use the simple version
        if m * n < 512 {
            gemm_neon_simple(a, b, c, m, n, k);
            return;
        }

        // Cache-blocking: block over m and k for better cache utilization
        let mc = 64.min(m);
        let kc = 128.min(k);

        for k_block in (0..k).step_by(kc) {
            let k_end = (k_block + kc).min(k);

            for m_block in (0..m).step_by(mc) {
                let m_end = (m_block + mc).min(m);

                for i in m_block..m_end {
                    let mut j = 0;
                    // Process 4 elements at a time when possible
                    while j + 16 <= n {
                        let mut sum0 = 0.0f32;
                        let mut sum1 = 0.0f32;
                        let mut sum2 = 0.0f32;
                        let mut sum3 = 0.0f32;
                        let mut sum4 = 0.0f32;
                        let mut sum5 = 0.0f32;
                        let mut sum6 = 0.0f32;
                        let mut sum7 = 0.0f32;
                        let mut sum8 = 0.0f32;
                        let mut sum9 = 0.0f32;
                        let mut sum10 = 0.0f32;
                        let mut sum11 = 0.0f32;
                        let mut sum12 = 0.0f32;
                        let mut sum13 = 0.0f32;
                        let mut sum14 = 0.0f32;
                        let mut sum15 = 0.0f32;

                        for p in k_block..k_end {
                            let a_val = *a.get_unchecked(i * k + p);
                            let a_val2 = vdupq_n_f32(a_val);
                            let b_ptr = b.as_ptr().add(p * n + j);

                            // Process 16 B values at once
                            let b0 = *b_ptr;
                            let b1 = *b_ptr.add(1);
                            let b2 = *b_ptr.add(2);
                            let b3 = *b_ptr.add(3);
                            let b4 = *b_ptr.add(4);
                            let b5 = *b_ptr.add(5);
                            let b6 = *b_ptr.add(6);
                            let b7 = *b_ptr.add(7);
                            let b8 = *b_ptr.add(8);
                            let b9 = *b_ptr.add(9);
                            let b10 = *b_ptr.add(10);
                            let b11 = *b_ptr.add(11);
                            let b12 = *b_ptr.add(12);
                            let b13 = *b_ptr.add(13);
                            let b14 = *b_ptr.add(14);
                            let b15 = *b_ptr.add(15);

                            sum0 += a_val * b0;
                            sum1 += a_val * b1;
                            sum2 += a_val * b2;
                            sum3 += a_val * b3;
                            sum4 += a_val * b4;
                            sum5 += a_val * b5;
                            sum6 += a_val * b6;
                            sum7 += a_val * b7;
                            sum8 += a_val * b8;
                            sum9 += a_val * b9;
                            sum10 += a_val * b10;
                            sum11 += a_val * b11;
                            sum12 += a_val * b12;
                            sum13 += a_val * b13;
                            sum14 += a_val * b14;
                            sum15 += a_val * b15;
                        }

                        let c_idx = i * n + j;
                        *c.get_unchecked_mut(c_idx) += sum0;
                        *c.get_unchecked_mut(c_idx + 1) += sum1;
                        *c.get_unchecked_mut(c_idx + 2) += sum2;
                        *c.get_unchecked_mut(c_idx + 3) += sum3;
                        *c.get_unchecked_mut(c_idx + 4) += sum4;
                        *c.get_unchecked_mut(c_idx + 5) += sum5;
                        *c.get_unchecked_mut(c_idx + 6) += sum6;
                        *c.get_unchecked_mut(c_idx + 7) += sum7;
                        *c.get_unchecked_mut(c_idx + 8) += sum8;
                        *c.get_unchecked_mut(c_idx + 9) += sum9;
                        *c.get_unchecked_mut(c_idx + 10) += sum10;
                        *c.get_unchecked_mut(c_idx + 11) += sum11;
                        *c.get_unchecked_mut(c_idx + 12) += sum12;
                        *c.get_unchecked_mut(c_idx + 13) += sum13;
                        *c.get_unchecked_mut(c_idx + 14) += sum14;
                        *c.get_unchecked_mut(c_idx + 15) += sum15;
                        j += 16;
                    }
                    // Handle remaining n % 16
                    while j < n {
                        let mut sum = 0.0f32;
                        for p in k_block..k_end {
                            let a_val = *a.get_unchecked(i * k + p);
                            let b_val = *b.get_unchecked(p * n + j);
                            sum += a_val * b_val;
                        }
                        *c.get_unchecked_mut(i * n + j) += sum;
                        j += 1;
                    }
                }
            }
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn gemm_neon_simple(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        use std::arch::aarch64::*;
        for m_idx in 0..m {
            for n_idx in 0..n {
                let mut sum = 0.0f32;
                let mut k_idx = 0;
                while k_idx + 4 <= k {
                    let a_vec = vld1q_f32(a.as_ptr().add(m_idx * k + k_idx));
                    let b0 = vld1q_f32(b.as_ptr().add(k_idx * n + n_idx));
                    let b1 = vld1q_f32(b.as_ptr().add((k_idx + 1) * n + n_idx));
                    let b2 = vld1q_f32(b.as_ptr().add((k_idx + 2) * n + n_idx));
                    let b3 = vld1q_f32(b.as_ptr().add((k_idx + 3) * n + n_idx));
                    let a0 = vdupq_n_f32(vgetq_lane_f32(a_vec, 0));
                    let a1 = vdupq_n_f32(vgetq_lane_f32(a_vec, 1));
                    let a2 = vdupq_n_f32(vgetq_lane_f32(a_vec, 2));
                    let a3 = vdupq_n_f32(vgetq_lane_f32(a_vec, 3));
                    sum += vgetq_lane_f32(vmulq_f32(a0, b0), 0);
                    sum += vgetq_lane_f32(vmulq_f32(a1, b1), 0);
                    sum += vgetq_lane_f32(vmulq_f32(a2, b2), 0);
                    sum += vgetq_lane_f32(vmulq_f32(a3, b3), 0);
                    k_idx += 4;
                }
                while k_idx < k {
                    sum += a[m_idx * k + k_idx] * b[k_idx * n + n_idx];
                    k_idx += 1;
                }
                c[m_idx * n + n_idx] += sum;
            }
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn exp_neon(input: &[f32], output: &mut [f32], len: usize) {
        use std::arch::aarch64::*;
        let mut i = 0;
        while i + 4 <= len {
            // Load 4 floats and compute exp using scalar (NEON doesn't have exp intrinsic)
            let mut vals = [0.0f32; 4];
            for j in 0..4 {
                vals[j] = input[i + j].exp();
            }
            let result = vld1q_f32(vals.as_ptr());
            vst1q_f32(&mut output[i], result);
            i += 4;
        }
        while i < len {
            output[i] = input[i].exp();
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn sub_scalar_neon(input: &[f32], output: &mut [f32], scalar: f32, len: usize) {
        use std::arch::aarch64::*;
        let scalar_vec = vdupq_n_f32(scalar);
        let mut i = 0;
        while i + 4 <= len {
            let x = vld1q_f32(&input[i]);
            let result = vsubq_f32(x, scalar_vec);
            vst1q_f32(&mut output[i], result);
            i += 4;
        }
        while i < len {
            output[i] = input[i] - scalar;
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn div_scalar_neon(input: &[f32], output: &mut [f32], scalar: f32, len: usize) {
        use std::arch::aarch64::*;
        let scalar_vec = vdupq_n_f32(scalar);
        let mut i = 0;
        while i + 4 <= len {
            let x = vld1q_f32(&input[i]);
            let result = vdivq_f32(x, scalar_vec);
            vst1q_f32(&mut output[i], result);
            i += 4;
        }
        while i < len {
            output[i] = input[i] / scalar;
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn mul_scalar_neon(input: &[f32], output: &mut [f32], scalar: f32, len: usize) {
        use std::arch::aarch64::*;
        let scalar_vec = vdupq_n_f32(scalar);
        let mut i = 0;
        while i + 4 <= len {
            let x = vld1q_f32(&input[i]);
            let result = vmulq_f32(x, scalar_vec);
            vst1q_f32(&mut output[i], result);
            i += 4;
        }
        while i < len {
            output[i] = input[i] * scalar;
            i += 1;
        }
    }
}

// Aliases for the parent module to use
#[cfg(target_arch = "x86_64")]
use x86_64_impls::*;
#[cfg(target_arch = "aarch64")]
use aarch64_impls::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_simd() {
        let input = vec![-3.0, -1.0, 0.0, 1.0, 2.0, 5.0, 7.0, 8.0];
        let mut output = vec![0.0f32; 8];
        let level = detect_simd_level();
        relu_simd(&input, &mut output, level);
        assert_eq!(output, vec![0.0, 0.0, 0.0, 1.0, 2.0, 5.0, 7.0, 8.0]);
    }

    #[test]
    fn test_relu6_simd() {
        let input = vec![-3.0, -1.0, 0.0, 1.0, 2.0, 5.0, 7.0, 8.0];
        let mut output = vec![0.0f32; 8];
        let level = detect_simd_level();
        relu6_simd(&input, &mut output, level);
        assert_eq!(output, vec![0.0, 0.0, 0.0, 1.0, 2.0, 5.0, 6.0, 6.0]);
    }

    #[test]
    fn test_add_simd() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut c = vec![0.0f32; 8];
        let level = detect_simd_level();
        add_simd(&a, &b, &mut c, level);
        assert_eq!(c, vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]);
    }

    #[test]
    fn test_mul_simd() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut c = vec![0.0f32; 8];
        let level = detect_simd_level();
        mul_simd(&a, &b, &mut c, level);
        assert_eq!(c, vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0]);
    }

    #[test]
    fn test_gemm_simd() {
        // A: 2x3, B: 3x2, C: 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = vec![0.0f32; 4];
        let level = detect_simd_level();
        gemm_simd(&a, &b, &mut c, 2, 2, 3, level);
        // C[0][0] = 1*1 + 2*3 + 3*5 = 22
        // C[0][1] = 1*2 + 2*4 + 3*6 = 28
        // C[1][0] = 4*1 + 5*3 + 6*5 = 49
        // C[1][1] = 4*2 + 5*4 + 6*6 = 64
        assert_eq!(c, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_horizontal_sum() {
        let arr = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let level = detect_simd_level();
        let sum = horizontal_sum(&arr, level);
        assert!((sum - 36.0).abs() < 1e-5);
    }

    #[test]
    fn test_detect_simd_level() {
        let level = detect_simd_level();
        println!("Detected SIMD level: {:?}", level);
        // On aarch64, should always detect Neon or better
        #[cfg(target_arch = "aarch64")]
        assert!(matches!(level, SimdLevel::Neon | SimdLevel::Neonfp16 | SimdLevel::None));
        // On x86_64, should detect at least SSE2
        #[cfg(target_arch = "x86_64")]
        assert!(matches!(level, SimdLevel::Sse2 | SimdLevel::Sse3 | SimdLevel::Ssse3 |
                                SimdLevel::Sse4_1 | SimdLevel::Sse4_2 | SimdLevel::Avx |
                                SimdLevel::Avx2 | SimdLevel::Avx512 | SimdLevel::None));
    }
}
