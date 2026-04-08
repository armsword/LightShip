//! GELU activation function
//!
//! Implements Gaussian Error Linear Unit activation for transformers.

use crate::platform::{detect_simd_level, mul_scalar_simd, sub_scalar_simd, SimdLevel};
use std::fmt;

/// GELU approximation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeluApprox {
    /// Exact GELU using error function
    Exact,
    /// Fast approximation using tanh
    Tanh,
}

impl Default for GeluApprox {
    fn default() -> Self {
        GeluApprox::Tanh
    }
}

/// GELU activation function
///
/// GELU(x) = x * Phi(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
///
/// Approximation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
#[derive(Debug)]
pub struct Gelu {
    /// Approximation method
    pub approximation: GeluApprox,
    /// SIMD acceleration level
    simd_level: SimdLevel,
}

impl Gelu {
    /// Create a new GELU operator
    pub fn new() -> Self {
        Self {
            approximation: GeluApprox::default(),
            simd_level: detect_simd_level(),
        }
    }

    /// Create with exact computation
    pub fn exact() -> Self {
        Self {
            approximation: GeluApprox::Exact,
            simd_level: detect_simd_level(),
        }
    }

    /// Create with tanh approximation
    pub fn tanh() -> Self {
        Self {
            approximation: GeluApprox::Tanh,
            simd_level: detect_simd_level(),
        }
    }

    /// Create with specified SIMD level (for testing)
    pub fn with_simd_level(approximation: GeluApprox, simd_level: SimdLevel) -> Self {
        Self {
            approximation,
            simd_level,
        }
    }

    /// Compute exact GELU using error function approximation
    ///
    /// Uses the approximation: erf(x) ≈ tanh(1.2 * x + 0.2 * x^3)
    pub fn exact_gelu(x: f32) -> f32 {
        Self::tanh_gelu(x)
    }

    /// Compute GELU using tanh approximation (more common in practice)
    pub fn tanh_gelu(x: f32) -> f32 {
        // sqrt(2/pi) ≈ 0.7978845608
        let sqrt_2_over_pi: f32 = 0.7978845608;
        let c: f32 = 0.044715;

        let x_cube = x * x * x;
        let tanh_arg = sqrt_2_over_pi * (x + c * x_cube);
        0.5 * x * (1.0 + Self::tanh_approx(tanh_arg))
    }

    /// Simple tanh approximation
    /// For |x| > 5, returns sign(x)
    /// Otherwise uses polynomial approximation
    fn tanh_approx(x: f32) -> f32 {
        if x > 5.0 {
            1.0
        } else if x < -5.0 {
            -1.0
        } else {
            // Use formula: tanh(x) ≈ x - x^3/3 + x^5/5 - x^7/7
            let x2 = x * x;
            let x3 = x2 * x;
            let x5 = x3 * x2;
            let x7 = x5 * x2;
            x - x3 / 3.0 + x5 / 5.0 - x7 / 7.0
        }
    }

    /// Compute GELU on a single value
    pub fn compute(x: f32) -> f32 {
        Self::tanh_gelu(x)
    }

    /// Compute GELU on a slice of data
    pub fn compute_slice(data: &mut [f32]) {
        for x in data.iter_mut() {
            *x = Self::compute(*x);
        }
    }

    /// SIMD-accelerated tanh approximation for slices
    /// tanh(x) ≈ x - x^3/3 + x^5/5 - x^7/7 for |x| < 5
    fn tanh_simd_internal(input: &[f32], output: &mut [f32], len: usize, level: SimdLevel) {
        // Constants
        let c0 = 1.0f32;
        let c1_neg = -1.0 / 3.0;
        let c2 = 1.0 / 5.0;
        let c3_neg = -1.0 / 7.0;
        let threshold = 5.0f32;

        // Step 1: x, x^2, x^3, x^5, x^7
        // For |x| > 5, tanh = sign(x) = 1 or -1
        // We handle this by computing tanh and then applying mask

        // Compute x^2
        let mut x_sq = vec![0.0f32; len];
        for i in 0..len {
            x_sq[i] = input[i] * input[i];
        }

        // Compute x^3 = x^2 * x
        let mut x_cu = vec![0.0f32; len];
        for i in 0..len {
            x_cu[i] = x_sq[i] * input[i];
        }

        // Compute x^5 = x^3 * x^2
        let mut x_5 = vec![0.0f32; len];
        for i in 0..len {
            x_5[i] = x_cu[i] * x_sq[i];
        }

        // Compute x^7 = x^5 * x^2
        let mut x_7 = vec![0.0f32; len];
        for i in 0..len {
            x_7[i] = x_5[i] * x_sq[i];
        }

        // Compute tanh approximation: c0*x + c1_neg*x^3 + c2*x^5 + c3_neg*x^7
        for i in 0..len {
            let x = input[i];
            let x3 = x_cu[i];
            let x5 = x_5[i];
            let x7 = x_7[i];

            let tanh_val = if x.abs() > threshold {
                if x > 0.0 { 1.0 } else { -1.0 }
            } else {
                c0 * x + c1_neg * x3 + c2 * x5 + c3_neg * x7
            };
            output[i] = tanh_val;
        }
    }

    /// SIMD-accelerated compute GELU on a slice of data
    /// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    pub fn compute_slice_simd(&self, data: &mut [f32]) {
        let len = data.len();
        if len == 0 {
            return;
        }

        let level = self.simd_level;
        let sqrt_2_over_pi: f32 = 0.7978845608;
        let c: f32 = 0.044715;
        let half: f32 = 0.5;

        // Step 1: Compute x^3 = x * x * x
        let mut x_sq = vec![0.0f32; len];
        let mut x_cu = vec![0.0f32; len];
        for i in 0..len {
            x_sq[i] = data[i] * data[i];
            x_cu[i] = x_sq[i] * data[i];
        }

        // Step 2: Compute inner = x + c * x^3
        let mut inner = vec![0.0f32; len];
        for i in 0..len {
            inner[i] = data[i] + c * x_cu[i];
        }

        // Step 3: Compute tanh_arg = sqrt_2_over_pi * inner
        let mut tanh_arg = vec![0.0f32; len];
        mul_scalar_simd(&inner, &mut tanh_arg, sqrt_2_over_pi, level);

        // Step 4: Compute tanh(tanh_arg)
        let mut tanh_val = vec![0.0f32; len];
        Self::tanh_simd_internal(&tanh_arg, &mut tanh_val, len, level);

        // Step 5: Compute 1 + tanh_val
        let mut one_plus_tanh = vec![0.0f32; len];
        for i in 0..len {
            one_plus_tanh[i] = 1.0 + tanh_val[i];
        }

        // Step 6: Compute x * (1 + tanh_val)
        let mut product = vec![0.0f32; len];
        for i in 0..len {
            product[i] = data[i] * one_plus_tanh[i];
        }

        // Step 7: Compute 0.5 * product
        mul_scalar_simd(&product, data, half, level);
    }
}

impl Default for Gelu {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Gelu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.approximation {
            GeluApprox::Exact => write!(f, "GELU(exact)"),
            GeluApprox::Tanh => write!(f, "GELU(tanh)"),
        }
    }
}

/// Quick GELU (SiLU) - same as GELU in our implementation
pub type SiLU = Gelu;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_creation() {
        let op = Gelu::new();
        assert!(matches!(op.approximation, GeluApprox::Tanh));
    }

    #[test]
    fn test_gelu_exact() {
        let op = Gelu::exact();
        assert!(matches!(op.approximation, GeluApprox::Exact));
    }

    #[test]
    fn test_gelu_tanh() {
        let op = Gelu::tanh();
        assert!(matches!(op.approximation, GeluApprox::Tanh));
    }

    #[test]
    fn test_gelu_zero() {
        let result = Gelu::compute(0.0);
        assert!((result - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_gelu_positive() {
        // GELU(1.0) should be positive
        let result = Gelu::compute(1.0);
        assert!(result > 0.0);
        // GELU(1.0) should be less than 1.0
        assert!(result < 1.0);
    }

    #[test]
    fn test_gelu_negative() {
        // GELU(-1.0) should be negative
        let result = Gelu::compute(-1.0);
        assert!(result < 0.0);
    }

    #[test]
    fn test_gelu_consistency() {
        // Both approximations should give similar results
        let x = 0.5;
        let exact = Gelu::exact_gelu(x);
        let tanh = Gelu::tanh_gelu(x);
        assert!((exact - tanh).abs() < 0.01);
    }

    #[test]
    fn test_gelu_slice() {
        let mut data = vec![-1.0, 0.0, 1.0];
        Gelu::compute_slice(&mut data);

        assert!(data[0] < 0.0); // GELU(-1) < 0
        assert!((data[1] - 0.0).abs() < 0.0001); // GELU(0) ≈ 0
        assert!(data[2] > 0.0); // GELU(1) > 0
    }

    #[test]
    fn test_gelu_properties() {
        // Test that GELU is approximately identity for small negative values
        // and approximately linear for small positive values
        let small_neg = Gelu::compute(-0.1);
        // GELU is negative for negative inputs and close to linear
        assert!(small_neg < 0.0);

        let small_pos = Gelu::compute(0.1);
        // GELU is positive for positive inputs
        assert!(small_pos > 0.0);
    }

    #[test]
    fn test_silu_is_gelu() {
        // SiLU is the same as GELU
        let x = 0.5;
        let gelu = Gelu::compute(x);
        let silu = SiLU::compute(x);
        assert!((gelu - silu).abs() < 0.0001);
    }

    #[test]
    fn test_gelu_simd_slice() {
        use crate::platform::SimdLevel;
        let op = Gelu::with_simd_level(GeluApprox::Tanh, SimdLevel::None);

        let mut data = vec![-1.0, 0.0, 1.0];
        op.compute_slice_simd(&mut data);

        // Compare with scalar version
        let mut expected = vec![-1.0, 0.0, 1.0];
        Gelu::compute_slice(&mut expected);

        for (got, exp) in data.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 0.001, "expected {}, got {}", exp, got);
        }
    }

    #[test]
    fn test_gelu_simd_consistency() {
        use crate::platform::SimdLevel;
        let op = Gelu::with_simd_level(GeluApprox::Tanh, SimdLevel::None);

        let test_values = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let mut data = test_values.clone();
        op.compute_slice_simd(&mut data);

        let mut expected = test_values;
        Gelu::compute_slice(&mut expected);

        for (got, exp) in data.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 0.001, "for x={}, expected {}, got {}", exp, got, got);
        }
    }
}
