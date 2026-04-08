//! SiLU (Sigmoid Linear Unit) activation function
//!
//! SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//! Also known as Swish activation.
//!
//! The SiLU activation was introduced in "Swish: a Self-Gated Activation Function" (Ramachandran et al., 2017).
//! It has been shown to outperform ReLU and other activations in many deep learning tasks.

use crate::platform::{detect_simd_level, exp_simd, mul_simd, SimdLevel};
use std::fmt;

/// SiLU activation function (also known as Swish)
///
/// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
///
/// Properties:
/// - Smooth, non-monotonic activation
/// - Self-gated (multiplies the input by the sigmoid of itself)
/// - Bounded below (approaches 0 for x → -∞)
/// - Unbounded above (grows linearly for x → +∞)
/// - Smooth gradient (no dead gradients problem like ReLU)
#[derive(Debug)]
pub struct SiLU {
    /// SIMD acceleration level
    simd_level: SimdLevel,
}

impl SiLU {
    /// Create a new SiLU operator
    pub fn new() -> Self {
        Self {
            simd_level: detect_simd_level(),
        }
    }

    /// Create with specified SIMD level (for testing)
    pub fn with_simd_level(simd_level: SimdLevel) -> Self {
        Self { simd_level }
    }

    /// Compute SiLU on a single value (numerically stable)
    /// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    ///
    /// Numerically stable computation:
    /// - For x > 20: sigmoid(x) ≈ 1, so SiLU(x) ≈ x
    /// - For x < -20: sigmoid(x) ≈ 0, so SiLU(x) ≈ 0
    pub fn compute(x: f32) -> f32 {
        if x > 20.0 {
            // SiLU(20) ≈ 20 * 0.9999999979 ≈ 19.999999958
            // Approximating as x for large positive values
            return x;
        }
        if x < -20.0 {
            // SiLU(-20) ≈ -20 * 2.06e-9 ≈ -4.12e-8
            // Approximating as 0 for large negative values
            return 0.0;
        }
        // Standard formula: x / (1 + exp(-x))
        // This is numerically stable for moderate x values
        x / (1.0 + (-x).exp())
    }

    /// Compute SiLU on a slice of data (scalar version)
    pub fn compute_slice(data: &mut [f32]) {
        for x in data.iter_mut() {
            *x = Self::compute(*x);
        }
    }

    /// SIMD-accelerated compute SiLU on a slice of data
    ///
    /// Uses exp_simd for sigmoid computation, then multiplies by input
    /// SiLU(x) = x * sigmoid(x) = x * exp(x) / (1 + exp(x))
    ///
    /// For numerical stability:
    /// - Large positive x: exp(x) overflows → result ≈ x
    /// - Large negative x: exp(x) underflows to 0 → result ≈ 0
    pub fn compute_slice_simd(&self, data: &mut [f32]) {
        let len = data.len();
        if len == 0 {
            return;
        }

        let level = self.simd_level;

        // Compute exp(x) for each element using SIMD
        let mut exp_x = vec![0.0f32; len];
        exp_simd(data, &mut exp_x, level);

        // Compute SiLU using loop with scalar division (only division needs scalar op)
        // SiLU(x) = x * exp(x) / (1 + exp(x))
        for i in 0..len {
            let exp_xi = exp_x[i];
            let sigmoid = if exp_xi.is_infinite() && exp_xi.is_sign_positive() {
                // Large positive x: sigmoid ≈ 1
                1.0f32
            } else if exp_xi == 0.0 {
                // Large negative x: sigmoid ≈ 0
                0.0f32
            } else {
                exp_xi / (1.0 + exp_xi)
            };
            data[i] = data[i] * sigmoid;
        }
    }

    /// SIMD-accelerated SiLU with explicit output buffer
    pub fn compute_slice_simd_with_output(&self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        if len == 0 {
            return;
        }

        let level = self.simd_level;

        // Compute exp(x) for each element using SIMD
        let mut exp_x = vec![0.0f32; len];
        exp_simd(input, &mut exp_x, level);

        // Compute SiLU using loop with scalar division
        for i in 0..len {
            let xi = input[i];
            let exp_xi = exp_x[i];
            let sigmoid = if exp_xi.is_infinite() && exp_xi.is_sign_positive() {
                1.0f32
            } else if exp_xi == 0.0 {
                0.0f32
            } else {
                exp_xi / (1.0 + exp_xi)
            };
            output[i] = xi * sigmoid;
        }
    }
}

impl Default for SiLU {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SiLU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SiLU")
    }
}

/// SiLU alias (same as above)
pub type Swish = SiLU;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_creation() {
        let op = SiLU::new();
        assert!(matches!(op.simd_level, SimdLevel::Neon | SimdLevel::Sse2 | SimdLevel::Avx2 | SimdLevel::Avx512 | SimdLevel::None));
    }

    #[test]
    fn test_silu_compute() {
        // SiLU(0) = 0 * sigmoid(0) = 0
        let result = SiLU::compute(0.0);
        assert!((result - 0.0).abs() < 0.0001);

        // SiLU(1) = 1 * sigmoid(1) ≈ 1 * 0.731 = 0.731
        let result = SiLU::compute(1.0);
        assert!((result - 0.731).abs() < 0.001);

        // SiLU(-1) = -1 * sigmoid(-1) ≈ -1 * 0.269 = -0.269
        let result = SiLU::compute(-1.0);
        assert!((result - (-0.269)).abs() < 0.001);
    }

    #[test]
    fn test_silu_slice() {
        let mut data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        SiLU::compute_slice(&mut data);

        // Verify each value
        assert!((data[0] - SiLU::compute(-2.0)).abs() < 0.0001);
        assert!((data[1] - SiLU::compute(-1.0)).abs() < 0.0001);
        assert!((data[2] - 0.0).abs() < 0.0001);
        assert!((data[3] - SiLU::compute(1.0)).abs() < 0.0001);
        assert!((data[4] - SiLU::compute(2.0)).abs() < 0.0001);
    }

    #[test]
    fn test_silu_simd() {
        let op = SiLU::with_simd_level(SimdLevel::None);

        let test_values = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let mut data = test_values.clone();
        op.compute_slice_simd(&mut data);

        // Compare with scalar version
        let mut expected = test_values;
        SiLU::compute_slice(&mut expected);

        for (got, exp) in data.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 0.0001, "for x={}, expected {}, got {}", exp, got, got);
        }
    }

    #[test]
    fn test_silu_properties() {
        // SiLU(x) should be approximately x for large positive x
        let large_pos = SiLU::compute(100.0);
        assert!((large_pos - 100.0).abs() < 1.0, "SiLU(100) = {} should be close to 100", large_pos);

        // SiLU(x) should be approximately 0 for large negative x
        let large_neg = SiLU::compute(-100.0);
        assert!(large_neg.abs() < 0.001, "SiLU(-100) = {} should be close to 0", large_neg);

        // SiLU(0) = 0
        assert!((SiLU::compute(0.0)).abs() < 0.0001);

        // SiLU is positive for positive x
        assert!(SiLU::compute(1.0) > 0.0);

        // SiLU is negative for negative x
        assert!(SiLU::compute(-1.0) < 0.0);
    }

    #[test]
    fn test_swish_is_silu() {
        // SiLU is the same as Swish
        let x = 0.5;
        let silu = SiLU::compute(x);
        let swish = Swish::compute(x);
        assert!((silu - swish).abs() < 0.0001);
    }

    #[test]
    fn test_silu_simd_consistency() {
        use crate::platform::SimdLevel;
        let op = SiLU::with_simd_level(SimdLevel::None);

        let test_values = vec![-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0];
        let mut data = test_values.clone();
        op.compute_slice_simd(&mut data);

        let mut expected = test_values;
        SiLU::compute_slice(&mut expected);

        for (got, exp) in data.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 0.001, "for x={}, expected {}, got {}", exp, got, got);
        }
    }

    #[test]
    fn test_silu_output_buffer() {
        use crate::platform::SimdLevel;
        let op = SiLU::with_simd_level(SimdLevel::None);

        let input = vec![-1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0f32; 4];
        op.compute_slice_simd_with_output(&input, &mut output);

        let mut expected = input.clone();
        SiLU::compute_slice(&mut expected);

        for (got, exp) in output.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 0.001, "expected {}, got {}", exp, got);
        }
    }

    #[test]
    fn test_silu_comparison_with_relu() {
        // SiLU should be similar to ReLU for positive values (since sigmoid(x) ≈ 1 for large x)
        let x = 10.0;
        let silu = SiLU::compute(x);
        let relu = x.max(0.0);
        assert!((silu - relu).abs() < 0.1, "SiLU({}) = {} should be close to ReLU({}) = {}", x, silu, x, relu);

        // For negative values, SiLU is different from ReLU
        let x = -5.0;
        let silu = SiLU::compute(x);
        let relu = x.max(0.0);
        assert!(silu < relu, "SiLU({}) = {} should be less than ReLU({}) = 0", x, silu, x);
    }
}