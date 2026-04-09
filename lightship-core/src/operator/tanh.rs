//! Tanh activation function
//!
//! Implements hyperbolic tangent activation: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
//! Uses the identity: tanh(x) = 2 * sigmoid(2x) - 1 for numerical stability.

use crate::platform::{detect_simd_level, exp_simd, mul_scalar_simd, SimdLevel};
use std::fmt;

/// Tanh activation function
///
/// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
///         = 2 * sigmoid(2x) - 1
///
/// Range: (-1, 1)
///
/// Properties:
/// - Smooth, S-shaped (sigmoid-like)
/// - Zero-centered output
/// - Strong gradient for moderate values
#[derive(Debug)]
pub struct Tanh {
    /// SIMD acceleration level
    simd_level: SimdLevel,
}

impl Tanh {
    /// Create a new Tanh operator
    pub fn new() -> Self {
        Self {
            simd_level: detect_simd_level(),
        }
    }

    /// Create with specified SIMD level (for testing)
    pub fn with_simd_level(simd_level: SimdLevel) -> Self {
        Self { simd_level }
    }

    /// Compute tanh on a single value using identity: tanh(x) = 2 * sigmoid(2x) - 1
    ///
    /// This is numerically stable because:
    /// - sigmoid(2x) is computed directly rather than using exp(x) - exp(-x)
    /// - For large x, sigmoid(2x) → 1, so tanh(x) → 2*1 - 1 = 1
    /// - For large negative x, sigmoid(2x) → 0, so tanh(x) → 2*0 - 1 = -1
    pub fn compute(x: f32) -> f32 {
        // Use identity: tanh(x) = 2 * sigmoid(2x) - 1
        // sigmoid(z) = 1 / (1 + exp(-z))
        // For tanh: z = 2x
        let z = 2.0 * x;

        if z > 20.0 {
            // sigmoid(20) ≈ 0.9999999979, so tanh ≈ 1
            return 1.0;
        }
        if z < -20.0 {
            // sigmoid(-20) ≈ 2.06e-9, so tanh ≈ -1
            return -1.0;
        }

        // sigmoid(z) = 1 / (1 + exp(-z))
        let sigmoid_z = 1.0 / (1.0 + (-z).exp());

        // tanh(x) = 2 * sigmoid(2x) - 1
        2.0 * sigmoid_z - 1.0
    }

    /// Compute tanh using exponential formula (alternative)
    /// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    pub fn compute_exp(x: f32) -> f32 {
        if x > 20.0 {
            return 1.0;
        }
        if x < -20.0 {
            return -1.0;
        }
        let exp_x = x.exp();
        let exp_neg_x = (-x).exp();
        (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
    }

    /// Compute tanh on a slice of data (scalar version)
    pub fn compute_slice(data: &mut [f32]) {
        for x in data.iter_mut() {
            *x = Self::compute(*x);
        }
    }

    /// SIMD-accelerated compute tanh on a slice of data
    /// Uses: tanh(x) = 2 * sigmoid(2x) - 1
    ///
    /// Steps:
    /// 1. Compute z = 2 * x (scalar mul)
    /// 2. Compute exp(-z) for each element
    /// 3. Compute sigmoid(z) = 1 / (1 + exp(-z))
    /// 4. Compute tanh = 2 * sigmoid - 1
    pub fn compute_slice_simd(&self, data: &mut [f32]) {
        let len = data.len();
        if len == 0 {
            return;
        }

        let level = self.simd_level;

        // Step 1: Compute z = 2 * x
        let mut z = vec![0.0f32; len];
        mul_scalar_simd(data, &mut z, 2.0, level);

        // Step 2: Compute exp(-z) for each element
        let mut neg_z = vec![0.0f32; len];
        for i in 0..len {
            neg_z[i] = -z[i];
        }
        let mut exp_neg_z = vec![0.0f32; len];
        exp_simd(&neg_z, &mut exp_neg_z, level);

        // Step 3: Compute tanh = (1 - exp(-2x)) / (1 + exp(-2x))
        // This is equivalent to: tanh = 2 * sigmoid(2x) - 1
        // sigmoid(2x) = 1 / (1 + exp(-2x))
        // tanh = 2 / (1 + exp(-2x)) - 1 = (2 - (1 + exp(-2x))) / (1 + exp(-2x))
        // tanh = (1 - exp(-2x)) / (1 + exp(-2x))
        for i in 0..len {
            let exp_neg_zi = exp_neg_z[i];

            // Handle numerical stability
            let tanh_val = if exp_neg_zi.is_infinite() && exp_neg_zi.is_sign_positive() {
                // exp(-z) is infinite means z is very negative (large negative x)
                // tanh → -1
                -1.0f32
            } else if exp_neg_zi == 0.0 {
                // exp(-z) is 0 means z is very positive (large positive x)
                // tanh → 1
                1.0f32
            } else {
                // tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
                (1.0 - exp_neg_zi) / (1.0 + exp_neg_zi)
            };
            data[i] = tanh_val;
        }
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Tanh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tanh")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tanh_creation() {
        let op = Tanh::new();
        assert!(matches!(op.simd_level, SimdLevel::Neon | SimdLevel::Sse2 | SimdLevel::Avx2 | SimdLevel::Avx512 | SimdLevel::None));
    }

    #[test]
    fn test_tanh_compute() {
        // tanh(0) = 0
        let result = Tanh::compute(0.0);
        assert!((result - 0.0).abs() < 0.0001);

        // tanh(1) ≈ 0.7616
        let result = Tanh::compute(1.0);
        assert!((result - 0.7616).abs() < 0.001);

        // tanh(-1) ≈ -0.7616
        let result = Tanh::compute(-1.0);
        assert!((result - (-0.7616)).abs() < 0.001);
    }

    #[test]
    fn test_tanh_symmetry() {
        // tanh(-x) = -tanh(x)
        let x = 0.5;
        let pos = Tanh::compute(x);
        let neg = Tanh::compute(-x);
        assert!((pos + neg).abs() < 0.0001);
    }

    #[test]
    fn test_tanh_bounds() {
        // tanh should always be in (-1, 1)
        let test_values = vec![-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0];
        for x in test_values {
            let result = Tanh::compute(x);
            assert!(result >= -1.0 && result <= 1.0, "tanh({}) = {} not in [-1, 1]", x, result);
        }
    }

    #[test]
    fn test_tanh_extreme() {
        // tanh should saturate at ±1 for large |x|
        let large_pos = Tanh::compute(100.0);
        assert!((large_pos - 1.0).abs() < 0.001, "tanh(100) = {} should be close to 1", large_pos);

        let large_neg = Tanh::compute(-100.0);
        assert!((large_neg - (-1.0)).abs() < 0.001, "tanh(-100) = {} should be close to -1", large_neg);
    }

    #[test]
    fn test_tanh_slice() {
        let mut data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        Tanh::compute_slice(&mut data);

        // Verify each value
        assert!((data[0] - Tanh::compute(-2.0)).abs() < 0.0001);
        assert!((data[1] - Tanh::compute(-1.0)).abs() < 0.0001);
        assert!((data[2] - 0.0).abs() < 0.0001);
        assert!((data[3] - Tanh::compute(1.0)).abs() < 0.0001);
        assert!((data[4] - Tanh::compute(2.0)).abs() < 0.0001);
    }

    #[test]
    fn test_tanh_simd() {
        let op = Tanh::with_simd_level(SimdLevel::None);

        let test_values = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let mut data = test_values.clone();
        op.compute_slice_simd(&mut data);

        // Compare with scalar version
        let mut expected = test_values;
        Tanh::compute_slice(&mut expected);

        for (got, exp) in data.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 0.0001, "for x={}, expected {}, got {}", exp, got, got);
        }
    }

    #[test]
    fn test_tanh_simd_consistency() {
        use crate::platform::SimdLevel;
        let op = Tanh::with_simd_level(SimdLevel::None);

        let test_values = vec![-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0];
        let mut data = test_values.clone();
        op.compute_slice_simd(&mut data);

        let mut expected = test_values;
        Tanh::compute_slice(&mut expected);

        for (got, exp) in data.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 0.001, "for x={}, expected {}, got {}", exp, got, got);
        }
    }

    #[test]
    fn test_tanh_exp_vs_direct() {
        // The direct formula should match exp formula
        let test_values = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        for &x in &test_values {
            let direct = Tanh::compute(x);
            let exp_val = Tanh::compute_exp(x);
            assert!((direct - exp_val).abs() < 0.0001, "tanh({}) = {} differs from exp formula {}", x, direct, exp_val);
        }
    }

    #[test]
    fn test_tanh_accuracy() {
        // Test against known values
        // tanh(0.5) ≈ 0.4621
        let result = Tanh::compute(0.5);
        assert!((result - 0.4621).abs() < 0.01, "tanh(0.5) = {} should be close to 0.4621", result);

        // tanh(2.0) ≈ 0.9640
        let result = Tanh::compute(2.0);
        assert!((result - 0.9640).abs() < 0.01, "tanh(2.0) = {} should be close to 0.9640", result);
    }

    #[test]
    fn test_tanh_derivative() {
        // d/dx tanh(x) = sech^2(x) = 1 - tanh^2(x)
        // At x=0, tanh(0)=0, so derivative = 1
        // Using finite differences
        let h = 0.0001;
        let deriv_approx = (Tanh::compute(h) - Tanh::compute(-h)) / (2.0 * h);
        assert!((deriv_approx - 1.0).abs() < 0.01, "tanh'(0) ≈ 1, got {}", deriv_approx);
    }
}