//! Sigmoid activation function
//!
//! Implements sigmoid activation: sigmoid(x) = 1 / (1 + exp(-x))
//! Also provides optimized Softmax which can be used for sigmoid when needed.

use crate::platform::{detect_simd_level, exp_simd, sub_scalar_simd, div_scalar_simd, SimdLevel};
use std::fmt;

/// Sigmoid activation function
///
/// sigmoid(x) = 1 / (1 + exp(-x))
/// Returns values in range (0, 1)
#[derive(Debug)]
pub struct Sigmoid {
    /// SIMD acceleration level
    simd_level: SimdLevel,
}

impl Sigmoid {
    /// Create a new Sigmoid operator
    pub fn new() -> Self {
        Self {
            simd_level: detect_simd_level(),
        }
    }

    /// Create with specified SIMD level (for testing)
    pub fn with_simd_level(simd_level: SimdLevel) -> Self {
        Self { simd_level }
    }

    /// Compute sigmoid on a single value (numerically stable)
    /// sigmoid(x) = 1 / (1 + exp(-x))
    /// For large positive x, use: sigmoid(x) = 1 / (1 + exp(-x)) ≈ 1
    /// For large negative x, use: sigmoid(x) = exp(x) / (1 + exp(x)) ≈ exp(x)
    pub fn compute(x: f32) -> f32 {
        if x > 20.0 {
            // sigmoid(20) ≈ 0.9999999979, close enough to 1
            return 1.0;
        }
        if x < -20.0 {
            // sigmoid(-20) ≈ 2.06e-9, close enough to 0
            return 0.0;
        }
        // Standard formula for normal range
        1.0 / (1.0 + (-x).exp())
    }

    /// Compute sigmoid on a slice of data
    pub fn compute_slice(data: &mut [f32]) {
        for x in data.iter_mut() {
            *x = Self::compute(*x);
        }
    }

    /// SIMD-accelerated compute sigmoid on a slice of data
    /// Uses numerically stable approach for large values
    pub fn compute_slice_simd(&self, data: &mut [f32]) {
        let len = data.len();
        if len == 0 {
            return;
        }

        let level = self.simd_level;

        // Compute exp(x) for each element
        let mut exp_x = vec![0.0f32; len];
        exp_simd(data, &mut exp_x, level);

        // sigmoid = exp(x) / (1 + exp(x))
        // For large positive x: exp(x) is huge, result ≈ 1
        // For large negative x: exp(x) ≈ 0, result ≈ 0
        for i in 0..len {
            let exp_xi = exp_x[i];
            if exp_xi.is_infinite() && exp_xi.is_sign_positive() {
                // Large positive x
                data[i] = 1.0;
            } else if exp_xi == 0.0 {
                // Large negative x (exp(-x) overflowed to inf, so x was very negative)
                data[i] = 0.0;
            } else {
                data[i] = exp_xi / (1.0 + exp_xi);
            }
        }
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Sigmoid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sigmoid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_creation() {
        let op = Sigmoid::new();
        assert!(matches!(op.simd_level, SimdLevel::Neon | SimdLevel::Sse2 | SimdLevel::Avx2 | SimdLevel::Avx512 | SimdLevel::None));
    }

    #[test]
    fn test_sigmoid_compute() {
        // sigmoid(0) = 0.5
        let result = Sigmoid::compute(0.0);
        assert!((result - 0.5).abs() < 0.0001);

        // sigmoid(1) > 0.5
        let result = Sigmoid::compute(1.0);
        assert!(result > 0.5 && result < 1.0);

        // sigmoid(-1) < 0.5
        let result = Sigmoid::compute(-1.0);
        assert!(result > 0.0 && result < 0.5);
    }

    #[test]
    fn test_sigmoid_slice() {
        let mut data = vec![-1.0, 0.0, 1.0];
        Sigmoid::compute_slice(&mut data);

        assert!(data[0] > 0.0 && data[0] < 0.5);  // sigmoid(-1) in (0, 0.5)
        assert!((data[1] - 0.5).abs() < 0.0001);   // sigmoid(0) = 0.5
        assert!(data[2] > 0.5 && data[2] < 1.0);  // sigmoid(1) in (0.5, 1)
    }

    #[test]
    fn test_sigmoid_simd() {
        let op = Sigmoid::with_simd_level(SimdLevel::None);

        let test_values = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let mut data = test_values.clone();
        op.compute_slice_simd(&mut data);

        // Compare with scalar version
        let mut expected = test_values;
        Sigmoid::compute_slice(&mut expected);

        for (got, exp) in data.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 0.0001, "for x={}, expected {}, got {}", exp, got, got);
        }
    }

    #[test]
    fn test_sigmoid_range() {
        // Sigmoid should always return values in (0, 1)
        // For extreme values, sigmoid returns 0 or 1
        let test_values = vec![-50.0, -20.0, -10.0, 0.0, 10.0, 20.0, 50.0];
        for x in test_values {
            let result = Sigmoid::compute(x);
            assert!(result >= 0.0 && result <= 1.0, "sigmoid({}) = {} not in [0, 1]", x, result);
        }
    }
}
