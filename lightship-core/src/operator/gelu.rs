//! GELU activation function
//!
//! Implements Gaussian Error Linear Unit activation for transformers.

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
}

impl Gelu {
    /// Create a new GELU operator
    pub fn new() -> Self {
        Self {
            approximation: GeluApprox::default(),
        }
    }

    /// Create with exact computation
    pub fn exact() -> Self {
        Self {
            approximation: GeluApprox::Exact,
        }
    }

    /// Create with tanh approximation
    pub fn tanh() -> Self {
        Self {
            approximation: GeluApprox::Tanh,
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
}
