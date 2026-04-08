//! Softmax operator
//!
//! Implements softmax normalization for multi-class classification.

use crate::ir::Tensor;
use crate::platform::{detect_simd_level, div_scalar_simd, exp_simd, horizontal_sum, sub_scalar_simd, SimdLevel};
use std::fmt;

/// Softmax operator result
#[derive(Debug)]
pub struct SoftmaxOutput {
    /// Output tensor
    pub output: Tensor,
    /// Maximum value used for numerical stability
    pub max_val: f32,
    /// Sum of exponentials
    pub exp_sum: f32,
}

/// Softmax normalization axis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxAxis {
    /// Apply softmax along the last dimension (default for NLP)
    Last,
    /// Apply softmax along the first dimension (for CNN)
    First,
    /// Apply softmax along specified axis
    Axis(usize),
}

impl Default for SoftmaxAxis {
    fn default() -> Self {
        SoftmaxAxis::Last
    }
}

/// Softmax operator
///
/// Computes softmax(x_i) = exp(x_i) / sum(exp(x_j))
///
/// Uses the numerically stable version:
/// softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
#[derive(Debug)]
pub struct Softmax {
    /// Axis along which to apply softmax
    pub axis: SoftmaxAxis,
    /// Number of elements to process (for partial softmax)
    pub num_elements: Option<usize>,
    /// SIMD acceleration level
    simd_level: SimdLevel,
}

impl Softmax {
    /// Create a new Softmax operator
    pub fn new() -> Self {
        Self {
            axis: SoftmaxAxis::default(),
            num_elements: None,
            simd_level: detect_simd_level(),
        }
    }

    /// Create Softmax along the last dimension
    pub fn last() -> Self {
        Self {
            axis: SoftmaxAxis::Last,
            num_elements: None,
            simd_level: detect_simd_level(),
        }
    }

    /// Create Softmax along the first dimension
    pub fn first() -> Self {
        Self {
            axis: SoftmaxAxis::First,
            num_elements: None,
            simd_level: detect_simd_level(),
        }
    }

    /// Create Softmax along a specific axis
    pub fn axis(axis: usize) -> Self {
        Self {
            axis: SoftmaxAxis::Axis(axis),
            num_elements: None,
            simd_level: detect_simd_level(),
        }
    }

    /// Create Softmax with specified SIMD level (for testing)
    pub fn with_simd_level(axis: SoftmaxAxis, simd_level: SimdLevel) -> Self {
        Self {
            axis,
            num_elements: None,
            simd_level,
        }
    }

    /// Compute softmax output shape (same as input shape)
    pub fn output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        input_shape.to_vec()
    }

    /// Apply softmax on a slice of data
    /// Returns the sum of exponentials for numerical stability
    pub fn compute_sum_exp(data: &[f32]) -> (f32, f32) {
        if data.is_empty() {
            return (0.0, 0.0);
        }

        // Find max for numerical stability
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute sum of exponentials
        let exp_sum: f32 = data.iter().map(|&x| (x - max_val).exp()).sum();

        (max_val, exp_sum)
    }

    /// SIMD-accelerated version of compute_sum_exp
    /// Returns the sum of exponentials for numerical stability
    pub fn compute_sum_exp_simd(&self, data: &[f32]) -> (f32, f32) {
        if data.is_empty() {
            return (0.0, 0.0);
        }

        // Find max for numerical stability (scalar, as SIMD reduction for max is complex)
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp(x - max) for each element using SIMD
        // Step 1: sub_scalar into temp1
        let mut temp1 = vec![0.0f32; data.len()];
        sub_scalar_simd(data, &mut temp1, max_val, self.simd_level);
        // Step 2: exp into temp2 (need separate buffer)
        let mut temp2 = vec![0.0f32; data.len()];
        exp_simd(&temp1, &mut temp2, self.simd_level);

        // Sum exponentials using SIMD horizontal sum
        let exp_sum = horizontal_sum(&temp2, self.simd_level);

        (max_val, exp_sum)
    }

    /// Compute softmax on a slice of data
    pub fn compute(data: &mut [f32]) {
        if data.is_empty() {
            return;
        }

        let (max_val, exp_sum) = Self::compute_sum_exp(data);

        if exp_sum > 0.0 {
            let scale = 1.0 / exp_sum;
            for x in data.iter_mut() {
                *x = ((*x - max_val).exp()) * scale;
            }
        }
    }

    /// SIMD-accelerated compute softmax on a slice of data
    pub fn compute_simd(&self, data: &mut [f32]) {
        if data.is_empty() {
            return;
        }

        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        if max_val == f32::NEG_INFINITY {
            return;
        }

        // Compute exp(x - max) and sum
        let mut temp1 = vec![0.0f32; data.len()];
        sub_scalar_simd(data, &mut temp1, max_val, self.simd_level);
        let mut temp2 = vec![0.0f32; data.len()];
        exp_simd(&temp1, &mut temp2, self.simd_level);
        let exp_sum = horizontal_sum(&temp2, self.simd_level);

        if exp_sum > 0.0 {
            // Apply softmax: exp(x - max) / exp_sum
            div_scalar_simd(&temp2, data, exp_sum, self.simd_level);
        }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Softmax {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.axis {
            SoftmaxAxis::Last => write!(f, "Softmax(axis=last)"),
            SoftmaxAxis::First => write!(f, "Softmax(axis=first)"),
            SoftmaxAxis::Axis(a) => write!(f, "Softmax(axis={})", a),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_creation() {
        let op = Softmax::new();
        assert!(matches!(op.axis, SoftmaxAxis::Last));
    }

    #[test]
    fn test_softmax_first() {
        let op = Softmax::first();
        assert!(matches!(op.axis, SoftmaxAxis::First));
    }

    #[test]
    fn test_softmax_output_shape() {
        let op = Softmax::new();
        let shape = op.output_shape(&[2, 3, 4]);
        assert_eq!(shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_softmax_compute_sum_exp() {
        let data = vec![1.0, 2.0, 3.0];
        let (max_val, exp_sum) = Softmax::compute_sum_exp(&data);

        assert_eq!(max_val, 3.0);
        // exp(1-3) + exp(2-3) + exp(3-3) = exp(-2) + exp(-1) + exp(0)
        assert!((exp_sum - (std::f32::consts::E.powi(-2) + std::f32::consts::E.powi(-1) + 1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_softmax_compute() {
        let mut data = vec![1.0, 2.0, 3.0];
        Softmax::compute(&mut data);

        // Sum should be 1
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 0.0001);

        // All values should be positive
        assert!(data.iter().all(|&x| x > 0.0));

        // Largest value should be the largest output
        let max_output = data.iter().fold(0.0f32, |a, &b| a.max(b));
        assert!((max_output - data[2]).abs() < 0.0001);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow without stabilization
        let mut data = vec![1000.0, 1001.0, 1002.0];
        Softmax::compute(&mut data);

        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_softmax_simd_compute() {
        let level = detect_simd_level();
        let op = Softmax::with_simd_level(SoftmaxAxis::Last, level);

        let mut data = vec![1.0, 2.0, 3.0];
        op.compute_simd(&mut data);

        // Sum should be 1
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 0.0001);

        // All values should be positive
        assert!(data.iter().all(|&x| x > 0.0));

        // Largest value should be the largest output
        let max_output = data.iter().fold(0.0f32, |a, &b| a.max(b));
        assert!((max_output - data[2]).abs() < 0.0001);
    }

    #[test]
    fn test_softmax_simd_compute_sum_exp() {
        use crate::platform::SimdLevel;

        let op = Softmax::with_simd_level(SoftmaxAxis::Last, SimdLevel::None);
        let data = vec![1.0, 2.0, 3.0];
        let (max_val, exp_sum) = op.compute_sum_exp_simd(&data);

        assert_eq!(max_val, 3.0);
        // exp(1-3) + exp(2-3) + exp(3-3) = exp(-2) + exp(-1) + exp(0)
        assert!((exp_sum - (std::f32::consts::E.powi(-2) + std::f32::consts::E.powi(-1) + 1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_softmax_simd_numerical_stability() {
        let level = detect_simd_level();
        let op = Softmax::with_simd_level(SoftmaxAxis::Last, level);

        // Large values that would overflow without stabilization
        let mut data = vec![1000.0, 1001.0, 1002.0];
        op.compute_simd(&mut data);

        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 0.0001);
    }
}
