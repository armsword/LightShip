//! Layer normalization operator
//!
//! Implements layer normalization for transformer models.

use crate::ir::Tensor;
use crate::platform::{detect_simd_level, div_scalar_simd, horizontal_sum, mul_scalar_simd, mul_simd, sub_scalar_simd, SimdLevel};
use std::fmt;

/// LayerNorm operator result
#[derive(Debug)]
pub struct LayerNormOutput {
    /// Output tensor
    pub output: Tensor,
    /// Mean computed
    pub mean: f32,
    /// Variance computed
    pub variance: f32,
}

/// Layer normalization affine parameters
#[derive(Debug, Clone)]
pub struct LayerNormAffine {
    /// Weight (gamma)
    pub weight: Option<Vec<f32>>,
    /// Bias (beta)
    pub bias: Option<Vec<f32>>,
}

impl LayerNormAffine {
    /// Create new affine parameters
    pub fn new() -> Self {
        Self {
            weight: None,
            bias: None,
        }
    }

    /// With weight
    pub fn with_weight(mut self, weight: Vec<f32>) -> Self {
        self.weight = Some(weight);
        self
    }

    /// With bias
    pub fn with_bias(mut self, bias: Vec<f32>) -> Self {
        self.bias = Some(bias);
        self
    }
}

impl Default for LayerNormAffine {
    fn default() -> Self {
        Self::new()
    }
}

/// Layer normalization operator
///
/// LayerNorm(x) = (x - mean) / sqrt(variance + eps) * gamma + beta
///
/// Typically applied over the last D dimensions where D is the normalized shape.
#[derive(Debug)]
pub struct LayerNorm {
    /// Number of features in the normalized shape
    pub normalized_shape: Vec<usize>,
    /// Epsilon for numerical stability
    pub epsilon: f32,
    /// Affine transformation parameters
    pub affine: LayerNormAffine,
    /// Whether to train (compute gradients) mode
    pub training: bool,
    /// SIMD acceleration level
    simd_level: SimdLevel,
}

impl LayerNorm {
    /// Create a new LayerNorm operator
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        Self {
            normalized_shape,
            epsilon: 1e-5,
            affine: LayerNormAffine::new(),
            training: false,
            simd_level: detect_simd_level(),
        }
    }

    /// Create with epsilon
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Create with affine transformation
    pub fn with_affine(mut self, affine: LayerNormAffine) -> Self {
        self.affine = affine;
        self
    }

    /// Create training mode
    pub fn training(mut self) -> Self {
        self.training = true;
        self
    }

    /// Create with specified SIMD level (for testing)
    pub fn with_simd_level(normalized_shape: Vec<usize>, simd_level: SimdLevel) -> Self {
        Self {
            normalized_shape,
            epsilon: 1e-5,
            affine: LayerNormAffine::new(),
            training: false,
            simd_level,
        }
    }

    /// Get output shape (same as input shape)
    pub fn output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        input_shape.to_vec()
    }

    /// Compute mean of a slice
    pub fn compute_mean(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f32>() / data.len() as f32
    }

    /// Compute variance of a slice
    pub fn compute_variance(data: &[f32], mean: f32) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        let sum_sq_diff: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum();
        sum_sq_diff / data.len() as f32
    }

    /// Normalize a slice of data
    pub fn normalize(data: &mut [f32], epsilon: f32) -> (f32, f32) {
        let mean = Self::compute_mean(data);
        let variance = Self::compute_variance(data, mean);
        let std_dev = (variance + epsilon).sqrt();

        for x in data.iter_mut() {
            *x = (*x - mean) / std_dev;
        }

        (mean, variance)
    }

    /// SIMD-accelerated compute mean of a slice
    pub fn compute_mean_simd(&self, data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        horizontal_sum(data, self.simd_level) / data.len() as f32
    }

    /// SIMD-accelerated compute variance of a slice
    pub fn compute_variance_simd(&self, data: &[f32], mean: f32) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        let len = data.len();
        // Compute (x - mean)^2
        let mut temp = vec![0.0f32; len];
        sub_scalar_simd(data, &mut temp, mean, self.simd_level);  // temp = x - mean
        let mut temp_sq = vec![0.0f32; len];
        mul_simd(&temp, &temp, &mut temp_sq, self.simd_level);  // temp_sq = (x - mean)^2
        let sum_sq_diff = horizontal_sum(&temp_sq, self.simd_level);
        sum_sq_diff / len as f32
    }

    /// SIMD-accelerated normalize a slice of data
    pub fn normalize_simd(&self, data: &mut [f32], epsilon: f32) -> (f32, f32) {
        if data.is_empty() {
            return (0.0, 0.0);
        }
        let len = data.len();
        let mean = self.compute_mean_simd(data);
        let variance = self.compute_variance_simd(data, mean);
        let std_dev = (variance + epsilon).sqrt();

        // Compute (x - mean) / std_dev
        let mut temp = vec![0.0f32; len];
        sub_scalar_simd(data, &mut temp, mean, self.simd_level);
        div_scalar_simd(&temp, data, std_dev, self.simd_level);

        // Apply gamma (weight) if present
        if let Some(ref gamma) = self.affine.weight {
            let mut temp2 = vec![0.0f32; len];
            for i in 0..len {
                temp2[i] = gamma[i % gamma.len()] * data[i];
            }
            data.copy_from_slice(&temp2);
        }

        // Add beta if present
        if let Some(ref beta) = self.affine.bias {
            for i in 0..len {
                data[i] += beta[i % beta.len()];
            }
        }

        (mean, variance)
    }
}

impl Default for LayerNorm {
    fn default() -> Self {
        Self::new(vec![1])
    }
}

impl fmt::Display for LayerNorm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LayerNorm(shape={:?}, eps={}, training={})",
            self.normalized_shape, self.epsilon, self.training
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layernorm_creation() {
        let op = LayerNorm::new(vec![4]);
        assert_eq!(op.normalized_shape, vec![4]);
        assert_eq!(op.epsilon, 1e-5);
        assert!(!op.training);
    }

    #[test]
    fn test_layernorm_with_epsilon() {
        let op = LayerNorm::new(vec![4]).with_epsilon(1e-3);
        assert_eq!(op.epsilon, 1e-3);
    }

    #[test]
    fn test_layernorm_output_shape() {
        let op = LayerNorm::new(vec![4]);
        let shape = op.output_shape(&[2, 3, 4]);
        assert_eq!(shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_compute_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mean = LayerNorm::compute_mean(&data);
        assert!((mean - 2.5).abs() < 0.0001);
    }

    #[test]
    fn test_compute_variance() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mean = LayerNorm::compute_mean(&data);
        let variance = LayerNorm::compute_variance(&data, mean);
        // Variance of [1,2,3,4] is 1.25
        assert!((variance - 1.25).abs() < 0.0001);
    }

    #[test]
    fn test_normalize() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let (mean, variance) = LayerNorm::normalize(&mut data, 1e-5);

        assert!((mean - 2.5).abs() < 0.0001);
        // Variance should be close to 1.0 after normalization
        let new_variance = LayerNorm::compute_variance(&data, LayerNorm::compute_mean(&data));
        assert!((new_variance - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_layernorm_affine() {
        let affine = LayerNormAffine::new()
            .with_weight(vec![1.0, 2.0, 3.0, 4.0])
            .with_bias(vec![0.1, 0.2, 0.3, 0.4]);

        let op = LayerNorm::new(vec![4]).with_affine(affine);
        assert!(op.affine.weight.is_some());
        assert!(op.affine.bias.is_some());
    }

    #[test]
    fn test_compute_mean_simd() {
        use crate::platform::SimdLevel;
        let op = LayerNorm::with_simd_level(vec![4], SimdLevel::None);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mean = op.compute_mean_simd(&data);
        assert!((mean - 2.5).abs() < 0.0001);
    }

    #[test]
    fn test_compute_variance_simd() {
        use crate::platform::SimdLevel;
        let op = LayerNorm::with_simd_level(vec![4], SimdLevel::None);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mean = op.compute_mean_simd(&data);
        let variance = op.compute_variance_simd(&data, mean);
        // Variance of [1,2,3,4] is 1.25
        assert!((variance - 1.25).abs() < 0.0001);
    }

    #[test]
    fn test_normalize_simd() {
        use crate::platform::SimdLevel;
        let op = LayerNorm::with_simd_level(vec![4], SimdLevel::None);
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let (mean, variance) = op.normalize_simd(&mut data, 1e-5);

        assert!((mean - 2.5).abs() < 0.0001);
        // Variance should be close to 1.0 after normalization
        let new_variance = op.compute_variance_simd(&data, op.compute_mean_simd(&data));
        assert!((new_variance - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_normalize_simd_with_affine() {
        use crate::platform::SimdLevel;
        let affine = LayerNormAffine::new()
            .with_weight(vec![1.0, 1.0, 1.0, 1.0])
            .with_bias(vec![0.0, 0.0, 0.0, 0.0]);
        let op = LayerNorm::with_simd_level(vec![4], SimdLevel::None).with_affine(affine);
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let (mean, _) = op.normalize_simd(&mut data, 1e-5);

        // Mean should be close to 0 after normalization
        assert!((mean - 2.5).abs() < 0.0001);
    }
}
