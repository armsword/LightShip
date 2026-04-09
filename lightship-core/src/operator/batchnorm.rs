//! Batch Normalization operator
//!
//! Implements Batch Normalization for neural networks.
//! BatchNorm(x) = gamma * (x - mean) / sqrt(variance + eps) + beta

use crate::platform::{detect_simd_level, div_scalar_simd, horizontal_sum, mul_scalar_simd, sub_scalar_simd, SimdLevel};
use crate::ir::Tensor;
use crate::common::{DataType, Result};
use std::fmt;

/// Batch Normalization operator
///
/// BatchNorm(x) = gamma * (x - mean) / sqrt(variance + eps) + beta
///
/// During training:
/// - Computes mean and variance over the batch
/// - Uses running mean/variance for inference
///
/// During inference:
/// - Uses precomputed mean and variance
#[derive(Debug)]
pub struct BatchNorm {
    /// Number of features (channels)
    pub num_features: usize,
    /// Momentum for running mean/variance
    pub momentum: f32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
    /// Training mode
    pub training: bool,
    /// Running mean (for inference)
    running_mean: Option<Vec<f32>>,
    /// Running variance (for inference)
    running_var: Option<Vec<f32>>,
    /// Gamma (scale) parameter
    gamma: Option<Vec<f32>>,
    /// Beta (bias) parameter
    beta: Option<Vec<f32>>,
    /// SIMD acceleration level
    simd_level: SimdLevel,
}

impl BatchNorm {
    /// Create a new BatchNorm operator
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            momentum: 0.9,
            epsilon: 1e-5,
            training: false,
            running_mean: None,
            running_var: None,
            gamma: None,
            beta: None,
            simd_level: detect_simd_level(),
        }
    }

    /// Create with epsilon
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Create with momentum
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set training mode
    pub fn training(mut self) -> Self {
        self.training = true;
        self
    }

    /// Set inference mode
    pub fn eval(mut self) -> Self {
        self.training = false;
        self
    }

    /// Set gamma (scale) parameter
    pub fn with_gamma(mut self, gamma: Vec<f32>) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Set beta (bias) parameter
    pub fn with_beta(mut self, beta: Vec<f32>) -> Self {
        self.beta = Some(beta);
        self
    }

    /// Set running statistics (for inference)
    pub fn with_running_stats(mut self, mean: Vec<f32>, var: Vec<f32>) -> Self {
        self.running_mean = Some(mean);
        self.running_var = Some(var);
        self
    }

    /// Create with specified SIMD level (for testing)
    pub fn with_simd_level(num_features: usize, simd_level: SimdLevel) -> Self {
        Self {
            num_features,
            momentum: 0.9,
            epsilon: 1e-5,
            training: false,
            running_mean: None,
            running_var: None,
            gamma: None,
            beta: None,
            simd_level,
        }
    }

    /// Compute batch norm on a single channel
    /// x_norm = (x - mean) / sqrt(var + eps)
    /// y = gamma * x_norm + beta
    pub fn compute_channel(&self, input: &[f32], mean: f32, variance: f32, gamma: f32, beta: f32) -> Vec<f32> {
        let len = input.len();
        let std_dev = (variance + self.epsilon).sqrt();
        let inv_std = 1.0 / std_dev;

        let mut output = Vec::with_capacity(len);
        for &x in input {
            let x_norm = (x - mean) * inv_std;
            output.push(gamma * x_norm + beta);
        }
        output
    }

    /// Compute mean over a slice
    fn compute_mean(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f32>() / data.len() as f32
    }

    /// Compute variance
    fn compute_variance(data: &[f32], mean: f32) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        let sum_sq_diff: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum();
        sum_sq_diff / data.len() as f32
    }

    /// Forward pass for inference (uses running statistics)
    ///
    /// Input: [N, C, H, W] or [N, C, D, H, W]
    /// Output: Same shape as input
    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor> {
        let shape = &input.shape;

        if shape.len() < 3 {
            return Err(crate::common::LightShipError::InvalidParam(
                "BatchNorm input must be at least 3D tensor".into(),
            ));
        }

        let n = shape[0];
        let c = shape[1];
        let spatial_size: usize = shape[2..].iter().product();

        if c != self.num_features {
            return Err(crate::common::LightShipError::InvalidParam(
                format!("Channel mismatch: expected {}, got {}", self.num_features, c),
            ));
        }

        // Extract data
        let input_data = self.extract_f32_data(input);
        let mut output_data = vec![0.0f32; input_data.len()];

        // Get gamma and beta, or use defaults
        let gamma_defaults = vec![1.0f32; c];
        let beta_defaults = vec![0.0f32; c];
        let gamma = self.gamma.as_ref().map(|g| g.as_slice()).unwrap_or(gamma_defaults.as_slice());
        let beta = self.beta.as_ref().map(|b| b.as_slice()).unwrap_or(beta_defaults.as_slice());

        // Get running statistics
        let mean_defaults = vec![0.0f32; c];
        let var_defaults = vec![1.0f32; c];
        let mean = self.running_mean.as_ref().map(|m| m.as_slice()).unwrap_or(mean_defaults.as_slice());
        let var = self.running_var.as_ref().map(|v| v.as_slice()).unwrap_or(var_defaults.as_slice());

        // Process each channel
        for c_idx in 0..c {
            let channel_start = c_idx * n * spatial_size;
            let channel_end = channel_start + n * spatial_size;
            let channel_data = &input_data[channel_start..channel_end];

            let mean_val = mean[c_idx];
            let var_val = var[c_idx];
            let gamma_val = gamma[c_idx];
            let beta_val = beta[c_idx];

            // Compute normalized output
            let std_dev = (var_val + self.epsilon).sqrt();
            let inv_std = 1.0 / std_dev;

            for (i, &x) in channel_data.iter().enumerate() {
                let x_norm = (x - mean_val) * inv_std;
                output_data[channel_start + i] = gamma_val * x_norm + beta_val;
            }
        }

        Ok(Tensor::new(
            "batchnorm_output".to_string(),
            shape.clone(),
            DataType::F32,
        ).with_data(output_data))
    }

    /// Forward pass for training (computes batch statistics)
    /// Note: Takes `&mut self` to update running statistics
    pub fn forward_training(&mut self, input: &Tensor) -> Result<(Tensor, f32, f32)> {
        let shape = &input.shape;

        if shape.len() < 3 {
            return Err(crate::common::LightShipError::InvalidParam(
                "BatchNorm input must be at least 3D tensor".into(),
            ));
        }

        let n = shape[0];
        let c = shape[1];
        let spatial_size: usize = shape[2..].iter().product();

        if c != self.num_features {
            return Err(crate::common::LightShipError::InvalidParam(
                format!("Channel mismatch: expected {}, got {}", self.num_features, c),
            ));
        }

        // Extract data
        let input_data = self.extract_f32_data(input);
        let mut output_data = vec![0.0f32; input_data.len()];

        // Get gamma and beta, or use defaults
        let gamma_defaults = vec![1.0f32; c];
        let beta_defaults = vec![0.0f32; c];
        let gamma = self.gamma.as_ref().map(|g| g.as_slice()).unwrap_or(gamma_defaults.as_slice());
        let beta = self.beta.as_ref().map(|b| b.as_slice()).unwrap_or(beta_defaults.as_slice());

        let mut batch_mean = vec![0.0f32; c];
        let mut batch_var = vec![0.0f32; c];

        // First pass: compute mean
        for c_idx in 0..c {
            let channel_start = c_idx * n * spatial_size;
            let channel_end = channel_start + n * spatial_size;
            let channel_data = &input_data[channel_start..channel_end];

            let mean_val = Self::compute_mean(channel_data);
            batch_mean[c_idx] = mean_val;
        }

        // Second pass: compute variance
        for c_idx in 0..c {
            let channel_start = c_idx * n * spatial_size;
            let channel_end = channel_start + n * spatial_size;
            let channel_data = &input_data[channel_start..channel_end];

            let var_val = Self::compute_variance(channel_data, batch_mean[c_idx]);
            batch_var[c_idx] = var_val;
        }

        // Third pass: normalize and scale
        for c_idx in 0..c {
            let channel_start = c_idx * n * spatial_size;
            let channel_end = channel_start + n * spatial_size;
            let channel_data = &input_data[channel_start..channel_end];

            let mean_val = batch_mean[c_idx];
            let var_val = batch_var[c_idx];
            let gamma_val = gamma[c_idx];
            let beta_val = beta[c_idx];

            let std_dev = (var_val + self.epsilon).sqrt();
            let inv_std = 1.0 / std_dev;

            for (i, &x) in channel_data.iter().enumerate() {
                let x_norm = (x - mean_val) * inv_std;
                output_data[channel_start + i] = gamma_val * x_norm + beta_val;
            }
        }

        // Update running statistics
        if let (Some(running_mean), Some(running_var)) = (&mut self.running_mean, &mut self.running_var) {
            for c_idx in 0..c {
                running_mean[c_idx] = self.momentum * running_mean[c_idx] + (1.0 - self.momentum) * batch_mean[c_idx];
                running_var[c_idx] = self.momentum * running_var[c_idx] + (1.0 - self.momentum) * batch_var[c_idx];
            }
        }

        // Compute overall mean (for return value)
        let overall_mean = batch_mean.iter().sum::<f32>() / c as f32;
        let overall_var = batch_var.iter().sum::<f32>() / c as f32;

        Ok((
            Tensor::new(
                "batchnorm_output".to_string(),
                shape.clone(),
                DataType::F32,
            ).with_data(output_data),
            overall_mean,
            overall_var,
        ))
    }

    /// Unified forward method
    /// Note: For training mode, takes `&mut self`
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        if self.training {
            Ok(self.forward_training(input)?.0)
        } else {
            self.forward_inference(input)
        }
    }

    /// Extract f32 data from tensor
    fn extract_f32_data(&self, tensor: &Tensor) -> Vec<f32> {
        let bytes = tensor.data_as_bytes();
        let count = bytes.len() / 4;
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&bytes[i * 4..(i + 1) * 4]);
            result.push(f32::from_le_bytes(buf));
        }
        result
    }

    /// Get running mean
    pub fn running_mean(&self) -> Option<&[f32]> {
        self.running_mean.as_ref().map(|v| v.as_slice())
    }

    /// Get running variance
    pub fn running_var(&self) -> Option<&[f32]> {
        self.running_var.as_ref().map(|v| v.as_slice())
    }
}

/// Helper trait to add with_data method to Tensor
trait TensorWithData {
    fn with_data(self, data: Vec<f32>) -> Tensor;
}

impl TensorWithData for Tensor {
    fn with_data(mut self, data: Vec<f32>) -> Tensor {
        let byte_size = data.len() * 4;
        let mut bytes = Vec::with_capacity(byte_size);
        for f in data {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        self.data = crate::ir::tensor::TensorData::Owned(bytes);
        self
    }
}

impl Default for BatchNorm {
    fn default() -> Self {
        Self::new(1)
    }
}

impl fmt::Display for BatchNorm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BatchNorm(features={}, eps={}, training={})",
            self.num_features, self.epsilon, self.training
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batchnorm_creation() {
        let mut bn = BatchNorm::new(64);
        assert_eq!(bn.num_features, 64);
        assert_eq!(bn.epsilon, 1e-5);
        assert!(!bn.training);
    }

    #[test]
    fn test_batchnorm_with_params() {
        let mut bn = BatchNorm::new(64)
            .with_epsilon(1e-3)
            .with_momentum(0.99)
            .training();

        assert_eq!(bn.epsilon, 1e-3);
        assert_eq!(bn.momentum, 0.99);
        assert!(bn.training);
    }

    #[test]
    fn test_batchnorm_with_gamma_beta() {
        let gamma = vec![1.0f32; 64];
        let beta = vec![0.0f32; 64];
        let mut bn = BatchNorm::new(64)
            .with_gamma(gamma)
            .with_beta(beta);

        assert!(bn.gamma.is_some());
        assert!(bn.beta.is_some());
    }

    #[test]
    fn test_batchnorm_inference_single_channel() {
        let mut bn = BatchNorm::new(1)
            .with_running_stats(vec![0.0], vec![1.0])
            .with_gamma(vec![1.0])
            .with_beta(vec![0.0]);

        // Input: 1x1x4, all zeros
        let input = Tensor::new(
            "input".to_string(),
            vec![1, 1, 4],
            DataType::F32,
        ).with_data(vec![0.0, 0.0, 0.0, 0.0]);

        let output = bn.forward(&input).unwrap();

        let out_bytes = output.data_as_bytes();
        let mut out_data = Vec::new();
        for chunk in out_bytes.chunks(4) {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            out_data.push(f32::from_le_bytes(buf));
        }

        // mean=0, var=1, gamma=1, beta=0
        // output = (x - 0) / sqrt(1 + eps) * 1 + 0 = x
        assert_eq!(out_data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_batchnorm_inference_nonzero_mean() {
        let mut bn = BatchNorm::new(1)
            .with_running_stats(vec![2.0], vec![1.0])
            .with_gamma(vec![1.0])
            .with_beta(vec![0.0]);

        // Input: 1x1x4 = [0, 1, 2, 3]
        let input = Tensor::new(
            "input".to_string(),
            vec![1, 1, 4],
            DataType::F32,
        ).with_data(vec![0.0, 1.0, 2.0, 3.0]);

        let output = bn.forward(&input).unwrap();

        let out_bytes = output.data_as_bytes();
        let mut out_data = Vec::new();
        for chunk in out_bytes.chunks(4) {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            out_data.push(f32::from_le_bytes(buf));
        }

        // mean=2, var=1, gamma=1, beta=0
        // output = (x - 2) / sqrt(1) = x - 2
        for (i, &val) in out_data.iter().enumerate() {
            let expected = (i as f32) - 2.0;
            assert!((val - expected).abs() < 0.001, "at index {}, got {}, expected {}", i, val, expected);
        }
    }

    #[test]
    fn test_batchnorm_inference_with_gamma_beta() {
        let mut bn = BatchNorm::new(1)
            .with_running_stats(vec![0.0], vec![1.0])
            .with_gamma(vec![2.0])
            .with_beta(vec![1.0]);

        // Input: 1x1x4 = [0, 1, 2, 3]
        let input = Tensor::new(
            "input".to_string(),
            vec![1, 1, 4],
            DataType::F32,
        ).with_data(vec![0.0, 1.0, 2.0, 3.0]);

        let output = bn.forward(&input).unwrap();

        let out_bytes = output.data_as_bytes();
        let mut out_data = Vec::new();
        for chunk in out_bytes.chunks(4) {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            out_data.push(f32::from_le_bytes(buf));
        }

        // mean=0, var=1, gamma=2, beta=1
        // output = (x - 0) / 1 * 2 + 1 = 2*x + 1
        for (i, &val) in out_data.iter().enumerate() {
            let expected = 2.0 * (i as f32) + 1.0;
            assert!((val - expected).abs() < 0.001, "at index {}, got {}, expected {}", i, val, expected);
        }
    }

    #[test]
    fn test_batchnorm_training() {
        let mut bn = BatchNorm::new(1)
            .with_gamma(vec![1.0])
            .with_beta(vec![0.0])
            .training();

        // Input: 1x1x4 = [0, 1, 2, 3]
        let input = Tensor::new(
            "input".to_string(),
            vec![1, 1, 4],
            DataType::F32,
        ).with_data(vec![0.0, 1.0, 2.0, 3.0]);

        let (output, mean, var) = bn.forward_training(&input).unwrap();

        let out_bytes = output.data_as_bytes();
        let mut out_data = Vec::new();
        for chunk in out_bytes.chunks(4) {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            out_data.push(f32::from_le_bytes(buf));
        }

        // mean = (0+1+2+3)/4 = 1.5
        assert!((mean - 1.5).abs() < 0.001);

        // var = ((0-1.5)^2 + (1-1.5)^2 + (2-1.5)^2 + (3-1.5)^2) / 4 = 1.25
        assert!((var - 1.25).abs() < 0.001);

        // Normalized: x_norm = (x - mean) / sqrt(var + eps)
        // [0, 1, 2, 3] -> [-1.5, -0.5, 0.5, 1.5] / sqrt(1.25) ≈ [-1.34, -0.45, 0.45, 1.34]
        // With gamma=1, beta=0, output should be normalized values
        for (i, &val) in out_data.iter().enumerate() {
            let x = i as f32;
            let expected = (x - 1.5) / 1.25_f32.sqrt(); // sqrt(1.25) ≈ 1.118
            assert!((val - expected).abs() < 0.01, "at index {}, got {}, expected {}", i, val, expected);
        }
    }

    #[test]
    fn test_batchnorm_multi_channel() {
        let mut bn = BatchNorm::new(2)
            .with_running_stats(vec![0.0, 1.0], vec![1.0, 4.0])
            .with_gamma(vec![1.0, 0.5])
            .with_beta(vec![0.0, 1.0]);

        // Input: 1x2x2
        // Channel 0: [0, 1]
        // Channel 1: [0, 1]
        let input = Tensor::new(
            "input".to_string(),
            vec![1, 2, 2],
            DataType::F32,
        ).with_data(vec![0.0, 1.0, 0.0, 1.0]);

        let output = bn.forward(&input).unwrap();

        let out_bytes = output.data_as_bytes();
        let mut out_data = Vec::new();
        for chunk in out_bytes.chunks(4) {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            out_data.push(f32::from_le_bytes(buf));
        }

        // Channel 0: mean=0, var=1, gamma=1, beta=0
        // Channel 0, idx 0: (0-0)/1 = 0
        // Channel 0, idx 1: (1-0)/1 = 1
        // Channel 1: mean=1, var=4, gamma=0.5, beta=1
        // Channel 1, idx 2: (0-1)*0.5/sqrt(4) + 1 = -0.5/2 + 1 = 0.75
        // Channel 1, idx 3: (1-1)*0.5/sqrt(4) + 1 = 0 + 1 = 1
        assert!((out_data[0] - 0.0).abs() < 0.001);
        assert!((out_data[1] - 1.0).abs() < 0.001);
        assert!((out_data[2] - 0.75).abs() < 0.001);
        assert!((out_data[3] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_batchnorm_batch_error() {
        let mut bn = BatchNorm::new(1);

        // Invalid input: 1D tensor
        let input = Tensor::new(
            "input".to_string(),
            vec![4],
            DataType::F32,
        ).with_data(vec![0.0, 1.0, 2.0, 3.0]);

        let result = bn.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_batchnorm_channel_mismatch() {
        let mut bn = BatchNorm::new(2);

        // Input has 1 channel but BatchNorm expects 2
        let input = Tensor::new(
            "input".to_string(),
            vec![1, 1, 4],
            DataType::F32,
        ).with_data(vec![0.0, 1.0, 2.0, 3.0]);

        let result = bn.forward(&input);
        assert!(result.is_err());
    }
}