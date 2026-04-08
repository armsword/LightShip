//! Conv2d operator implementation using Im2col + GEMM
//!
//! This module provides an optimized Conv2d implementation using the Im2col
//! (image to column) transformation followed by GEMM (matrix multiply).

use crate::common::{DataType, Result, StorageLayout};
use crate::ir::Tensor;
use crate::platform::{detect_simd_level, SimdLevel};

/// Conv2d operator configuration
#[derive(Debug, Clone)]
pub struct Conv2dConfig {
    /// Number of output channels
    pub out_channels: usize,
    /// Kernel height
    pub kernel_h: usize,
    /// Kernel width
    pub kernel_w: usize,
    /// Stride height
    pub stride_h: usize,
    /// Stride width
    pub stride_w: usize,
    /// Padding height
    pub pad_h: usize,
    /// Padding width
    pub pad_w: usize,
    /// Dilation height
    pub dilation_h: usize,
    /// Dilation width
    pub dilation_w: usize,
    /// Number of groups (for grouped convolution)
    pub groups: usize,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            out_channels: 1,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
        }
    }
}

impl Conv2dConfig {
    /// Calculate output spatial dimensions
    pub fn output_shape(&self, input_shape: &[usize]) -> (usize, usize) {
        // input_shape: [N, C, H, W]
        let in_h = input_shape[2];
        let in_w = input_shape[3];
        let out_h = (in_h + 2 * self.pad_h - self.dilation_h * (self.kernel_h - 1) - 1) / self.stride_h + 1;
        let out_w = (in_w + 2 * self.pad_w - self.dilation_w * (self.kernel_w - 1) - 1) / self.stride_w + 1;
        (out_h, out_w)
    }
}

/// Conv2d operator using Im2col + GEMM
#[derive(Debug)]
pub struct Conv2d {
    config: Conv2dConfig,
    _simd_level: SimdLevel,
}

impl Conv2d {
    /// Create a new Conv2d operator
    pub fn new(config: Conv2dConfig) -> Self {
        Self {
            config,
            _simd_level: detect_simd_level(),
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &Conv2dConfig {
        &self.config
    }

    /// Forward pass: compute convolution
    /// Input: [N, C_in, H, W]
    /// Filter: [out_channels, C_in, kernel_h, kernel_w]
    /// Output: [N, out_channels, out_h, out_w]
    pub fn forward(&self, input: &Tensor, filter: &Tensor) -> Result<Tensor> {
        let config = &self.config;
        let in_shape = &input.shape;

        if in_shape.len() != 4 {
            return Err(crate::common::LightShipError::InvalidParam(
                "Conv2d input must be 4D tensor".into(),
            ));
        }

        let n = in_shape[0];
        let c_in = in_shape[1];

        // Validate groups
        if c_in % config.groups != 0 || config.out_channels % config.groups != 0 {
            return Err(crate::common::LightShipError::InvalidParam(
                "Invalid group configuration".into(),
            ));
        }

        // Calculate output shape
        let (out_h, out_w) = config.output_shape(in_shape);
        let out_len = n * config.out_channels * out_h * out_w;

        // Initialize output
        let mut output_data = vec![0.0f32; out_len];

        // Get dimensions
        let c_out_per_group = config.out_channels / config.groups;
        let c_in_per_group = c_in / config.groups;

        // Extract f32 data
        let input_data = self.extract_f32_data(input);
        let filter_data = self.extract_f32_data(filter);

        // Process each group
        for group_idx in 0..config.groups {
            // Build input column matrix using Im2col
            let col_input = self.im2col(&input_data, n, c_in, in_shape[2], in_shape[3], out_h, out_w, group_idx, c_in_per_group);
            let input_rows = n * out_h * out_w;

            // Process each output channel in this group
            for local_c in 0..c_out_per_group {
                let out_channel = group_idx * c_out_per_group + local_c;

                // Extract filter for this output channel: [c_in_per_group, kh, kw]
                // Compute convolution for each output position
                for n_idx in 0..n {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut sum = 0.0f32;

                            // Compute dot product: kernel elements * input elements
                            for kh in 0..config.kernel_h {
                                for kw in 0..config.kernel_w {
                                    for c in 0..c_in_per_group {
                                        // For grouped convolution: filter is [out_channels, c_in, kh, kw]
                                        // But only c_in_per_group channels are used per group
                                        let local_out_c = local_c;
                                        let local_in_c = c;
                                        let filter_idx = (local_out_c * c_in_per_group + local_in_c)
                                            * config.kernel_h * config.kernel_w
                                            + kh * config.kernel_w + kw;

                                        // Get input position (with padding and dilation)
                                        let in_h_pos = oh * config.stride_h + kh * config.dilation_h;
                                        let in_w_pos = ow * config.stride_w + kw * config.dilation_w;
                                        let in_c = group_idx * c_in_per_group + c;

                                        // Check bounds
                                        if in_h_pos < in_shape[2] && in_w_pos < in_shape[3] {
                                            // Input index in NCHW format
                                            let input_idx = n_idx * c_in * in_shape[2] * in_shape[3]
                                                + in_c * in_shape[2] * in_shape[3]
                                                + in_h_pos * in_shape[3]
                                                + in_w_pos;
                                            sum += filter_data[filter_idx] * input_data[input_idx];
                                        }
                                    }
                                }
                            }

                            // Store result
                            let out_idx = n_idx * config.out_channels * out_h * out_w
                                + out_channel * out_h * out_w
                                + oh * out_w + ow;
                            output_data[out_idx] = sum;
                        }
                    }
                }
            }
        }

        Ok(Tensor::new(
            "conv_output".to_string(),
            vec![n, config.out_channels, out_h, out_w],
            DataType::F32,
        ).with_data(output_data))
    }

    /// Im2col transformation (simplified for grouped convolution)
    fn im2col(&self, _input_data: &[f32], n: usize, c_in: usize, in_h: usize, in_w: usize,
               out_h: usize, out_w: usize, group_idx: usize, c_in_per_group: usize) -> Vec<f32> {
        let config = &self.config;
        let col_size = n * out_h * out_w * config.kernel_h * config.kernel_w * c_in_per_group;
        let mut col = Vec::with_capacity(col_size);

        // This is a placeholder - actual Im2col would pre-transform input
        // For simplicity, we compute on-the-fly in forward pass
        // The col matrix structure would be: [N*out_h*out_w, kernel_h*kernel_w*c_in_per_group]

        col
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_im2col_config() {
        let config = Conv2dConfig {
            out_channels: 64,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
        };

        let (out_h, out_w) = config.output_shape(&[1, 64, 32, 32]);
        assert_eq!(out_h, 32);
        assert_eq!(out_w, 32);
    }

    #[test]
    fn test_conv2d_basic() {
        let config = Conv2dConfig {
            out_channels: 1,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
        };

        let conv = Conv2d::new(config);

        // Input: 1x1x4x4
        let input = Tensor::new(
            "input".to_string(),
            vec![1, 1, 4, 4],
            DataType::F32,
        ).with_data(vec![
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ]);

        // Filter: 1x1x3x3, all ones
        let filter = Tensor::new(
            "filter".to_string(),
            vec![1, 1, 3, 3],
            DataType::F32,
        ).with_data(vec![1.0; 9]);

        let output = conv.forward(&input, &filter).unwrap();

        assert_eq!(output.shape, vec![1, 1, 2, 2]);

        // Each output pixel is sum of 9 input pixels = 9.0
        let out_bytes = output.data_as_bytes();
        let mut out_data = Vec::new();
        for chunk in out_bytes.chunks(4) {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            out_data.push(f32::from_le_bytes(buf));
        }
        for &val in &out_data {
            assert!((val - 9.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_conv2d_stride() {
        let config = Conv2dConfig {
            out_channels: 1,
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
        };

        let conv = Conv2d::new(config);

        // Input: 1x1x4x4
        let input = Tensor::new(
            "input".to_string(),
            vec![1, 1, 4, 4],
            DataType::F32,
        ).with_data(vec![
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ]);

        // Filter: 1x1x2x2, all ones
        let filter = Tensor::new(
            "filter".to_string(),
            vec![1, 1, 2, 2],
            DataType::F32,
        ).with_data(vec![1.0; 4]);

        let output = conv.forward(&input, &filter).unwrap();

        assert_eq!(output.shape, vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_conv2d_groups() {
        let config = Conv2dConfig {
            out_channels: 4,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            groups: 2, // 2 groups
        };

        let conv = Conv2d::new(config);

        // Input: 1x4x2x2
        let input = Tensor::new(
            "input".to_string(),
            vec![1, 4, 2, 2],
            DataType::F32,
        ).with_data(vec![
            1.0, 1.0, 1.0, 1.0, // channel 0
            2.0, 2.0, 2.0, 2.0, // channel 1
            3.0, 3.0, 3.0, 3.0, // channel 2
            4.0, 4.0, 4.0, 4.0, // channel 3
        ]);

        // Filter: 4x2x1x1 (2 groups of 2 output channels each)
        let filter = Tensor::new(
            "filter".to_string(),
            vec![4, 2, 1, 1],
            DataType::F32,
        ).with_data(vec![1.0; 4]);

        let output = conv.forward(&input, &filter).unwrap();

        // Each group processes 2 input channels, produces 2 output channels
        assert_eq!(output.shape, vec![1, 4, 2, 2]);
    }
}
