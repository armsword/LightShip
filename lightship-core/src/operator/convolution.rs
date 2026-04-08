//! Conv2d operator implementation using Im2col + GEMM
//!
//! This module provides an optimized Conv2d implementation using the Im2col
//! (image to column) transformation followed by GEMM (matrix multiply).

use crate::common::{DataType, Result, StorageLayout};
use crate::ir::Tensor;
use crate::platform::{detect_simd_level, gemm_simd, SimdLevel};

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
    simd_level: SimdLevel,
}

impl Conv2d {
    /// Create a new Conv2d operator
    pub fn new(config: Conv2dConfig) -> Self {
        Self {
            config,
            simd_level: detect_simd_level(),
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
        let in_h = in_shape[2];
        let in_w = in_shape[3];

        // Validate groups
        if c_in % config.groups != 0 || config.out_channels % config.groups != 0 {
            return Err(crate::common::LightShipError::InvalidParam(
                "Invalid group configuration".into(),
            ));
        }

        // Calculate output shape
        let (out_h, out_w) = config.output_shape(in_shape);

        // Initialize output
        let out_len = n * config.out_channels * out_h * out_w;
        let mut output_data = vec![0.0f32; out_len];

        // Get dimensions
        let c_out_per_group = config.out_channels / config.groups;
        let c_in_per_group = c_in / config.groups;
        let kernel_size = config.kernel_h * config.kernel_w * c_in_per_group;

        // Extract f32 data
        let input_data = self.extract_f32_data(input);
        let filter_data = self.extract_f32_data(filter);

        // Process each group
        for group_idx in 0..config.groups {
            // Build Im2col matrix for this group
            // Col matrix: [out_h * out_w, kernel_h * kernel_w * c_in_per_group]
            let col_matrix = self.im2col(
                &input_data, n, c_in, in_h, in_w,
                out_h, out_w, group_idx, c_in_per_group
            );

            // Filter matrix for this group: [c_out_per_group, kernel_size]
            // Reshape filter from [out_channels, c_in, kh, kw] to [c_out_per_group, kernel_size]
            let filter_matrix = self.reshape_filter(
                &filter_data,
                group_idx, c_out_per_group, c_in_per_group
            );

            // For each batch element, compute GEMM: output_slice = filter_matrix @ col_matrix.T
            // But since we have multiple output channels, we do:
            // [c_out_per_group, out_h*out_w] = [c_out_per_group, kernel_size] @ [kernel_size, out_h*out_w]
            for n_idx in 0..n {
                // Extract column slice for this batch element
                // Col is organized as: [n_idx][oh][ow][kh][kw][c] -> linearized
                // We want: [kh*kw*c_in_per_group][out_h*out_w]
                let mut col_slice = vec![0.0f32; kernel_size * out_h * out_w];
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        for kh in 0..config.kernel_h {
                            for kw in 0..config.kernel_w {
                                for c in 0..c_in_per_group {
                                    let col_idx = oh * out_w * kernel_size + ow * kernel_size +
                                        kh * config.kernel_w * c_in_per_group +
                                        kw * c_in_per_group + c;
                                    // Input access index
                                    let in_h_pos = oh * config.stride_h + kh * config.dilation_h;
                                    let in_w_pos = ow * config.stride_w + kw * config.dilation_w;
                                    let in_c = group_idx * c_in_per_group + c;
                                    let in_idx = n_idx * c_in * in_h * in_w +
                                        in_c * in_h * in_w +
                                        in_h_pos * in_w +
                                        in_w_pos;
                                    let valid = in_h_pos < in_h && in_w_pos < in_w;
                                    col_slice[col_idx] = if valid { input_data[in_idx] } else { 0.0 };
                                }
                            }
                        }
                    }
                }

                // Transpose col_slice: from [out_h*out_w][kernel_size] to [kernel_size][out_h*out_w]
                let mut col_transposed = vec![0.0f32; kernel_size * out_h * out_w];
                for i in 0..kernel_size {
                    for j in 0..out_h * out_w {
                        col_transposed[i * out_h * out_w + j] = col_slice[j * kernel_size + i];
                    }
                }

                // GEMM: output[n_idx] = filter_matrix @ col_transposed
                // output: [c_out_per_group, out_h*out_w]
                let mut output_slice = vec![0.0f32; c_out_per_group * out_h * out_w];
                gemm_simd(
                    &filter_matrix,
                    &col_transposed,
                    &mut output_slice,
                    c_out_per_group, // M
                    out_h * out_w,    // N
                    kernel_size,      // K
                    self.simd_level,
                );

                // Copy to output tensor (NCHW format)
                for out_c in 0..c_out_per_group {
                    let global_out_c = group_idx * c_out_per_group + out_c;
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let out_idx = n_idx * config.out_channels * out_h * out_w +
                                global_out_c * out_h * out_w +
                                oh * out_w + ow;
                            let gemm_idx = out_c * out_h * out_w + oh * out_w + ow;
                            output_data[out_idx] = output_slice[gemm_idx];
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

    /// Im2col transformation: convert input to column matrix
    fn im2col(&self, input_data: &[f32], n: usize, c_in: usize, in_h: usize, in_w: usize,
               out_h: usize, out_w: usize, group_idx: usize, c_in_per_group: usize) -> Vec<f32> {
        let config = &self.config;
        let kernel_size = config.kernel_h * config.kernel_w * c_in_per_group;
        let col_size = n * out_h * out_w * kernel_size;
        let mut col = Vec::with_capacity(col_size);

        // For each batch element
        for n_idx in 0..n {
            // For each output position
            for oh in 0..out_h {
                for ow in 0..out_w {
                    // Kernel sliding window
                    for kh in 0..config.kernel_h {
                        for kw in 0..config.kernel_w {
                            for c in 0..c_in_per_group {
                                let in_h_pos = oh * config.stride_h + kh * config.dilation_h - config.pad_h;
                                let in_w_pos = ow * config.stride_w + kw * config.dilation_w - config.pad_w;
                                let in_c = group_idx * c_in_per_group + c;

                                let value = if in_h_pos < in_h && in_w_pos < in_w && in_h_pos >= 0 && in_w_pos >= 0 {
                                    let in_idx = n_idx * c_in * in_h * in_w +
                                        in_c * in_h * in_w +
                                        in_h_pos * in_w +
                                        in_w_pos;
                                    input_data[in_idx]
                                } else {
                                    0.0
                                };
                                col.push(value);
                            }
                        }
                    }
                }
            }
        }

        col
    }

    /// Reshape filter for grouped convolution
    /// Filter: [out_channels, c_in_total/groups, kh, kw] (per group)
    /// Output: [c_out_per_group, kernel_size]
    fn reshape_filter(&self, filter_data: &[f32], group_idx: usize, c_out_per_group: usize, c_in_per_group: usize) -> Vec<f32> {
        let config = &self.config;
        let kernel_size = config.kernel_h * config.kernel_w * c_in_per_group;
        let mut result = Vec::with_capacity(c_out_per_group * kernel_size);

        for out_c in 0..c_out_per_group {
            let global_out_c = group_idx * c_out_per_group + out_c;
            for c in 0..c_in_per_group {
                for kh in 0..config.kernel_h {
                    for kw in 0..config.kernel_w {
                        // Filter layout: [out_channels, c_in_per_group, kh, kw]
                        // Each group has its own slice of output channels
                        let filter_idx = global_out_c * c_in_per_group * config.kernel_h * config.kernel_w +
                            c * config.kernel_h * config.kernel_w +
                            kh * config.kernel_w + kw;
                        result.push(filter_data[filter_idx]);
                    }
                }
            }
        }

        result
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
            groups: 2,
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
        // Shape: [out_channels=4, c_in=2, kh=1, kw=1] = 8 elements
        let filter = Tensor::new(
            "filter".to_string(),
            vec![4, 2, 1, 1],
            DataType::F32,
        ).with_data(vec![1.0; 8]);

        let output = conv.forward(&input, &filter).unwrap();

        assert_eq!(output.shape, vec![1, 4, 2, 2]);
    }
}
