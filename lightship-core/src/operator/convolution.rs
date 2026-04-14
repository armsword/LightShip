//! Conv2d operator implementation using Im2col + GEMM
//!
//! This module provides an optimized Conv2d implementation using the Im2col
//! (image to column) transformation followed by GEMM (matrix multiply).

use crate::common::{DataType, Result};
use crate::ir::Tensor;
use crate::platform::{detect_simd_level, gemm_simd, SimdLevel};
use std::sync::Arc;

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

    /// Forward pass: compute convolution (multi-threaded over batch dimension)
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

        if c_in % config.groups != 0 || config.out_channels % config.groups != 0 {
            return Err(crate::common::LightShipError::InvalidParam(
                "Invalid group configuration".into(),
            ));
        }

        let (out_h, out_w) = config.output_shape(in_shape);
        let batch_out_size = config.out_channels * out_h * out_w;

        let input_data = Arc::new(self.extract_f32_data(input));
        let filter_data = self.extract_f32_data(filter);

        let c_out_per_group = config.out_channels / config.groups;
        let c_in_per_group = c_in / config.groups;

        // Pre-compute filter matrices for each group (shared across all batch elements)
        let filter_matrices: Arc<Vec<Vec<f32>>> = Arc::new(
            (0..config.groups)
                .map(|g| self.reshape_filter(&filter_data, g, c_out_per_group, c_in_per_group))
                .collect()
        );

        let mut output_data = vec![0.0f32; n * batch_out_size];

        // Split output buffer into per-batch chunks; each thread writes exclusively to its chunk
        let chunks: Vec<&mut [f32]> = output_data.chunks_mut(batch_out_size).collect();

        std::thread::scope(|s| {
            for (n_idx, batch_out) in chunks.into_iter().enumerate() {
                let input_data = Arc::clone(&input_data);
                let filter_matrices = Arc::clone(&filter_matrices);
                let simd_level = self.simd_level;

                s.spawn(move || {
                    Self::compute_batch_element(
                        n_idx, batch_out,
                        &input_data, &filter_matrices,
                        c_in, in_h, in_w, out_h, out_w,
                        c_in_per_group, c_out_per_group,
                        config,
                        simd_level,
                    );
                });
            }
        });

        Ok(Tensor::new(
            "conv_output".to_string(),
            vec![n, config.out_channels, out_h, out_w],
            DataType::F32,
        ).with_data(output_data))
    }

    /// Compute convolution for a single batch element.
    /// Writes NCHW-format results into `batch_out` (size: out_channels * out_h * out_w).
    #[allow(clippy::too_many_arguments)]
    fn compute_batch_element(
        n_idx: usize,
        batch_out: &mut [f32],
        input_data: &[f32],
        filter_matrices: &[Vec<f32>],
        c_in: usize, in_h: usize, in_w: usize,
        out_h: usize, out_w: usize,
        c_in_per_group: usize, c_out_per_group: usize,
        config: &Conv2dConfig,
        simd_level: SimdLevel,
    ) {
        let kernel_size = config.kernel_h * config.kernel_w * c_in_per_group;

        for group_idx in 0..config.groups {
            let filter_matrix = &filter_matrices[group_idx];

            // Build im2col matrix: [kernel_size, out_h*out_w]
            // Layout: for each (oh, ow): append the flattened kernel window
            let mut col = vec![0.0f32; kernel_size * out_h * out_w];
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let col_base = (oh * out_w + ow) * kernel_size;
                    let mut ki = 0;
                    for kh in 0..config.kernel_h {
                        for kw in 0..config.kernel_w {
                            for c in 0..c_in_per_group {
                                // Compute padded input coordinates
                                let in_h_idx = (oh * config.stride_h + kh * config.dilation_h)
                                    .wrapping_sub(config.pad_h);
                                let in_w_idx = (ow * config.stride_w + kw * config.dilation_w)
                                    .wrapping_sub(config.pad_w);
                                let in_c = group_idx * c_in_per_group + c;
                                let valid = in_h_idx < in_h && in_w_idx < in_w;
                                col[col_base + ki] = if valid {
                                    input_data[n_idx * c_in * in_h * in_w
                                        + in_c * in_h * in_w
                                        + in_h_idx * in_w
                                        + in_w_idx]
                                } else {
                                    0.0
                                };
                                ki += 1;
                            }
                        }
                    }
                }
            }

            // Transpose col: [out_h*out_w, kernel_size] → [kernel_size, out_h*out_w]
            let out_pixels = out_h * out_w;
            let mut col_t = vec![0.0f32; kernel_size * out_pixels];
            for i in 0..kernel_size {
                for j in 0..out_pixels {
                    col_t[i * out_pixels + j] = col[j * kernel_size + i];
                }
            }

            // GEMM: [c_out_per_group, out_pixels] = filter_matrix [c_out_per_group, kernel_size]
            //                                        @ col_t [kernel_size, out_pixels]
            let mut out_slice = vec![0.0f32; c_out_per_group * out_pixels];
            gemm_simd(
                filter_matrix,
                &col_t,
                &mut out_slice,
                c_out_per_group,
                out_pixels,
                kernel_size,
                simd_level,
            );

            // Scatter into batch_out (NCHW layout within the batch chunk)
            for out_c in 0..c_out_per_group {
                let global_out_c = group_idx * c_out_per_group + out_c;
                let dst_base = global_out_c * out_pixels;
                let src_base = out_c * out_pixels;
                batch_out[dst_base..dst_base + out_pixels]
                    .copy_from_slice(&out_slice[src_base..src_base + out_pixels]);
            }
        }
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

    #[test]
    fn test_conv2d_batch_parallel() {
        // Test batch_size=4 to verify multi-threaded correctness
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

        // Input: 4x1x4x4 (batch of 4)
        let mut input_data = Vec::new();
        for _ in 0..4 {
            input_data.extend_from_slice(&[1.0f32; 16]); // all-ones 4x4 frame
        }
        let input = Tensor::new(
            "input".to_string(),
            vec![4, 1, 4, 4],
            DataType::F32,
        ).with_data(input_data);

        // Filter: 1x1x3x3, all ones
        let filter = Tensor::new(
            "filter".to_string(),
            vec![1, 1, 3, 3],
            DataType::F32,
        ).with_data(vec![1.0; 9]);

        let output = conv.forward(&input, &filter).unwrap();

        // Each batch's output shape: 1x2x2, value 9.0
        assert_eq!(output.shape, vec![4, 1, 2, 2]);
        let out_bytes = output.data_as_bytes();
        let out_data: Vec<f32> = out_bytes.chunks(4).map(|c| {
            f32::from_le_bytes([c[0], c[1], c[2], c[3]])
        }).collect();
        assert_eq!(out_data.len(), 16);
        for &val in &out_data {
            assert!((val - 9.0).abs() < 1e-5, "expected 9.0 but got {}", val);
        }
    }
}
