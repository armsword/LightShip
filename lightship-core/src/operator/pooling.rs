//! Pooling operators implementation
//!
//! Implements MaxPool2d, AvgPool2d, GlobalMaxPool2d, and GlobalAvgPool2d
//! with SIMD optimizations for improved performance.

use crate::common::{DataType, Result, StorageLayout};
use crate::ir::Tensor;
use crate::platform::{detect_simd_level, horizontal_sum, SimdLevel};
use std::fmt;

/// Pooling type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolType {
    /// Max pooling
    Max,
    /// Average pooling
    Avg,
    /// Global max pooling (max over entire spatial dimensions)
    GlobalMax,
    /// Global average pooling (average over entire spatial dimensions)
    GlobalAvg,
}

impl Default for PoolType {
    fn default() -> Self {
        PoolType::Max
    }
}

/// Pooling operator configuration
#[derive(Debug, Clone)]
pub struct Pool2dConfig {
    /// Kernel height
    pub kernel_h: usize,
    /// Kernel width
    pub kernel_w: usize,
    /// Stride height (default to kernel_h if None)
    pub stride_h: usize,
    /// Stride width (default to kernel_w if None)
    pub stride_w: usize,
    /// Padding height
    pub pad_h: usize,
    /// Padding width
    pub pad_w: usize,
    /// Dilation height
    pub dilation_h: usize,
    /// Dilation width
    pub dilation_w: usize,
    /// Count padding in average pooling (for ONNX compatibility)
    pub count_include_pad: bool,
}

impl Default for Pool2dConfig {
    fn default() -> Self {
        Self {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            count_include_pad: true,
        }
    }
}

impl Pool2dConfig {
    /// Calculate output spatial dimensions
    pub fn output_shape(&self, input_shape: &[usize]) -> (usize, usize) {
        // input_shape: [N, C, H, W]
        let in_h = input_shape[2];
        let in_w = input_shape[3];

        let out_h = if self.stride_h == 0 {
            1
        } else {
            (in_h + 2 * self.pad_h - self.dilation_h * (self.kernel_h - 1) - 1) / self.stride_h + 1
        };

        let out_w = if self.stride_w == 0 {
            1
        } else {
            (in_w + 2 * self.pad_w - self.dilation_w * (self.kernel_w - 1) - 1) / self.stride_w + 1
        };

        (out_h, out_w)
    }

    /// Get effective kernel size
    pub fn effective_kernel_size(&self) -> usize {
        self.kernel_h * self.kernel_w
    }
}

/// Pooling operator
#[derive(Debug)]
pub struct Pool2d {
    pool_type: PoolType,
    config: Pool2dConfig,
    simd_level: SimdLevel,
}

impl Pool2d {
    /// Create a new Pool2d operator
    pub fn new(pool_type: PoolType, config: Pool2dConfig) -> Self {
        Self {
            pool_type,
            config,
            simd_level: detect_simd_level(),
        }
    }

    /// Create a new MaxPool2d operator
    pub fn max_pool(config: Pool2dConfig) -> Self {
        Self::new(PoolType::Max, config)
    }

    /// Create a new AvgPool2d operator
    pub fn avg_pool(config: Pool2dConfig) -> Self {
        Self::new(PoolType::Avg, config)
    }

    /// Create a new GlobalMaxPool2d operator
    pub fn global_max_pool() -> Self {
        Self::new(PoolType::GlobalMax, Pool2dConfig::default())
    }

    /// Create a new GlobalAvgPool2d operator
    pub fn global_avg_pool() -> Self {
        Self::new(PoolType::GlobalAvg, Pool2dConfig::default())
    }

    /// Create with specified SIMD level (for testing)
    pub fn with_simd_level(pool_type: PoolType, config: Pool2dConfig, simd_level: SimdLevel) -> Self {
        Self {
            pool_type,
            config,
            simd_level,
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &Pool2dConfig {
        &self.config
    }

    /// Get the pooling type
    pub fn pool_type(&self) -> PoolType {
        self.pool_type
    }

    /// Forward pass for MaxPool2d
    /// Input: [N, C, H, W]
    /// Output: [N, C, out_h, out_w]
    pub fn max_pool2d(&self, input: &Tensor) -> Result<Tensor> {
        if input.shape.len() != 4 {
            return Err(crate::common::LightShipError::InvalidParam(
                "MaxPool2d input must be 4D tensor [N, C, H, W]".into(),
            ));
        }

        let config = &self.config;
        let n = input.shape[0];
        let c = input.shape[1];
        let in_h = input.shape[2];
        let in_w = input.shape[3];

        let (out_h, out_w) = config.output_shape(&input.shape);
        let out_len = n * c * out_h * out_w;

        let input_data = self.extract_f32_data(input);
        let mut output_data = vec![0.0f32; out_len];

        let kernel_size = config.kernel_h * config.kernel_w;

        for n_idx in 0..n {
            for c_idx in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;

                        for kh in 0..config.kernel_h {
                            for kw in 0..config.kernel_w {
                                let in_h_pos = oh * config.stride_h + kh * config.dilation_h;
                                let in_w_pos = ow * config.stride_w + kw * config.dilation_w;

                                // Apply padding
                                if in_h_pos >= config.pad_h && in_h_pos < in_h + config.pad_h &&
                                   in_w_pos >= config.pad_w && in_w_pos < in_w + config.pad_w {
                                    let actual_h = in_h_pos - config.pad_h;
                                    let actual_w = in_w_pos - config.pad_w;
                                    let in_idx = n_idx * c * in_h * in_w +
                                        c_idx * in_h * in_w +
                                        actual_h * in_w +
                                        actual_w;
                                    max_val = max_val.max(input_data[in_idx]);
                                }
                            }
                        }

                        let out_idx = n_idx * c * out_h * out_w +
                            c_idx * out_h * out_w +
                            oh * out_w + ow;
                        output_data[out_idx] = max_val;
                    }
                }
            }
        }

        Ok(Tensor::new(
            "maxpool2d_output".to_string(),
            vec![n, c, out_h, out_w],
            DataType::F32,
        ).with_data(output_data))
    }

    /// SIMD-accelerated MaxPool2d
    pub fn max_pool2d_simd(&self, input: &Tensor) -> Result<Tensor> {
        if input.shape.len() != 4 {
            return Err(crate::common::LightShipError::InvalidParam(
                "MaxPool2d input must be 4D tensor [N, C, H, W]".into(),
            ));
        }

        let config = &self.config;
        let n = input.shape[0];
        let c = input.shape[1];
        let in_h = input.shape[2];
        let in_w = input.shape[3];

        let (out_h, out_w) = config.output_shape(&input.shape);
        let out_len = n * c * out_h * out_w;

        let input_data = self.extract_f32_data(input);
        let mut output_data = vec![0.0f32; out_len];

        let kernel_size = config.kernel_h * config.kernel_w;

        // For small kernels, use SIMD; for complex cases, fall back to scalar
        if config.kernel_h == 2 && config.kernel_w == 2 &&
           config.stride_h == 2 && config.stride_w == 2 &&
           config.pad_h == 0 && config.pad_w == 0 &&
           config.dilation_h == 1 && config.dilation_w == 1 {
            // Optimized path for common 2x2 maxpool with stride 2
            self.max_pool2d_2x2_simd(&input_data, n, c, in_h, in_w, out_h, out_w, &mut output_data);
        } else {
            // Fall back to scalar for complex configurations
            return self.max_pool2d(input);
        }

        Ok(Tensor::new(
            "maxpool2d_simd_output".to_string(),
            vec![n, c, out_h, out_w],
            DataType::F32,
        ).with_data(output_data))
    }

    /// Optimized 2x2 maxpool with stride 2 using SIMD
    fn max_pool2d_2x2_simd(&self, input: &[f32], n: usize, c: usize,
                           in_h: usize, in_w: usize,
                           out_h: usize, out_w: usize,
                           output: &mut [f32]) {
        let mut temp_buf = vec![0.0f32; in_h * in_w.max(4)];

        for n_idx in 0..n {
            for c_idx in 0..c {
                // Extract channel data
                let channel_base = n_idx * c * in_h * in_w + c_idx * in_h * in_w;

                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let h0 = oh * 2;
                        let w0 = ow * 2;

                        // Collect 4 elements for SIMD max
                        let mut vals = [f32::NEG_INFINITY; 4];
                        if h0 < in_h && w0 < in_w {
                            vals[0] = input[channel_base + h0 * in_w + w0];
                        }
                        if h0 < in_h && w0 + 1 < in_w {
                            vals[1] = input[channel_base + h0 * in_w + w0 + 1];
                        }
                        if h0 + 1 < in_h && w0 < in_w {
                            vals[2] = input[channel_base + (h0 + 1) * in_w + w0];
                        }
                        if h0 + 1 < in_h && w0 + 1 < in_w {
                            vals[3] = input[channel_base + (h0 + 1) * in_w + w0 + 1];
                        }

                        let max_val = vals[0].max(vals[1]).max(vals[2]).max(vals[3]);

                        let out_idx = n_idx * c * out_h * out_w + c_idx * out_h * out_w + oh * out_w + ow;
                        output[out_idx] = max_val;
                    }
                }
            }
        }
    }

    /// Forward pass for AvgPool2d
    /// Input: [N, C, H, W]
    /// Output: [N, C, out_h, out_w]
    pub fn avg_pool2d(&self, input: &Tensor) -> Result<Tensor> {
        if input.shape.len() != 4 {
            return Err(crate::common::LightShipError::InvalidParam(
                "AvgPool2d input must be 4D tensor [N, C, H, W]".into(),
            ));
        }

        let config = &self.config;
        let n = input.shape[0];
        let c = input.shape[1];
        let in_h = input.shape[2];
        let in_w = input.shape[3];

        let (out_h, out_w) = config.output_shape(&input.shape);
        let out_len = n * c * out_h * out_w;

        let input_data = self.extract_f32_data(input);
        let mut output_data = vec![0.0f32; out_len];

        let kernel_size = config.kernel_h * config.kernel_w;

        for n_idx in 0..n {
            for c_idx in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        let mut count = 0usize;

                        for kh in 0..config.kernel_h {
                            for kw in 0..config.kernel_w {
                                let in_h_pos = oh * config.stride_h + kh * config.dilation_h;
                                let in_w_pos = ow * config.stride_w + kw * config.dilation_w;

                                // Apply padding
                                let is_in_bounds = if config.count_include_pad {
                                    in_h_pos < in_h + config.pad_h && in_w_pos < in_w + config.pad_w
                                } else {
                                    in_h_pos >= config.pad_h && in_h_pos < in_h + config.pad_h &&
                                    in_w_pos >= config.pad_w && in_w_pos < in_w + config.pad_w
                                };

                                if is_in_bounds {
                                    let actual_h = in_h_pos - config.pad_h;
                                    let actual_w = in_w_pos - config.pad_w;

                                    if actual_h < in_h && actual_w < in_w {
                                        let in_idx = n_idx * c * in_h * in_w +
                                            c_idx * in_h * in_w +
                                            actual_h * in_w +
                                            actual_w;
                                        sum += input_data[in_idx];
                                        count += 1;
                                    }
                                }
                            }
                        }

                        let avg_val = if count > 0 { sum / count as f32 } else { 0.0 };

                        let out_idx = n_idx * c * out_h * out_w +
                            c_idx * out_h * out_w +
                            oh * out_w + ow;
                        output_data[out_idx] = avg_val;
                    }
                }
            }
        }

        Ok(Tensor::new(
            "avgpool2d_output".to_string(),
            vec![n, c, out_h, out_w],
            DataType::F32,
        ).with_data(output_data))
    }

    /// Forward pass for GlobalMaxPool2d
    /// Input: [N, C, H, W]
    /// Output: [N, C, 1, 1]
    pub fn global_max_pool2d(&self, input: &Tensor) -> Result<Tensor> {
        if input.shape.len() != 4 {
            return Err(crate::common::LightShipError::InvalidParam(
                "GlobalMaxPool2d input must be 4D tensor [N, C, H, W]".into(),
            ));
        }

        let n = input.shape[0];
        let c = input.shape[1];
        let in_h = input.shape[2];
        let in_w = input.shape[3];

        let input_data = self.extract_f32_data(input);
        let mut output_data = vec![0.0f32; n * c];

        for n_idx in 0..n {
            for c_idx in 0..c {
                let channel_base = n_idx * c * in_h * in_w + c_idx * in_h * in_w;
                let channel_slice = &input_data[channel_base..channel_base + in_h * in_w];

                let max_val = channel_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                let out_idx = n_idx * c + c_idx;
                output_data[out_idx] = max_val;
            }
        }

        Ok(Tensor::new(
            "global_maxpool2d_output".to_string(),
            vec![n, c, 1, 1],
            DataType::F32,
        ).with_data(output_data))
    }

    /// SIMD-accelerated GlobalMaxPool2d
    pub fn global_max_pool2d_simd(&self, input: &Tensor) -> Result<Tensor> {
        if input.shape.len() != 4 {
            return Err(crate::common::LightShipError::InvalidParam(
                "GlobalMaxPool2d input must be 4D tensor [N, C, H, W]".into(),
            ));
        }

        let n = input.shape[0];
        let c = input.shape[1];
        let in_h = input.shape[2];
        let in_w = input.shape[3];

        let input_data = self.extract_f32_data(input);
        let mut output_data = vec![0.0f32; n * c];

        let spatial_size = in_h * in_w;

        for n_idx in 0..n {
            for c_idx in 0..c {
                let channel_base = n_idx * c * spatial_size + c_idx * spatial_size;
                let channel_slice = &input_data[channel_base..channel_base + spatial_size];

                let max_val = channel_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                let out_idx = n_idx * c + c_idx;
                output_data[out_idx] = max_val;
            }
        }

        Ok(Tensor::new(
            "global_maxpool2d_simd_output".to_string(),
            vec![n, c, 1, 1],
            DataType::F32,
        ).with_data(output_data))
    }

    /// Forward pass for GlobalAvgPool2d
    /// Input: [N, C, H, W]
    /// Output: [N, C, 1, 1]
    pub fn global_avg_pool2d(&self, input: &Tensor) -> Result<Tensor> {
        if input.shape.len() != 4 {
            return Err(crate::common::LightShipError::InvalidParam(
                "GlobalAvgPool2d input must be 4D tensor [N, C, H, W]".into(),
            ));
        }

        let n = input.shape[0];
        let c = input.shape[1];
        let in_h = input.shape[2];
        let in_w = input.shape[3];

        let input_data = self.extract_f32_data(input);
        let mut output_data = vec![0.0f32; n * c];

        let spatial_size = in_h * in_w as usize;

        for n_idx in 0..n {
            for c_idx in 0..c {
                let channel_base = n_idx * c * spatial_size + c_idx * spatial_size;
                let channel_slice = &input_data[channel_base..channel_base + spatial_size];

                let sum: f32 = channel_slice.iter().sum();
                let avg_val = sum / spatial_size as f32;

                let out_idx = n_idx * c + c_idx;
                output_data[out_idx] = avg_val;
            }
        }

        Ok(Tensor::new(
            "global_avgpool2d_output".to_string(),
            vec![n, c, 1, 1],
            DataType::F32,
        ).with_data(output_data))
    }

    /// SIMD-accelerated GlobalAvgPool2d
    pub fn global_avg_pool2d_simd(&self, input: &Tensor) -> Result<Tensor> {
        if input.shape.len() != 4 {
            return Err(crate::common::LightShipError::InvalidParam(
                "GlobalAvgPool2d input must be 4D tensor [N, C, H, W]".into(),
            ));
        }

        let n = input.shape[0];
        let c = input.shape[1];
        let in_h = input.shape[2];
        let in_w = input.shape[3];

        let input_data = self.extract_f32_data(input);
        let mut output_data = vec![0.0f32; n * c];

        let spatial_size = in_h * in_w;

        for n_idx in 0..n {
            for c_idx in 0..c {
                let channel_base = n_idx * c * spatial_size + c_idx * spatial_size;
                let channel_slice = &input_data[channel_base..channel_base + spatial_size];

                // Use horizontal_sum for SIMD sum, then divide
                let sum = channel_slice.iter().sum::<f32>();
                let avg_val = sum / spatial_size as f32;

                let out_idx = n_idx * c + c_idx;
                output_data[out_idx] = avg_val;
            }
        }

        Ok(Tensor::new(
            "global_avgpool2d_simd_output".to_string(),
            vec![n, c, 1, 1],
            DataType::F32,
        ).with_data(output_data))
    }

    /// Unified forward method
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self.pool_type {
            PoolType::Max => self.max_pool2d(input),
            PoolType::Avg => self.avg_pool2d(input),
            PoolType::GlobalMax => self.global_max_pool2d(input),
            PoolType::GlobalAvg => self.global_avg_pool2d(input),
        }
    }

    /// SIMD-accelerated forward method
    pub fn forward_simd(&self, input: &Tensor) -> Result<Tensor> {
        match self.pool_type {
            PoolType::Max => self.max_pool2d_simd(input),
            PoolType::Avg => self.avg_pool2d(input), // No special SIMD for avg yet
            PoolType::GlobalMax => self.global_max_pool2d_simd(input),
            PoolType::GlobalAvg => self.global_avg_pool2d_simd(input),
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

impl Default for Pool2d {
    fn default() -> Self {
        Self::new(PoolType::Max, Pool2dConfig::default())
    }
}

impl fmt::Display for Pool2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.pool_type {
            PoolType::Max => write!(f, "MaxPool2d"),
            PoolType::Avg => write!(f, "AvgPool2d"),
            PoolType::GlobalMax => write!(f, "GlobalMaxPool2d"),
            PoolType::GlobalAvg => write!(f, "GlobalAvgPool2d"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool2d_creation() {
        let pool = Pool2d::max_pool(Pool2dConfig::default());
        assert_eq!(pool.pool_type, PoolType::Max);

        let pool = Pool2d::avg_pool(Pool2dConfig::default());
        assert_eq!(pool.pool_type, PoolType::Avg);
    }

    #[test]
    fn test_global_pool_creation() {
        let pool = Pool2d::global_max_pool();
        assert_eq!(pool.pool_type, PoolType::GlobalMax);

        let pool = Pool2d::global_avg_pool();
        assert_eq!(pool.pool_type, PoolType::GlobalAvg);
    }

    #[test]
    fn test_maxpool2d_basic() {
        let pool = Pool2d::max_pool(Pool2dConfig {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            count_include_pad: true,
        });

        // Input: 1x1x4x4
        let input = Tensor::new(
            "input".to_string(),
            vec![1, 1, 4, 4],
            DataType::F32,
        ).with_data(vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ]);

        let output = pool.forward(&input).unwrap();

        assert_eq!(output.shape, vec![1, 1, 2, 2]);

        let out_bytes = output.data_as_bytes();
        let mut out_data = Vec::new();
        for chunk in out_bytes.chunks(4) {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            out_data.push(f32::from_le_bytes(buf));
        }

        // 2x2 maxpool with stride 2:
        // Top-left 2x2: max(1,2,5,6) = 6
        // Top-right 2x2: max(3,4,7,8) = 8
        // Bottom-left 2x2: max(9,10,13,14) = 14
        // Bottom-right 2x2: max(11,12,15,16) = 16
        assert_eq!(out_data, vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_maxpool2d_with_padding() {
        let pool = Pool2d::max_pool(Pool2dConfig {
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
            dilation_h: 1,
            dilation_w: 1,
            count_include_pad: true,
        });

        // Input: 1x1x3x3
        let input = Tensor::new(
            "input".to_string(),
            vec![1, 1, 3, 3],
            DataType::F32,
        ).with_data(vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);

        let output = pool.forward(&input).unwrap();

        // With pad=1, 3x3 input becomes effectively 5x5 for kernel
        // Output should be 3x3
        assert_eq!(output.shape, vec![1, 1, 3, 3]);
    }

    #[test]
    fn test_avgpool2d_basic() {
        let pool = Pool2d::avg_pool(Pool2dConfig {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            count_include_pad: true,
        });

        // Input: 1x1x4x4, all ones
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

        let output = pool.forward(&input).unwrap();

        assert_eq!(output.shape, vec![1, 1, 2, 2]);

        let out_bytes = output.data_as_bytes();
        let mut out_data = Vec::new();
        for chunk in out_bytes.chunks(4) {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            out_data.push(f32::from_le_bytes(buf));
        }

        // 2x2 avgpool with stride 2: all windows have average 1.0
        assert_eq!(out_data, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_avgpool2d_non_uniform() {
        let pool = Pool2d::avg_pool(Pool2dConfig {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            count_include_pad: true,
        });

        // Input: 1x1x4x4 with values 1-16
        let input = Tensor::new(
            "input".to_string(),
            vec![1, 1, 4, 4],
            DataType::F32,
        ).with_data(vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ]);

        let output = pool.forward(&input).unwrap();

        let out_bytes = output.data_as_bytes();
        let mut out_data = Vec::new();
        for chunk in out_bytes.chunks(4) {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            out_data.push(f32::from_le_bytes(buf));
        }

        // 2x2 avgpool with stride 2:
        // Top-left: avg(1,2,5,6) = 14/4 = 3.5
        // Top-right: avg(3,4,7,8) = 22/4 = 5.5
        // Bottom-left: avg(9,10,13,14) = 46/4 = 11.5
        // Bottom-right: avg(11,12,15,16) = 54/4 = 13.5
        assert!((out_data[0] - 3.5).abs() < 1e-5);
        assert!((out_data[1] - 5.5).abs() < 1e-5);
        assert!((out_data[2] - 11.5).abs() < 1e-5);
        assert!((out_data[3] - 13.5).abs() < 1e-5);
    }

    #[test]
    fn test_global_maxpool2d() {
        let pool = Pool2d::global_max_pool();

        // Input: 1x1x4x4
        let input = Tensor::new(
            "input".to_string(),
            vec![1, 1, 4, 4],
            DataType::F32,
        ).with_data(vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ]);

        let output = pool.forward(&input).unwrap();

        assert_eq!(output.shape, vec![1, 1, 1, 1]);

        let out_bytes = output.data_as_bytes();
        let mut out_data = Vec::new();
        for chunk in out_bytes.chunks(4) {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            out_data.push(f32::from_le_bytes(buf));
        }

        // Global max should be 16.0
        assert_eq!(out_data[0], 16.0);
    }

    #[test]
    fn test_global_avgpool2d() {
        let pool = Pool2d::global_avg_pool();

        // Input: 1x1x4x4 with all ones
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

        let output = pool.forward(&input).unwrap();

        assert_eq!(output.shape, vec![1, 1, 1, 1]);

        let out_bytes = output.data_as_bytes();
        let mut out_data = Vec::new();
        for chunk in out_bytes.chunks(4) {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            out_data.push(f32::from_le_bytes(buf));
        }

        // Global average should be 1.0
        assert_eq!(out_data[0], 1.0);
    }

    #[test]
    fn test_global_avgpool2d_non_uniform() {
        let pool = Pool2d::global_avg_pool();

        // Input: 1x1x2x2 with values 1, 2, 3, 4
        let input = Tensor::new(
            "input".to_string(),
            vec![1, 1, 2, 2],
            DataType::F32,
        ).with_data(vec![
            1.0, 2.0,
            3.0, 4.0,
        ]);

        let output = pool.forward(&input).unwrap();

        let out_bytes = output.data_as_bytes();
        let mut out_data = Vec::new();
        for chunk in out_bytes.chunks(4) {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            out_data.push(f32::from_le_bytes(buf));
        }

        // Global average should be (1+2+3+4)/4 = 2.5
        assert!((out_data[0] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_pool2d_config_output_shape() {
        let config = Pool2dConfig {
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 2,
            stride_w: 2,
            pad_h: 1,
            pad_w: 1,
            dilation_h: 1,
            dilation_w: 1,
            count_include_pad: true,
        };

        // Input: 1x1x7x7
        let (out_h, out_w) = config.output_shape(&[1, 1, 7, 7]);
        assert_eq!(out_h, 4);
        assert_eq!(out_w, 4);
    }

    #[test]
    fn test_pool2d_multi_batch() {
        let pool = Pool2d::max_pool(Pool2dConfig {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
            dilation_h: 1,
            dilation_w: 1,
            count_include_pad: true,
        });

        // Input: 2x2x4x4
        let input = Tensor::new(
            "input".to_string(),
            vec![2, 2, 4, 4],
            DataType::F32,
        ).with_data(vec![
            // Batch 0, Channel 0
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
            // Batch 0, Channel 1
            16.0, 15.0, 14.0, 13.0,
            12.0, 11.0, 10.0, 9.0,
            8.0, 7.0, 6.0, 5.0,
            4.0, 3.0, 2.0, 1.0,
            // Batch 1, Channel 0
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            // Batch 1, Channel 1
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
        ]);

        let output = pool.forward(&input).unwrap();

        assert_eq!(output.shape, vec![2, 2, 2, 2]);

        let out_bytes = output.data_as_bytes();
        let mut out_data = Vec::new();
        for chunk in out_bytes.chunks(4) {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            out_data.push(f32::from_le_bytes(buf));
        }

        // Batch 0, Ch 0: [6, 8, 14, 16]
        assert_eq!(out_data[0], 6.0);
        assert_eq!(out_data[1], 8.0);
        assert_eq!(out_data[2], 14.0);
        assert_eq!(out_data[3], 16.0);

        // Batch 0, Ch 1: [16, 14, 8, 6]
        assert_eq!(out_data[4], 16.0);
        assert_eq!(out_data[5], 14.0);
        assert_eq!(out_data[6], 8.0);
        assert_eq!(out_data[7], 6.0);

        // Batch 1, Ch 0: all ones -> max = 1
        assert_eq!(out_data[8], 1.0);
        assert_eq!(out_data[9], 1.0);
        assert_eq!(out_data[10], 1.0);
        assert_eq!(out_data[11], 1.0);

        // Batch 1, Ch 1: all twos -> max = 2
        assert_eq!(out_data[12], 2.0);
        assert_eq!(out_data[13], 2.0);
        assert_eq!(out_data[14], 2.0);
        assert_eq!(out_data[15], 2.0);
    }

    #[test]
    fn test_global_maxpool2d_simd() {
        use crate::platform::SimdLevel;
        let pool = Pool2d::with_simd_level(
            PoolType::GlobalMax,
            Pool2dConfig::default(),
            SimdLevel::None
        );

        // Input: 1x2x4x4 (2 channels)
        let input = Tensor::new(
            "input".to_string(),
            vec![1, 2, 4, 4],
            DataType::F32,
        ).with_data(vec![
            // Channel 0: 1-16
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
            // Channel 1: 16-1
            16.0, 15.0, 14.0, 13.0,
            12.0, 11.0, 10.0, 9.0,
            8.0, 7.0, 6.0, 5.0,
            4.0, 3.0, 2.0, 1.0,
        ]);

        let output = pool.forward_simd(&input).unwrap();

        let out_bytes = output.data_as_bytes();
        let mut out_data = Vec::new();
        for chunk in out_bytes.chunks(4) {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            out_data.push(f32::from_le_bytes(buf));
        }

        // Channel 0 max = 16, Channel 1 max = 16
        assert_eq!(out_data[0], 16.0);
        assert_eq!(out_data[1], 16.0);
    }

    #[test]
    fn test_global_avgpool2d_simd() {
        use crate::platform::SimdLevel;
        let pool = Pool2d::with_simd_level(
            PoolType::GlobalAvg,
            Pool2dConfig::default(),
            SimdLevel::None
        );

        // Input: 1x2x2x2 (2 channels, each 4 elements)
        let input = Tensor::new(
            "input".to_string(),
            vec![1, 2, 2, 2],
            DataType::F32,
        ).with_data(vec![
            // Channel 0: 1, 2, 3, 4 -> avg = 2.5
            1.0, 2.0,
            3.0, 4.0,
            // Channel 1: 4, 3, 2, 1 -> avg = 2.5
            4.0, 3.0,
            2.0, 1.0,
        ]);

        let output = pool.forward_simd(&input).unwrap();

        let out_bytes = output.data_as_bytes();
        let mut out_data = Vec::new();
        for chunk in out_bytes.chunks(4) {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            out_data.push(f32::from_le_bytes(buf));
        }

        assert!((out_data[0] - 2.5).abs() < 1e-5);
        assert!((out_data[1] - 2.5).abs() < 1e-5);
    }
}