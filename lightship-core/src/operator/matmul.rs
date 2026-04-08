//! Matrix multiplication operator
//!
//! Implements general matrix multiplication (GEMM) for neural networks.

use crate::ir::Tensor;
use std::fmt;

/// MatMul operator result
#[derive(Debug)]
pub struct MatMulOutput {
    /// Output tensor
    pub output: Tensor,
    /// Operation time in cycles
    pub cycles: u64,
}

/// Matrix multiplication operator
///
/// Computes Y = X * W + B where:
/// - X: input tensor [M, K]
/// - W: weight tensor [K, N]
/// - B: bias tensor [N] (optional)
/// - Y: output tensor [M, N]
#[derive(Debug)]
pub struct MatMul {
    /// Transpose first input
    pub transpose_a: bool,
    /// Transpose second input
    pub transpose_b: bool,
    /// Alpha multiplier
    pub alpha: f32,
    /// Beta multiplier for bias
    pub beta: f32,
}

impl MatMul {
    /// Create a new MatMul operator
    pub fn new() -> Self {
        Self {
            transpose_a: false,
            transpose_b: false,
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Enable transpose on first input
    pub fn with_transpose_a(mut self, transpose: bool) -> Self {
        self.transpose_a = transpose;
        self
    }

    /// Enable transpose on second input
    pub fn with_transpose_b(mut self, transpose: bool) -> Self {
        self.transpose_b = transpose;
        self
    }

    /// Set alpha multiplier
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set beta multiplier
    pub fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    /// Get output shape for this matmul operation
    pub fn output_shape(&self, input_shape: &[usize], weight_shape: &[usize]) -> Vec<usize> {
        let (m, k) = if self.transpose_a {
            (input_shape[1], input_shape[0])
        } else {
            (input_shape[0], input_shape[1])
        };

        let (k2, n) = if self.transpose_b {
            (weight_shape[1], weight_shape[0])
        } else {
            (weight_shape[0], weight_shape[1])
        };

        assert_eq!(k, k2, "Matrix dimension mismatch: {} vs {}", k, k2);

        vec![m, n]
    }
}

impl Default for MatMul {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MatMul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MatMul(transpose_a={}, transpose_b={}, alpha={}, beta={})",
            self.transpose_a, self.transpose_b, self.alpha, self.beta
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_creation() {
        let op = MatMul::new();
        assert!(!op.transpose_a);
        assert!(!op.transpose_b);
        assert_eq!(op.alpha, 1.0);
        assert_eq!(op.beta, 1.0);
    }

    #[test]
    fn test_matmul_output_shape() {
        let op = MatMul::new();
        let shape = op.output_shape(&[2, 3], &[3, 4]);
        assert_eq!(shape, vec![2, 4]);
    }

    #[test]
    fn test_matmul_transpose_a() {
        let op = MatMul::new().with_transpose_a(true);
        let shape = op.output_shape(&[3, 2], &[3, 4]); // [2,3] x [3,4]
        assert_eq!(shape, vec![2, 4]);
    }

    #[test]
    fn test_matmul_transpose_b() {
        let op = MatMul::new().with_transpose_b(true);
        let shape = op.output_shape(&[2, 3], &[4, 3]); // [2,3] x [4,3]^T = [2,3] x [3,4]
        assert_eq!(shape, vec![2, 4]);
    }

    #[test]
    fn test_matmul_display() {
        let op = MatMul::new().with_transpose_a(true).with_alpha(0.5);
        let s = format!("{}", op);
        assert!(s.contains("transpose_a=true"));
        assert!(s.contains("alpha=0.5"));
    }
}
