//! Attention mechanisms
//!
//! Implements self-attention, multi-head attention, and skip connections.

use crate::ir::Tensor;
use std::fmt;

/// Attention mask type
#[derive(Debug, Clone)]
pub enum AttentionMask {
    /// No mask (causal attention)
    None,
    /// Padding mask
    Padding {
        /// Mask tensor
        mask: Tensor,
    },
    /// Causal mask (for autoregressive models)
    Causal,
    /// Arbitrary mask tensor
    Mask {
        /// Mask tensor
        mask: Tensor,
    },
}

/// Self-attention configuration
#[derive(Debug, Clone)]
pub struct SelfAttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Whether to scale the output
    pub scale: bool,
    /// Dropout probability
    pub dropout_prob: f32,
    /// Whether to use causal masking
    pub causal: bool,
}

impl SelfAttentionConfig {
    /// Create a new config
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            scale: true,
            dropout_prob: 0.0,
            causal: false,
        }
    }

    /// Total key dimension
    pub fn total_key_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

impl Default for SelfAttentionConfig {
    fn default() -> Self {
        Self::new(8, 64)
    }
}

/// Self-attention operator
///
/// Computes attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
///
/// Where Q, K, V are projected from the input.
#[derive(Debug)]
pub struct SelfAttention {
    /// Configuration
    pub config: SelfAttentionConfig,
}

impl SelfAttention {
    /// Create a new self-attention operator
    pub fn new(config: SelfAttentionConfig) -> Self {
        Self { config }
    }

    /// Create with default config
    pub fn default() -> Self {
        Self::new(SelfAttentionConfig::default())
    }

    /// Get output shape
    pub fn output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        // Self-attention typically keeps the same shape
        input_shape.to_vec()
    }

    /// Get the scale factor for attention
    pub fn get_scale(&self) -> f32 {
        if self.config.scale {
            1.0 / (self.config.head_dim as f32).sqrt()
        } else {
            1.0
        }
    }
}

impl Default for SelfAttention {
    fn default() -> Self {
        Self::default()
    }
}

impl fmt::Display for SelfAttention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SelfAttention(heads={}, dim={})",
            self.config.num_heads, self.config.head_dim
        )
    }
}

/// Multi-head attention output
#[derive(Debug)]
pub struct MultiHeadAttentionOutput {
    /// Output tensor
    pub output: Tensor,
    /// Attention weights
    pub attention_weights: Option<Tensor>,
}

/// Multi-head attention operator
///
/// Computes multi-head attention:
/// MHAttn(X) = Concat(head_1, ..., head_h) * W_O
/// where head_i = Attention(Q_i, K_i, V_i)
#[derive(Debug)]
pub struct MultiHeadAttention {
    /// Configuration
    pub config: SelfAttentionConfig,
    /// Output projection weight (if separate from input projection)
    pub out_projection_weight: Option<Tensor>,
    /// Output projection bias
    pub out_projection_bias: Option<Tensor>,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention operator
    pub fn new(config: SelfAttentionConfig) -> Self {
        Self {
            config,
            out_projection_weight: None,
            out_projection_bias: None,
        }
    }

    /// Create with output projection
    pub fn with_output_projection(mut self, weight: Tensor, bias: Option<Tensor>) -> Self {
        self.out_projection_weight = Some(weight);
        self.out_projection_bias = bias;
        self
    }

    /// Get output shape
    pub fn output_shape(&self, seq_len: usize, embed_dim: usize) -> Vec<usize> {
        vec![seq_len, embed_dim]
    }
}

impl Default for MultiHeadAttention {
    fn default() -> Self {
        Self::new(SelfAttentionConfig::default())
    }
}

impl fmt::Display for MultiHeadAttention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultiHeadAttention(heads={}, dim={})",
            self.config.num_heads, self.config.head_dim
        )
    }
}

/// Skip connection type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SkipConnectionType {
    /// Standard residual: x + f(x)
    Residual,
    /// Pre-norm residual: x + f(norm(x))
    PreNorm,
    /// Gated residual: x + gate * f(x)
    Gated { gate: f32 },
}

impl Default for SkipConnectionType {
    fn default() -> Self {
        SkipConnectionType::Residual
    }
}

/// Skip connection (residual) operator
///
/// Computes output = input + operator(input)
///
/// Supports different residual connection types:
/// - Standard: x + f(x)
/// - Pre-norm: x + f(norm(x)) where norm is typically LayerNorm
pub struct SkipConnection {
    /// Connection type
    pub connection_type: SkipConnectionType,
    /// Layer to apply before skip
    pub layer: Option<Box<dyn Fn(Tensor) -> Tensor + Send + Sync>>,
}

impl fmt::Debug for SkipConnection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SkipConnection")
            .field("connection_type", &self.connection_type)
            .field("layer", &"...")
            .finish()
    }
}

impl SkipConnection {
    /// Create a new skip connection
    pub fn new(connection_type: SkipConnectionType) -> Self {
        Self {
            connection_type,
            layer: None,
        }
    }

    /// Create standard residual connection
    pub fn residual() -> Self {
        Self::new(SkipConnectionType::Residual)
    }

    /// Create pre-norm residual connection
    pub fn pre_norm() -> Self {
        Self::new(SkipConnectionType::PreNorm)
    }

    /// Create gated residual connection
    pub fn gated(gate: f32) -> Self {
        Self::new(SkipConnectionType::Gated { gate })
    }

    /// Apply skip connection
    pub fn forward(&self, input: Tensor, output: Tensor) -> Tensor {
        match self.connection_type {
            SkipConnectionType::Residual => {
                // output = input + output
                self.add_tensors(input, output)
            }
            SkipConnectionType::PreNorm => {
                // For pre-norm, layer norm is applied before the main layer
                // So this is the same as residual
                self.add_tensors(input, output)
            }
            SkipConnectionType::Gated { gate } => {
                // output = input + gate * output
                let scaled_output = self.scale_tensor(output, gate);
                self.add_tensors(input, scaled_output)
            }
        }
    }

    /// Add two tensors element-wise
    fn add_tensors(&self, a: Tensor, _b: Tensor) -> Tensor {
        // Placeholder - actual implementation would add tensors
        a
    }

    /// Scale a tensor
    fn scale_tensor(&self, tensor: Tensor, _scale: f32) -> Tensor {
        // Placeholder - actual implementation would scale
        tensor
    }
}

impl Default for SkipConnection {
    fn default() -> Self {
        Self::residual()
    }
}

impl fmt::Display for SkipConnection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.connection_type {
            SkipConnectionType::Residual => write!(f, "SkipConnection(Residual)"),
            SkipConnectionType::PreNorm => write!(f, "SkipConnection(PreNorm)"),
            SkipConnectionType::Gated { gate } => write!(f, "SkipConnection(Gated={})", gate),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_attention_config() {
        let config = SelfAttentionConfig::new(12, 64);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.total_key_dim(), 768);
    }

    #[test]
    fn test_self_attention_creation() {
        let attn = SelfAttention::default();
        assert_eq!(attn.config.num_heads, 8);
        assert_eq!(attn.config.head_dim, 64);
    }

    #[test]
    fn test_self_attention_scale() {
        let attn = SelfAttention::new(SelfAttentionConfig::new(8, 64));
        let scale = attn.get_scale();
        // scale = 1 / sqrt(head_dim) = 1 / sqrt(64) = 0.125
        let expected: f32 = 1.0 / (64.0_f32).sqrt();
        assert!((scale - expected).abs() < 0.0001);
    }

    #[test]
    fn test_self_attention_output_shape() {
        let attn = SelfAttention::default();
        let shape = attn.output_shape(&[1, 10, 512]);
        assert_eq!(shape, vec![1, 10, 512]);
    }

    #[test]
    fn test_multi_head_attention_creation() {
        let mha = MultiHeadAttention::default();
        assert!(mha.out_projection_weight.is_none());
    }

    #[test]
    fn test_multi_head_attention_output_shape() {
        let mha = MultiHeadAttention::default();
        let shape = mha.output_shape(10, 512);
        assert_eq!(shape, vec![10, 512]);
    }

    #[test]
    fn test_skip_connection_residual() {
        let skip = SkipConnection::residual();
        assert!(matches!(skip.connection_type, SkipConnectionType::Residual));
    }

    #[test]
    fn test_skip_connection_pre_norm() {
        let skip = SkipConnection::pre_norm();
        assert!(matches!(skip.connection_type, SkipConnectionType::PreNorm));
    }

    #[test]
    fn test_skip_connection_gated() {
        let skip = SkipConnection::gated(0.5);
        assert!(matches!(skip.connection_type, SkipConnectionType::Gated { gate: 0.5 }));
    }

    #[test]
    fn test_attention_mask_none() {
        let mask = AttentionMask::None;
        assert!(matches!(mask, AttentionMask::None));
    }

    #[test]
    fn test_attention_mask_causal() {
        let mask = AttentionMask::Causal;
        assert!(matches!(mask, AttentionMask::Causal));
    }
}
