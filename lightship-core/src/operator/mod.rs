//! Operator implementations for LightShip
//!
//! This module contains implementations of various neural network operators
//! including activation functions, normalization, and attention mechanisms.

mod matmul;
mod softmax;
mod layernorm;
mod gelu;
mod attention;
mod convolution;

pub use matmul::MatMul;
pub use softmax::Softmax;
pub use layernorm::LayerNorm;
pub use gelu::Gelu;
pub use attention::{SelfAttention, MultiHeadAttention, SkipConnection};
pub use convolution::{Conv2d, Conv2dConfig};
