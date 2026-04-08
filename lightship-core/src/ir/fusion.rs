//! Fusion information for operator fusion optimization

use super::operator::OperatorType;
use std::fmt;

/// Fusion type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionType {
    /// Conv + ReLU fusion
    ConvReLU,
    /// Conv + ReLU6 fusion
    ConvReLU6,
    /// Conv + Sigmoid fusion
    ConvSigmoid,
    /// Conv + BatchNorm fusion
    ConvBatchNorm,
    /// BatchNorm + ReLU fusion
    BatchNormReLU,
    /// Add + ReLU fusion
    AddReLU,
    /// Mul + ReLU fusion
    MulReLU,
}

impl fmt::Display for FusionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FusionType::ConvReLU => "Conv+ReLU",
            FusionType::ConvReLU6 => "Conv+ReLU6",
            FusionType::ConvSigmoid => "Conv+Sigmoid",
            FusionType::ConvBatchNorm => "Conv+BatchNorm",
            FusionType::BatchNormReLU => "BatchNorm+ReLU",
            FusionType::AddReLU => "Add+ReLU",
            FusionType::MulReLU => "Mul+ReLU",
        };
        write!(f, "{}", s)
    }
}

/// Fusion information for fused operators
#[derive(Debug, Clone)]
pub struct FusionInfo {
    /// Fusion type
    pub fusion_type: FusionType,
    /// Original operators in this fusion
    pub original_ops: Vec<OperatorType>,
    /// Whether batch norm can be eliminated after fusion
    pub eliminate_batch_norm: bool,
}

impl FusionInfo {
    /// Create a new fusion info
    pub fn new(fusion_type: FusionType, original_ops: Vec<OperatorType>) -> Self {
        let eliminate_batch_norm = original_ops.contains(&OperatorType::BatchNorm);
        Self {
            fusion_type,
            original_ops,
            eliminate_batch_norm,
        }
    }

    /// Create Conv + ReLU fusion
    pub fn conv_relu() -> Self {
        Self::new(
            FusionType::ConvReLU,
            vec![OperatorType::Conv2d, OperatorType::ReLU],
        )
    }

    /// Create Conv + BatchNorm fusion
    pub fn conv_batch_norm() -> Self {
        Self::new(
            FusionType::ConvBatchNorm,
            vec![OperatorType::Conv2d, OperatorType::BatchNorm],
        )
    }
}
