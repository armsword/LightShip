//! Intermediate Representation (IR) for LightShip
//!
//! This module contains the core data structures for representing
//! neural network models as computational graphs.

pub mod tensor;
pub mod graph;
pub mod operator;
pub mod attribute;
pub mod fusion;

pub use tensor::{Tensor, TensorData, TensorShape};
pub use graph::{Graph, Node, NodeId, NodeIO};
pub use operator::{OperatorDef, OperatorType};
pub use attribute::{Attribute, AttributeMap, AttributeValue};
pub use fusion::{FusionInfo, FusionType};
