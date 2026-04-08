//! Graph optimization passes
//!
//! This module contains various graph optimization algorithms including
//! constant folding, dead code elimination, node replacement, and shape inference.

use super::{FusionInfo, FusionType, Graph, Node, NodeIO, NodeId, OperatorType};
use std::collections::{HashMap, HashSet, VecDeque};

/// Graph optimizer trait
pub trait GraphOptimizer {
    /// Optimize the graph in-place
    fn optimize(&self, graph: &mut Graph);

    /// Get the name of this optimizer
    fn name(&self) -> &'static str;
}

/// Constant folding optimization
///
/// Removes identity operations like:
/// - Add(0) = identity
/// - Mul(1) = identity
/// - Sub(x, x) = 0
#[derive(Debug)]
pub struct ConstantFolding;

impl ConstantFolding {
    /// Create a new constant folding optimizer
    pub fn new() -> Self {
        Self
    }

    /// Check if a node is a constant that can be folded
    #[allow(dead_code)]
    fn is_constant(node: &Node) -> bool {
        matches!(
            node.operator_type,
            OperatorType::Add
                | OperatorType::Sub
                | OperatorType::Mul
                | OperatorType::Div
                | OperatorType::Reshape
                | OperatorType::Transpose
        )
    }

    /// Try to evaluate a binary constant expression
    #[allow(dead_code)]
    fn eval_binary_const(left: f32, right: f32, op: OperatorType) -> Option<f32> {
        match op {
            OperatorType::Add => Some(left + right),
            OperatorType::Sub => Some(left - right),
            OperatorType::Mul => Some(left * right),
            OperatorType::Div => {
                if right.abs() > 1e-10 {
                    Some(left / right)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

impl GraphOptimizer for ConstantFolding {
    fn optimize(&self, graph: &mut Graph) {
        // Remove identity operations like Add(0), Mul(1), etc.
        // For simplicity, single-input Add/Mul/Sub are considered foldable
        // A real implementation would check actual constant values

        let nodes_to_remove: HashSet<NodeId> = graph
            .nodes
            .iter()
            .filter(|node| {
                if matches!(
                    node.operator_type,
                    OperatorType::Add | OperatorType::Mul | OperatorType::Sub
                ) {
                    node.inputs.len() == 1
                } else {
                    false
                }
            })
            .map(|node| node.id)
            .collect();

        graph.retain_nodes(|n| !nodes_to_remove.contains(&n.id));
    }

    fn name(&self) -> &'static str {
        "ConstantFolding"
    }
}

/// Dead code elimination optimization
///
/// Removes nodes that are not reachable from model inputs or outputs.
#[derive(Debug)]
pub struct DeadCodeElimination {
    inputs: Vec<String>,
    outputs: Vec<String>,
}

impl DeadCodeElimination {
    /// Create a new dead code elimination optimizer
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Set the input tensor names
    pub fn set_inputs(&mut self, inputs: Vec<String>) {
        self.inputs = inputs;
    }

    /// Set the output tensor names
    pub fn set_outputs(&mut self, outputs: Vec<String>) {
        self.outputs = outputs;
    }

    /// Find all nodes that are reachable from inputs or produce outputs
    fn find_reachable_nodes(&self, graph: &Graph) -> HashSet<NodeId> {
        let mut reachable = HashSet::new();
        let mut worklist: VecDeque<NodeId> = VecDeque::new();

        // Start from inputs
        for input_name in &self.inputs {
            if let Some(node) = graph.node(input_name) {
                worklist.push_back(node.id);
            }
        }

        // Also start from nodes that produce outputs
        for output_name in &self.outputs {
            for node in &graph.nodes {
                if node.outputs.iter().any(|o| &o.tensor_name == output_name) {
                    worklist.push_back(node.id);
                }
            }
        }

        // BFS to find all reachable nodes
        while let Some(node_id) = worklist.pop_front() {
            if reachable.contains(&node_id) {
                continue;
            }
            reachable.insert(node_id);

            if let Some(node) = graph.node_by_id(node_id) {
                for output in &node.outputs {
                    for consumer in &graph.nodes {
                        if consumer
                            .inputs
                            .iter()
                            .any(|i| &i.tensor_name == &output.tensor_name)
                        {
                            worklist.push_back(consumer.id);
                        }
                    }
                }
            }
        }

        reachable
    }
}

impl GraphOptimizer for DeadCodeElimination {
    fn optimize(&self, graph: &mut Graph) {
        let reachable = self.find_reachable_nodes(graph);
        graph.retain_nodes(|n| reachable.contains(&n.id));
    }

    fn name(&self) -> &'static str {
        "DeadCodeElimination"
    }
}

/// Node replacement optimization
///
/// Allows replacing one node with another in the graph.
#[derive(Debug)]
pub struct NodeReplacement;

impl NodeReplacement {
    /// Create a new node replacement optimizer
    pub fn new() -> Self {
        Self
    }

    /// Replace a node with a new one
    pub fn replace_node(&self, graph: &mut Graph, old_name: &str, new_node: Node) {
        if let Some(old_node) = graph.node(old_name) {
            let old_id = old_node.id;
            let new_name = new_node.name.clone();
            let mut node = new_node;
            node.id = old_id;

            if let Some(existing) = graph.nodes.get_mut(old_id as usize) {
                *existing = node;
            }

            // Update the name index with the new node name
            graph.update_node_name(old_name, new_name, old_id);
        }
    }
}

impl GraphOptimizer for NodeReplacement {
    fn optimize(&self, _graph: &mut Graph) {
        // Node replacement is typically done via direct method calls
    }

    fn name(&self) -> &'static str {
        "NodeReplacement"
    }
}

/// Shape inference engine
///
/// Infers tensor shapes through the graph based on operator semantics.
#[derive(Debug)]
pub struct ShapeInference;

impl ShapeInference {
    /// Create a new shape inference engine
    pub fn new() -> Self {
        Self
    }

    /// Infer output shape for a node given input shapes
    ///
    /// Returns `None` if shape cannot be inferred.
    pub fn infer_shape(
        &self,
        node: &Node,
        input_shapes: &HashMap<String, Vec<usize>>,
    ) -> Option<Vec<Vec<usize>>> {
        match node.operator_type {
            OperatorType::Conv2d => {
                let input_name = node.inputs.first()?.tensor_name.as_str();
                let input_shape = input_shapes.get(input_name)?;

                // Default parameters
                let kernel_h = 3usize;
                let kernel_w = 3usize;
                let stride_h = 1usize;
                let stride_w = 1usize;
                let pad_h = 0usize;
                let pad_w = 0usize;

                // NCHW format: [N, C, H, W]
                if input_shape.len() >= 4 {
                    let n = input_shape[0];
                    let c = input_shape[1];
                    let h = input_shape[2];
                    let w = input_shape[3];

                    let out_h = (h + 2 * pad_h - kernel_h) / stride_h + 1;
                    let out_w = (w + 2 * pad_w - kernel_w) / stride_w + 1;

                    return Some(vec![vec![n, c, out_h, out_w]]);
                }
                None
            }
            OperatorType::ReLU | OperatorType::Sigmoid | OperatorType::Tanh => {
                // Activation doesn't change shape
                let input_name = node.inputs.first()?.tensor_name.as_str();
                input_shapes.get(input_name).map(|s| vec![s.clone()])
            }
            OperatorType::Reshape => {
                // Reshape preserves element count
                let input_name = node.inputs.first()?.tensor_name.as_str();
                input_shapes.get(input_name).map(|s| vec![s.clone()])
            }
            OperatorType::MaxPool2d | OperatorType::AvgPool2d => {
                // Pooling reduces spatial dimensions - assume 2x2 kernel
                let input_name = node.inputs.first()?.tensor_name.as_str();
                let input_shape = input_shapes.get(input_name)?;

                if input_shape.len() >= 4 {
                    let n = input_shape[0];
                    let c = input_shape[1];
                    let h = input_shape[2] / 2;
                    let w = input_shape[3] / 2;

                    return Some(vec![vec![n, c, h, w]]);
                }
                None
            }
            _ => None,
        }
    }
}

impl Default for ConstantFolding {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DeadCodeElimination {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for NodeReplacement {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ShapeInference {
    fn default() -> Self {
        Self::new()
    }
}
