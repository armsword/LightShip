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

/// Operator fusion optimization
///
/// Fuses sequential operators into a single fused operator for better performance.
/// Supported fusions:
/// - Conv + ReLU
/// - Conv + ReLU6
/// - Add + ReLU (residual connections)
#[derive(Debug)]
pub struct FusionPass {
    /// Types of fusion to apply
    fusion_types: Vec<FusionType>,
}

impl FusionPass {
    /// Create a new fusion pass
    pub fn new() -> Self {
        Self {
            fusion_types: vec![
                FusionType::ConvReLU,
                FusionType::ConvReLU6,
                FusionType::AddReLU,
            ],
        }
    }

    /// Create a fusion pass with specific fusion types
    pub fn with_types(fusion_types: Vec<FusionType>) -> Self {
        Self { fusion_types }
    }

    /// Find candidate fusion patterns in the graph
    fn find_fusion_candidates(&self, graph: &Graph) -> Vec<(NodeId, NodeId, FusionType)> {
        let mut candidates = Vec::new();

        for i in 0..graph.nodes.len() {
            let node = &graph.nodes[i];

            // Check if this node can be a fusion producer
            match node.operator_type {
                OperatorType::Conv2d | OperatorType::Add => {
                    // Look for ReLU/ReLU6 consumers
                    if let Some(consumer) = self.find_relu_consumer(graph, node) {
                        // Only fuse if the ReLU node itself has exactly one consumer
                        // If ReLU has multiple consumers, we can't fuse without breaking the graph
                        if !self.has_single_consumer(graph, &consumer) {
                            continue;
                        }

                        if node.operator_type == OperatorType::Conv2d {
                            candidates.push((node.id, consumer.id, FusionType::ConvReLU));
                        } else if node.operator_type == OperatorType::Add {
                            candidates.push((node.id, consumer.id, FusionType::AddReLU));
                        }
                    }
                }
                _ => {}
            }
        }

        candidates
    }

    /// Find a ReLU/ReLU6 consumer of this node's output
    fn find_relu_consumer(&self, graph: &Graph, producer: &Node) -> Option<Node> {
        let output_name = producer.outputs.first()?.tensor_name.clone();

        // Find nodes that consume this output
        for node in &graph.nodes {
            if node.inputs.iter().any(|i| i.tensor_name == output_name) {
                // Check if it's a ReLU or ReLU6
                if matches!(node.operator_type, OperatorType::ReLU | OperatorType::ReLU6) {
                    return Some(node.clone());
                }
            }
        }
        None
    }

    /// Check if a node has only one consumer
    fn has_single_consumer(&self, graph: &Graph, node: &Node) -> bool {
        let output_name = node.outputs.first().map(|o| o.tensor_name.clone());
        if let Some(name) = output_name {
            graph.nodes.iter().filter(|n| n.inputs.iter().any(|i| i.tensor_name == name)).count() == 1
        } else {
            false
        }
    }

    /// Fuse Conv + ReLU into a single Conv node with fusion info
    fn fuse_conv_relu(&self, graph: &mut Graph, conv_id: NodeId, relu_id: NodeId) {
        // First, collect all the information we need
        let (relu_output, conv_output) = {
            let relu_out = graph.nodes.iter()
                .find(|n| n.id == relu_id)
                .and_then(|n| n.outputs.first())
                .map(|o| o.tensor_name.clone());

            let conv_out = graph.nodes.iter()
                .find(|n| n.id == conv_id)
                .and_then(|n| n.outputs.first())
                .map(|o| o.tensor_name.clone());

            (relu_out, conv_out)
        };

        // Now update the Conv node's fusion info
        if let Some(conv_node) = graph.nodes.iter_mut().find(|n| n.id == conv_id) {
            let fusion_info = FusionInfo::conv_relu();
            conv_node.fusion = Some(fusion_info);
        }

        // Update consumers of relu_output to use conv_output
        if let (Some(relu_out), Some(conv_out)) = (relu_output, conv_output) {
            for node in &mut graph.nodes {
                for input in &mut node.inputs {
                    if input.tensor_name == relu_out {
                        input.tensor_name = conv_out.clone();
                    }
                }
            }
        }

        // Remove the ReLU node
        graph.retain_nodes(|n| n.id != relu_id);
    }

    /// Fuse Add + ReLU for residual connections
    fn fuse_add_relu(&self, graph: &mut Graph, add_id: NodeId, relu_id: NodeId) {
        // First, collect all the information we need
        let (relu_output, add_output) = {
            let relu_out = graph.nodes.iter()
                .find(|n| n.id == relu_id)
                .and_then(|n| n.outputs.first())
                .map(|o| o.tensor_name.clone());

            let add_out = graph.nodes.iter()
                .find(|n| n.id == add_id)
                .and_then(|n| n.outputs.first())
                .map(|o| o.tensor_name.clone());

            (relu_out, add_out)
        };

        // Now update the Add node's fusion info
        if let Some(add_node) = graph.nodes.iter_mut().find(|n| n.id == add_id) {
            let fusion_info = FusionInfo::new(
                FusionType::AddReLU,
                vec![OperatorType::Add, OperatorType::ReLU],
            );
            add_node.fusion = Some(fusion_info);
        }

        // Update consumers of relu_output to use add_output
        if let (Some(relu_out), Some(add_out)) = (relu_output, add_output) {
            for node in &mut graph.nodes {
                for input in &mut node.inputs {
                    if input.tensor_name == relu_out {
                        input.tensor_name = add_out.clone();
                    }
                }
            }
        }

        // Remove the ReLU node
        graph.retain_nodes(|n| n.id != relu_id);
    }
}

impl GraphOptimizer for FusionPass {
    fn optimize(&self, graph: &mut Graph) {
        // Find all fusion candidates
        let candidates = self.find_fusion_candidates(graph);

        // Apply each fusion
        for (producer_id, consumer_id, fusion_type) in candidates {
            match fusion_type {
                FusionType::ConvReLU | FusionType::ConvReLU6 => {
                    self.fuse_conv_relu(graph, producer_id, consumer_id);
                }
                FusionType::AddReLU => {
                    self.fuse_add_relu(graph, producer_id, consumer_id);
                }
                _ => {}
            }
        }
    }

    fn name(&self) -> &'static str {
        "FusionPass"
    }
}

impl Default for FusionPass {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::DataType;
    use crate::ir::{Graph, Node, NodeIO, OperatorType};

    fn create_test_graph() -> Graph {
        Graph::new("test".to_string())
    }

    fn make_node_io(name: &str) -> NodeIO {
        NodeIO {
            tensor_name: name.to_string(),
            data_type: DataType::F32,
        }
    }

    fn add_generic_node(graph: &mut Graph, name: &str, op_type: OperatorType, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeId {
        let next_id = graph.nodes.len() as NodeId;
        let mut node = Node::new(next_id, name.to_string(), op_type);
        node.inputs = inputs.iter().map(|s| make_node_io(s)).collect();
        node.outputs = outputs.iter().map(|s| make_node_io(s)).collect();
        graph.add_node(node)
    }

    fn add_conv_node(graph: &mut Graph, name: &str) -> NodeId {
        add_generic_node(graph, name, OperatorType::Conv2d, vec!["input"], vec![&format!("{}_output", name)])
    }

    fn add_relu_node(graph: &mut Graph, name: &str, input: &str) -> NodeId {
        add_generic_node(graph, name, OperatorType::ReLU, vec![input], vec![&format!("{}_output", name)])
    }

    fn add_add_node(graph: &mut Graph, name: &str, input1: &str, input2: &str) -> NodeId {
        add_generic_node(graph, name, OperatorType::Add, vec![input1, input2], vec![&format!("{}_output", name)])
    }

    #[test]
    fn test_fusion_pass_creation() {
        let fusion = FusionPass::new();
        assert_eq!(fusion.name(), "FusionPass");
    }

    #[test]
    fn test_fusion_pass_with_types() {
        let fusion = FusionPass::with_types(vec![FusionType::ConvReLU]);
        assert_eq!(fusion.name(), "FusionPass");
    }

    #[test]
    fn test_find_conv_relu_fusion() {
        let fusion = FusionPass::new();
        let mut graph = create_test_graph();

        // Create: Conv -> ReLU
        let conv_id = add_conv_node(&mut graph, "conv");
        let _relu_id = add_relu_node(&mut graph, "relu", "conv_output");

        // Add an output consumer
        let _output_node = add_generic_node(&mut graph, "output", OperatorType::Reshape, vec!["relu_output"], vec!["final_output"]);

        let candidates = fusion.find_fusion_candidates(&graph);
        assert!(candidates.iter().any(|(_, _, ft)| *ft == FusionType::ConvReLU));
    }

    #[test]
    fn test_find_add_relu_fusion() {
        let fusion = FusionPass::new();
        let mut graph = create_test_graph();

        // Create: Add -> ReLU (residual connection pattern)
        let add_id = add_add_node(&mut graph, "add", "input1", "input2");
        let _relu_id = add_relu_node(&mut graph, "relu", "add_output");

        // Add an output consumer
        let _output_node = add_generic_node(&mut graph, "output", OperatorType::Reshape, vec!["relu_output"], vec!["final_output"]);

        let candidates = fusion.find_fusion_candidates(&graph);
        assert!(candidates.iter().any(|(_, _, ft)| *ft == FusionType::AddReLU));
    }

    #[test]
    fn test_conv_relu_fusion() {
        let fusion = FusionPass::new();
        let mut graph = create_test_graph();

        // Create: Conv -> ReLU
        let conv_id = add_conv_node(&mut graph, "conv");
        let relu_id = add_relu_node(&mut graph, "relu", "conv_output");

        // Add an output consumer
        let _output_node = add_generic_node(&mut graph, "output", OperatorType::Reshape, vec!["relu_output"], vec!["final_output"]);

        // Apply fusion
        fusion.optimize(&mut graph);

        // Check that ReLU is removed
        let relu_exists = graph.nodes.iter().any(|n| n.id == relu_id);
        assert!(!relu_exists, "ReLU node should be removed after fusion");

        // Check that Conv has fusion info
        let conv_node = graph.nodes.iter().find(|n| n.id == conv_id);
        assert!(conv_node.is_some());
        let conv = conv_node.unwrap();
        assert!(conv.fusion.is_some());
        assert_eq!(conv.fusion.as_ref().unwrap().fusion_type, FusionType::ConvReLU);
    }

    #[test]
    fn test_add_relu_fusion() {
        let fusion = FusionPass::new();
        let mut graph = create_test_graph();

        // Create: Add -> ReLU
        let add_id = add_add_node(&mut graph, "add", "input1", "input2");
        let relu_id = add_relu_node(&mut graph, "relu", "add_output");

        // Add an output consumer
        let _output_node = add_generic_node(&mut graph, "output", OperatorType::Reshape, vec!["relu_output"], vec!["final_output"]);

        // Apply fusion
        fusion.optimize(&mut graph);

        // Check that ReLU is removed
        let relu_exists = graph.nodes.iter().any(|n| n.id == relu_id);
        assert!(!relu_exists, "ReLU node should be removed after fusion");

        // Check that Add has fusion info
        let add_node = graph.nodes.iter().find(|n| n.id == add_id);
        assert!(add_node.is_some());
        let add = add_node.unwrap();
        assert!(add.fusion.is_some());
        assert_eq!(add.fusion.as_ref().unwrap().fusion_type, FusionType::AddReLU);
    }

    #[test]
    fn test_fusion_skips_multi_consumer() {
        let fusion = FusionPass::new();
        let mut graph = create_test_graph();

        // Create: Conv -> ReLU -> (consumer1, consumer2)
        let _conv_id = add_conv_node(&mut graph, "conv");
        let _relu_id = add_relu_node(&mut graph, "relu", "conv_output");

        // Two consumers of ReLU
        let _consumer1 = add_generic_node(&mut graph, "consumer1", OperatorType::Reshape, vec!["relu_output"], vec!["output1"]);
        let _consumer2 = add_generic_node(&mut graph, "consumer2", OperatorType::Reshape, vec!["relu_output"], vec!["output2"]);

        let candidates = fusion.find_fusion_candidates(&graph);

        // Should not find fusion candidate because ReLU has multiple consumers
        assert!(candidates.is_empty());
    }
}
