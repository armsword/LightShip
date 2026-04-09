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
            // === Convolution ===
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
            OperatorType::ConvTranspose2d => {
                // ConvTranspose increases spatial dimensions
                let input_name = node.inputs.first()?.tensor_name.as_str();
                let input_shape = input_shapes.get(input_name)?;

                if input_shape.len() >= 4 {
                    let n = input_shape[0];
                    let c = input_shape[1];
                    let h = input_shape[2] * 2;  // stride=2 by default
                    let w = input_shape[3] * 2;

                    return Some(vec![vec![n, c, h, w]]);
                }
                None
            }

            // === Activation functions (preserve shape) ===
            OperatorType::ReLU | OperatorType::ReLU6 | OperatorType::Sigmoid | OperatorType::Tanh | OperatorType::GELU | OperatorType::SiLU => {
                let input_name = node.inputs.first()?.tensor_name.as_str();
                input_shapes.get(input_name).map(|s| vec![s.clone()])
            }

            // === Normalization (preserve shape) ===
            OperatorType::BatchNorm | OperatorType::LayerNorm | OperatorType::InstanceNorm => {
                let input_name = node.inputs.first()?.tensor_name.as_str();
                input_shapes.get(input_name).map(|s| vec![s.clone()])
            }

            // === Softmax (preserves shape, typically on last dim) ===
            OperatorType::Softmax => {
                let input_name = node.inputs.first()?.tensor_name.as_str();
                input_shapes.get(input_name).map(|s| vec![s.clone()])
            }

            // === Element-wise binary operations (broadcast) ===
            OperatorType::Add | OperatorType::Sub | OperatorType::Mul | OperatorType::Div => {
                // For simplicity, assume shapes match or broadcast to larger
                let input_name = node.inputs.first()?.tensor_name.as_str();
                input_shapes.get(input_name).map(|s| vec![s.clone()])
            }

            // === Matrix multiplication ===
            OperatorType::MatMul | OperatorType::FullyConnected => {
                // [M, K] @ [K, N] = [M, N]
                // For FC: input [batch, in_features] -> output [batch, out_features]
                let input_name = node.inputs.first()?.tensor_name.as_str();
                let input_shape = input_shapes.get(input_name)?;

                if input_shape.len() >= 2 {
                    let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
                    output_shape.push(512); // Default output features
                    return Some(vec![output_shape]);
                }
                input_shapes.get(input_name).map(|s| vec![s.clone()])
            }

            // === Pooling ===
            OperatorType::MaxPool2d | OperatorType::AvgPool2d => {
                let input_name = node.inputs.first()?.tensor_name.as_str();
                let input_shape = input_shapes.get(input_name)?;

                if input_shape.len() >= 4 {
                    let n = input_shape[0];
                    let c = input_shape[1];
                    let h = input_shape[2] / 2;  // 2x2 kernel, stride 2
                    let w = input_shape[3] / 2;

                    return Some(vec![vec![n, c, h, w]]);
                }
                None
            }
            OperatorType::GlobalAvgPool2d | OperatorType::GlobalMaxPool2d => {
                // Global pooling reduces H,W to 1
                let input_name = node.inputs.first()?.tensor_name.as_str();
                let input_shape = input_shapes.get(input_name)?;

                if input_shape.len() >= 4 {
                    let n = input_shape[0];
                    let c = input_shape[1];

                    return Some(vec![vec![n, c, 1, 1]]);
                }
                None
            }

            // === Reshape operations ===
            OperatorType::Reshape => {
                // Reshape preserves element count
                let input_name = node.inputs.first()?.tensor_name.as_str();
                input_shapes.get(input_name).map(|s| vec![s.clone()])
            }
            OperatorType::Flatten => {
                // Flatten reshapes to [batch, features]
                let input_name = node.inputs.first()?.tensor_name.as_str();
                let input_shape = input_shapes.get(input_name)?;

                if input_shape.len() >= 2 {
                    let batch = input_shape[0];
                    let features: usize = input_shape[1..].iter().product();
                    return Some(vec![vec![batch, features]]);
                }
                None
            }
            OperatorType::Squeeze | OperatorType::Unsqueeze => {
                // Squeeze/Unsqueeze remove or add dimension of size 1
                let input_name = node.inputs.first()?.tensor_name.as_str();
                input_shapes.get(input_name).map(|s| vec![s.clone()])
            }

            // === Transpose ===
            OperatorType::Transpose => {
                let input_name = node.inputs.first()?.tensor_name.as_str();
                let input_shape = input_shapes.get(input_name)?;

                if input_shape.len() == 4 {
                    // [N, C, H, W] -> [N, H, W, C] (NHWC) or similar permutation
                    return Some(vec![vec![input_shape[0], input_shape[2], input_shape[3], input_shape[1]]]);
                }
                input_shapes.get(input_name).map(|s| vec![s.clone()])
            }

            // === Concat ===
            OperatorType::Concat => {
                // Concatenation along an axis preserves all other dimensions
                // For simplicity, return first input's shape
                // (actual concat would need axis from attributes)
                let input_name = node.inputs.first()?.tensor_name.as_str();
                input_shapes.get(input_name).map(|s| vec![s.clone()])
            }

            // === Split ===
            OperatorType::Split => {
                // Split produces multiple outputs with same shape as input on split axis
                let input_name = node.inputs.first()?.tensor_name.as_str();
                let input_shape = input_shapes.get(input_name)?;

                // Return same shape (actual split would produce chunks)
                Some(vec![input_shape.clone()])
            }

            // === Slice ===
            OperatorType::Slice => {
                let input_name = node.inputs.first()?.tensor_name.as_str();
                input_shapes.get(input_name).map(|s| vec![s.clone()])
            }

            // === Pad ===
            OperatorType::Pad => {
                let input_name = node.inputs.first()?.tensor_name.as_str();
                input_shapes.get(input_name).map(|s| vec![s.clone()])
            }

            // === Attention ===
            OperatorType::SelfAttention | OperatorType::MultiHeadAttention => {
                // Attention: [batch, seq_len, features] -> [batch, seq_len, features]
                let input_name = node.inputs.first()?.tensor_name.as_str();
                input_shapes.get(input_name).map(|s| vec![s.clone()])
            }

            // === Unknown/Identity ===
            _ => {
                // Default: preserve input shape
                let input_name = node.inputs.first()?.tensor_name.as_str();
                input_shapes.get(input_name).map(|s| vec![s.clone()])
            }
        }
    }

    /// Infer shapes for the entire graph
    ///
    /// Given input shapes for graph inputs, propagates shapes through all nodes.
    /// Returns a map from tensor name to inferred shape, or None if inference fails.
    pub fn infer_graph_shapes(
        &self,
        graph: &Graph,
        input_shapes: &HashMap<String, Vec<usize>>,
    ) -> Option<HashMap<String, Vec<usize>>> {
        let mut shapes = input_shapes.clone();

        // Topologically sort nodes
        let sorted = graph.topological_sort();

        // Process each node in order
        for node_id in sorted {
            let node = graph.nodes.iter().find(|n| n.id == node_id)?;

            // Collect input shapes for this node
            let node_input_shapes: HashMap<String, Vec<usize>> = node
                .inputs
                .iter()
                .filter_map(|input| {
                    shapes.get(&input.tensor_name).map(|s| (input.tensor_name.clone(), s.clone()))
                })
                .collect();

            // Infer output shapes
            if let Some(output_shapes) = self.infer_shape(node, &node_input_shapes) {
                for (i, output) in node.outputs.iter().enumerate() {
                    if i < output_shapes.len() {
                        shapes.insert(output.tensor_name.clone(), output_shapes[i].clone());
                    }
                }
            }
        }

        Some(shapes)
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
                FusionType::ConvBatchNorm,
                FusionType::BatchNormReLU,
                FusionType::MulReLU,
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

            match node.operator_type {
                OperatorType::Conv2d => {
                    // Look for ReLU/ReLU6 consumers (Conv + ReLU fusion)
                    if let Some(consumer) = self.find_relu_consumer(graph, node) {
                        if self.has_single_consumer(graph, &consumer) {
                            if self.fusion_types.contains(&FusionType::ConvReLU) {
                                candidates.push((node.id, consumer.id, FusionType::ConvReLU));
                            }
                        }
                    }

                    // Look for BatchNorm consumers (Conv + BN fusion)
                    if let Some(consumer) = self.find_batchnorm_consumer(graph, node) {
                        if self.has_single_consumer(graph, &consumer) {
                            if self.fusion_types.contains(&FusionType::ConvBatchNorm) {
                                candidates.push((node.id, consumer.id, FusionType::ConvBatchNorm));
                            }
                        }
                    }
                }
                OperatorType::Add => {
                    // Look for ReLU/ReLU6 consumers (Add + ReLU fusion)
                    if let Some(consumer) = self.find_relu_consumer(graph, node) {
                        if self.has_single_consumer(graph, &consumer) {
                            if self.fusion_types.contains(&FusionType::AddReLU) {
                                candidates.push((node.id, consumer.id, FusionType::AddReLU));
                            }
                        }
                    }
                }
                OperatorType::BatchNorm => {
                    // Look for ReLU/ReLU6 consumers (BatchNorm + ReLU fusion)
                    if let Some(consumer) = self.find_relu_consumer(graph, node) {
                        if self.has_single_consumer(graph, &consumer) {
                            if self.fusion_types.contains(&FusionType::BatchNormReLU) {
                                candidates.push((node.id, consumer.id, FusionType::BatchNormReLU));
                            }
                        }
                    }
                }
                OperatorType::Mul => {
                    // Look for ReLU/ReLU6 consumers (Mul + ReLU fusion)
                    if let Some(consumer) = self.find_relu_consumer(graph, node) {
                        if self.has_single_consumer(graph, &consumer) {
                            if self.fusion_types.contains(&FusionType::MulReLU) {
                                candidates.push((node.id, consumer.id, FusionType::MulReLU));
                            }
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

    /// Find a BatchNorm consumer of this node's output
    fn find_batchnorm_consumer(&self, graph: &Graph, producer: &Node) -> Option<Node> {
        let output_name = producer.outputs.first()?.tensor_name.clone();

        // Find nodes that consume this output
        for node in &graph.nodes {
            if node.inputs.iter().any(|i| i.tensor_name == output_name) {
                // Check if it's a BatchNorm
                if matches!(node.operator_type, OperatorType::BatchNorm) {
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

    /// Fuse Conv + BatchNorm into a single Conv node with fusion info
    ///
    /// This performs a "lightweight" fusion where:
    /// 1. The Conv node is marked with FusionInfo indicating Conv+BN fusion
    /// 2. The eliminate_batch_norm flag indicates BN can be eliminated from the graph
    /// 3. Consumers of BN are redirected to use Conv's output
    /// 4. The BN node is removed from the graph
    ///
    /// Note: For full fusion, Conv weights would need to be updated with BN parameters:
    ///   new_weight = gamma / sqrt(var + eps) * weight
    ///   new_bias = gamma / sqrt(var + eps) * (bias - mean) + beta
    fn fuse_conv_batch_norm(&self, graph: &mut Graph, conv_id: NodeId, bn_id: NodeId) {
        // First, collect all the information we need
        let (bn_output, conv_output) = {
            let bn_out = graph.nodes.iter()
                .find(|n| n.id == bn_id)
                .and_then(|n| n.outputs.first())
                .map(|o| o.tensor_name.clone());

            let conv_out = graph.nodes.iter()
                .find(|n| n.id == conv_id)
                .and_then(|n| n.outputs.first())
                .map(|o| o.tensor_name.clone());

            (bn_out, conv_out)
        };

        // Update the Conv node's fusion info
        if let Some(conv_node) = graph.nodes.iter_mut().find(|n| n.id == conv_id) {
            let fusion_info = FusionInfo::conv_batch_norm();
            conv_node.fusion = Some(fusion_info);
        }

        // Update consumers of bn_output to use conv_output
        if let (Some(bn_out), Some(conv_out)) = (bn_output, conv_output) {
            for node in &mut graph.nodes {
                for input in &mut node.inputs {
                    if input.tensor_name == bn_out {
                        input.tensor_name = conv_out.clone();
                    }
                }
            }
        }

        // Remove the BatchNorm node
        graph.retain_nodes(|n| n.id != bn_id);
    }

    /// Fuse BatchNorm + ReLU into a single BatchNorm node with fusion info
    ///
    /// This performs a "lightweight" fusion where:
    /// 1. The BatchNorm node is marked with FusionInfo indicating BN+ReLU fusion
    /// 2. Consumers of ReLU are redirected to use BatchNorm's output
    /// 3. The ReLU node is removed from the graph
    fn fuse_batchnorm_relu(&self, graph: &mut Graph, bn_id: NodeId, relu_id: NodeId) {
        // First, collect all the information we need
        let (relu_output, bn_output) = {
            let relu_out = graph.nodes.iter()
                .find(|n| n.id == relu_id)
                .and_then(|n| n.outputs.first())
                .map(|o| o.tensor_name.clone());

            let bn_out = graph.nodes.iter()
                .find(|n| n.id == bn_id)
                .and_then(|n| n.outputs.first())
                .map(|o| o.tensor_name.clone());

            (relu_out, bn_out)
        };

        // Update the BatchNorm node's fusion info
        if let Some(bn_node) = graph.nodes.iter_mut().find(|n| n.id == bn_id) {
            let fusion_info = FusionInfo::new(
                FusionType::BatchNormReLU,
                vec![OperatorType::BatchNorm, OperatorType::ReLU],
            );
            bn_node.fusion = Some(fusion_info);
        }

        // Update consumers of relu_output to use bn_output
        if let (Some(relu_out), Some(bn_out)) = (relu_output, bn_output) {
            for node in &mut graph.nodes {
                for input in &mut node.inputs {
                    if input.tensor_name == relu_out {
                        input.tensor_name = bn_out.clone();
                    }
                }
            }
        }

        // Remove the ReLU node
        graph.retain_nodes(|n| n.id != relu_id);
    }

    /// Fuse Mul + ReLU into a single Mul node with fusion info
    ///
    /// This performs a "lightweight" fusion where:
    /// 1. The Mul node is marked with FusionInfo indicating Mul+ReLU fusion
    /// 2. Consumers of ReLU are redirected to use Mul's output
    /// 3. The ReLU node is removed from the graph
    fn fuse_mul_relu(&self, graph: &mut Graph, mul_id: NodeId, relu_id: NodeId) {
        // First, collect all the information we need
        let (relu_output, mul_output) = {
            let relu_out = graph.nodes.iter()
                .find(|n| n.id == relu_id)
                .and_then(|n| n.outputs.first())
                .map(|o| o.tensor_name.clone());

            let mul_out = graph.nodes.iter()
                .find(|n| n.id == mul_id)
                .and_then(|n| n.outputs.first())
                .map(|o| o.tensor_name.clone());

            (relu_out, mul_out)
        };

        // Update the Mul node's fusion info
        if let Some(mul_node) = graph.nodes.iter_mut().find(|n| n.id == mul_id) {
            let fusion_info = FusionInfo::new(
                FusionType::MulReLU,
                vec![OperatorType::Mul, OperatorType::ReLU],
            );
            mul_node.fusion = Some(fusion_info);
        }

        // Update consumers of relu_output to use mul_output
        if let (Some(relu_out), Some(mul_out)) = (relu_output, mul_output) {
            for node in &mut graph.nodes {
                for input in &mut node.inputs {
                    if input.tensor_name == relu_out {
                        input.tensor_name = mul_out.clone();
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
                FusionType::ConvBatchNorm => {
                    self.fuse_conv_batch_norm(graph, producer_id, consumer_id);
                }
                FusionType::BatchNormReLU => {
                    self.fuse_batchnorm_relu(graph, producer_id, consumer_id);
                }
                FusionType::MulReLU => {
                    self.fuse_mul_relu(graph, producer_id, consumer_id);
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

    fn add_batchnorm_node(graph: &mut Graph, name: &str, input: &str) -> NodeId {
        add_generic_node(graph, name, OperatorType::BatchNorm, vec![input], vec![&format!("{}_output", name)])
    }

    fn add_mul_node(graph: &mut Graph, name: &str, input1: &str, input2: &str) -> NodeId {
        add_generic_node(graph, name, OperatorType::Mul, vec![input1, input2], vec![&format!("{}_output", name)])
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

    #[test]
    fn test_find_conv_batchnorm_fusion() {
        let fusion = FusionPass::new();
        let mut graph = create_test_graph();

        // Create: Conv -> BatchNorm
        let conv_id = add_conv_node(&mut graph, "conv");
        let _bn_id = add_batchnorm_node(&mut graph, "bn", "conv_output");

        // Add an output consumer
        let _output_node = add_generic_node(&mut graph, "output", OperatorType::Reshape, vec!["bn_output"], vec!["final_output"]);

        let candidates = fusion.find_fusion_candidates(&graph);
        assert!(candidates.iter().any(|(_, _, ft)| *ft == FusionType::ConvBatchNorm));
    }

    #[test]
    fn test_conv_batchnorm_fusion() {
        let fusion = FusionPass::new();
        let mut graph = create_test_graph();

        // Create: Conv -> BatchNorm
        let conv_id = add_conv_node(&mut graph, "conv");
        let bn_id = add_batchnorm_node(&mut graph, "bn", "conv_output");

        // Add an output consumer
        let _output_node = add_generic_node(&mut graph, "output", OperatorType::Reshape, vec!["bn_output"], vec!["final_output"]);

        // Apply fusion
        fusion.optimize(&mut graph);

        // Check that BatchNorm is removed
        let bn_exists = graph.nodes.iter().any(|n| n.id == bn_id);
        assert!(!bn_exists, "BatchNorm node should be removed after fusion");

        // Check that Conv has fusion info
        let conv_node = graph.nodes.iter().find(|n| n.id == conv_id);
        assert!(conv_node.is_some());
        let conv = conv_node.unwrap();
        assert!(conv.fusion.is_some());
        assert_eq!(conv.fusion.as_ref().unwrap().fusion_type, FusionType::ConvBatchNorm);
        assert!(conv.fusion.as_ref().unwrap().eliminate_batch_norm);
    }

    #[test]
    fn test_conv_batchnorm_fusion_eliminates_bn() {
        let fusion = FusionPass::new();
        let mut graph = create_test_graph();

        // Create: Conv -> BatchNorm -> Output
        let conv_id = add_conv_node(&mut graph, "conv");
        let _bn_id = add_batchnorm_node(&mut graph, "bn", "conv_output");
        let output_id = add_generic_node(&mut graph, "output", OperatorType::Reshape, vec!["bn_output"], vec!["final_output"]);

        // Apply fusion
        fusion.optimize(&mut graph);

        // The output node's input should now be conv_output instead of bn_output
        let output_node = graph.nodes.iter().find(|n| n.id == output_id);
        assert!(output_node.is_some());
        let output = output_node.unwrap();
        assert_eq!(output.inputs[0].tensor_name, "conv_output");
    }

    #[test]
    fn test_find_batchnorm_relu_fusion() {
        let fusion = FusionPass::new();
        let mut graph = create_test_graph();

        // Create: BatchNorm -> ReLU
        let bn_id = add_batchnorm_node(&mut graph, "bn", "input");
        let _relu_id = add_relu_node(&mut graph, "relu", "bn_output");

        // Add an output consumer
        let _output_node = add_generic_node(&mut graph, "output", OperatorType::Reshape, vec!["relu_output"], vec!["final_output"]);

        let candidates = fusion.find_fusion_candidates(&graph);
        assert!(candidates.iter().any(|(_, _, ft)| *ft == FusionType::BatchNormReLU));
    }

    #[test]
    fn test_batchnorm_relu_fusion() {
        let fusion = FusionPass::new();
        let mut graph = create_test_graph();

        // Create: BatchNorm -> ReLU
        let bn_id = add_batchnorm_node(&mut graph, "bn", "input");
        let relu_id = add_relu_node(&mut graph, "relu", "bn_output");

        // Add an output consumer
        let _output_node = add_generic_node(&mut graph, "output", OperatorType::Reshape, vec!["relu_output"], vec!["final_output"]);

        // Apply fusion
        fusion.optimize(&mut graph);

        // Check that ReLU is removed
        let relu_exists = graph.nodes.iter().any(|n| n.id == relu_id);
        assert!(!relu_exists, "ReLU node should be removed after fusion");

        // Check that BatchNorm has fusion info
        let bn_node = graph.nodes.iter().find(|n| n.id == bn_id);
        assert!(bn_node.is_some());
        let bn = bn_node.unwrap();
        assert!(bn.fusion.is_some());
        assert_eq!(bn.fusion.as_ref().unwrap().fusion_type, FusionType::BatchNormReLU);
    }

    #[test]
    fn test_batchnorm_relu_fusion_eliminates_relu() {
        let fusion = FusionPass::new();
        let mut graph = create_test_graph();

        // Create: BatchNorm -> ReLU -> Output
        let _bn_id = add_batchnorm_node(&mut graph, "bn", "input");
        let _relu_id = add_relu_node(&mut graph, "relu", "bn_output");
        let output_id = add_generic_node(&mut graph, "output", OperatorType::Reshape, vec!["relu_output"], vec!["final_output"]);

        // Apply fusion
        fusion.optimize(&mut graph);

        // The output node's input should now be bn_output instead of relu_output
        let output_node = graph.nodes.iter().find(|n| n.id == output_id);
        assert!(output_node.is_some());
        let output = output_node.unwrap();
        assert_eq!(output.inputs[0].tensor_name, "bn_output");
    }

    #[test]
    fn test_find_mul_relu_fusion() {
        let fusion = FusionPass::new();
        let mut graph = create_test_graph();

        // Create: Mul -> ReLU
        let mul_id = add_mul_node(&mut graph, "mul", "input1", "input2");
        let _relu_id = add_relu_node(&mut graph, "relu", "mul_output");

        // Add an output consumer
        let _output_node = add_generic_node(&mut graph, "output", OperatorType::Reshape, vec!["relu_output"], vec!["final_output"]);

        let candidates = fusion.find_fusion_candidates(&graph);
        assert!(candidates.iter().any(|(_, _, ft)| *ft == FusionType::MulReLU));
    }

    #[test]
    fn test_mul_relu_fusion() {
        let fusion = FusionPass::new();
        let mut graph = create_test_graph();

        // Create: Mul -> ReLU
        let mul_id = add_mul_node(&mut graph, "mul", "input1", "input2");
        let relu_id = add_relu_node(&mut graph, "relu", "mul_output");

        // Add an output consumer
        let _output_node = add_generic_node(&mut graph, "output", OperatorType::Reshape, vec!["relu_output"], vec!["final_output"]);

        // Apply fusion
        fusion.optimize(&mut graph);

        // Check that ReLU is removed
        let relu_exists = graph.nodes.iter().any(|n| n.id == relu_id);
        assert!(!relu_exists, "ReLU node should be removed after fusion");

        // Check that Mul has fusion info
        let mul_node = graph.nodes.iter().find(|n| n.id == mul_id);
        assert!(mul_node.is_some());
        let mul = mul_node.unwrap();
        assert!(mul.fusion.is_some());
        assert_eq!(mul.fusion.as_ref().unwrap().fusion_type, FusionType::MulReLU);
    }

    #[test]
    fn test_mul_relu_fusion_eliminates_relu() {
        let fusion = FusionPass::new();
        let mut graph = create_test_graph();

        // Create: Mul -> ReLU -> Output
        let _mul_id = add_mul_node(&mut graph, "mul", "input1", "input2");
        let _relu_id = add_relu_node(&mut graph, "relu", "mul_output");
        let output_id = add_generic_node(&mut graph, "output", OperatorType::Reshape, vec!["relu_output"], vec!["final_output"]);

        // Apply fusion
        fusion.optimize(&mut graph);

        // The output node's input should now be mul_output instead of relu_output
        let output_node = graph.nodes.iter().find(|n| n.id == output_id);
        assert!(output_node.is_some());
        let output = output_node.unwrap();
        assert_eq!(output.inputs[0].tensor_name, "mul_output");
    }
}

#[cfg(test)]
mod shape_inference_tests {
    use super::*;
    use crate::common::DataType;
    use std::collections::HashMap;

    fn make_node_io(name: &str) -> NodeIO {
        NodeIO {
            tensor_name: name.to_string(),
            data_type: DataType::F32,
        }
    }

    #[test]
    fn test_shape_inference_relu_preserves_shape() {
        let si = ShapeInference::new();
        let mut node = Node::new(0, "relu".to_string(), OperatorType::ReLU);
        node.inputs.push(make_node_io("input"));
        node.outputs.push(make_node_io("output"));

        let mut input_shapes = HashMap::new();
        input_shapes.insert("input".to_string(), vec![1, 64, 32, 32]);

        let result = si.infer_shape(&node, &input_shapes);
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0], vec![1, 64, 32, 32]);
    }

    #[test]
    fn test_shape_inference_conv2d() {
        let si = ShapeInference::new();
        let mut node = Node::new(0, "conv".to_string(), OperatorType::Conv2d);
        node.inputs.push(make_node_io("input"));
        node.outputs.push(make_node_io("output"));

        let mut input_shapes = HashMap::new();
        input_shapes.insert("input".to_string(), vec![1, 3, 32, 32]);  // NCHW

        let result = si.infer_shape(&node, &input_shapes);
        assert!(result.is_some());
        // Default: 3x3 kernel, stride 1, pad 0
        // out_h = (32 + 0 - 3) / 1 + 1 = 30
        // out_w = (32 + 0 - 3) / 1 + 1 = 30
        assert_eq!(result.unwrap()[0], vec![1, 3, 30, 30]);
    }

    #[test]
    fn test_shape_inference_batchnorm() {
        let si = ShapeInference::new();
        let mut node = Node::new(0, "bn".to_string(), OperatorType::BatchNorm);
        node.inputs.push(make_node_io("input"));
        node.outputs.push(make_node_io("output"));

        let mut input_shapes = HashMap::new();
        input_shapes.insert("input".to_string(), vec![1, 64, 16, 16]);

        let result = si.infer_shape(&node, &input_shapes);
        assert!(result.is_some());
        // BatchNorm preserves shape
        assert_eq!(result.unwrap()[0], vec![1, 64, 16, 16]);
    }

    #[test]
    fn test_shape_inference_matmul() {
        let si = ShapeInference::new();
        let mut node = Node::new(0, "matmul".to_string(), OperatorType::MatMul);
        node.inputs.push(make_node_io("input"));
        node.outputs.push(make_node_io("output"));

        let mut input_shapes = HashMap::new();
        input_shapes.insert("input".to_string(), vec![32, 128]);  // [batch, features]

        let result = si.infer_shape(&node, &input_shapes);
        assert!(result.is_some());
        // Output: [batch, 512] (default out_features)
        assert_eq!(result.unwrap()[0], vec![32, 512]);
    }

    #[test]
    fn test_shape_inference_maxpool() {
        let si = ShapeInference::new();
        let mut node = Node::new(0, "pool".to_string(), OperatorType::MaxPool2d);
        node.inputs.push(make_node_io("input"));
        node.outputs.push(make_node_io("output"));

        let mut input_shapes = HashMap::new();
        input_shapes.insert("input".to_string(), vec![1, 64, 32, 32]);

        let result = si.infer_shape(&node, &input_shapes);
        assert!(result.is_some());
        // 2x2 pool with stride 2: 32 -> 16
        assert_eq!(result.unwrap()[0], vec![1, 64, 16, 16]);
    }

    #[test]
    fn test_shape_inference_global_avg_pool() {
        let si = ShapeInference::new();
        let mut node = Node::new(0, "gap".to_string(), OperatorType::GlobalAvgPool2d);
        node.inputs.push(make_node_io("input"));
        node.outputs.push(make_node_io("output"));

        let mut input_shapes = HashMap::new();
        input_shapes.insert("input".to_string(), vec![1, 64, 8, 8]);

        let result = si.infer_shape(&node, &input_shapes);
        assert!(result.is_some());
        // Global pool reduces H,W to 1
        assert_eq!(result.unwrap()[0], vec![1, 64, 1, 1]);
    }

    #[test]
    fn test_shape_inference_flatten() {
        let si = ShapeInference::new();
        let mut node = Node::new(0, "flatten".to_string(), OperatorType::Flatten);
        node.inputs.push(make_node_io("input"));
        node.outputs.push(make_node_io("output"));

        let mut input_shapes = HashMap::new();
        input_shapes.insert("input".to_string(), vec![4, 8, 16, 16]);  // [batch, C, H, W]

        let result = si.infer_shape(&node, &input_shapes);
        assert!(result.is_some());
        // Flatten: [4, 8*16*16] = [4, 2048]
        assert_eq!(result.unwrap()[0], vec![4, 2048]);
    }

    #[test]
    fn test_shape_inference_transpose() {
        let si = ShapeInference::new();
        let mut node = Node::new(0, "transpose".to_string(), OperatorType::Transpose);
        node.inputs.push(make_node_io("input"));
        node.outputs.push(make_node_io("output"));

        let mut input_shapes = HashMap::new();
        input_shapes.insert("input".to_string(), vec![1, 3, 32, 32]);  // NCHW

        let result = si.infer_shape(&node, &input_shapes);
        assert!(result.is_some());
        // Transpose: NCHW -> NHWC = [1, 32, 32, 3]
        assert_eq!(result.unwrap()[0], vec![1, 32, 32, 3]);
    }

    #[test]
    fn test_shape_inference_softmax() {
        let si = ShapeInference::new();
        let mut node = Node::new(0, "softmax".to_string(), OperatorType::Softmax);
        node.inputs.push(make_node_io("input"));
        node.outputs.push(make_node_io("output"));

        let mut input_shapes = HashMap::new();
        input_shapes.insert("input".to_string(), vec![1, 10]);  // [batch, classes]

        let result = si.infer_shape(&node, &input_shapes);
        assert!(result.is_some());
        // Softmax preserves shape
        assert_eq!(result.unwrap()[0], vec![1, 10]);
    }

    #[test]
    fn test_shape_inference_unknown_operator() {
        let si = ShapeInference::new();
        let mut node = Node::new(0, "custom".to_string(), OperatorType::Custom);
        node.inputs.push(make_node_io("input"));
        node.outputs.push(make_node_io("output"));

        let mut input_shapes = HashMap::new();
        input_shapes.insert("input".to_string(), vec![1, 64, 16, 16]);

        // Unknown operators should preserve input shape
        let result = si.infer_shape(&node, &input_shapes);
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0], vec![1, 64, 16, 16]);
    }

    #[test]
    fn test_shape_inference_gelu() {
        let si = ShapeInference::new();
        let mut node = Node::new(0, "gelu".to_string(), OperatorType::GELU);
        node.inputs.push(make_node_io("input"));
        node.outputs.push(make_node_io("output"));

        let mut input_shapes = HashMap::new();
        input_shapes.insert("input".to_string(), vec![1, 512]);

        let result = si.infer_shape(&node, &input_shapes);
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0], vec![1, 512]);
    }

    #[test]
    fn test_shape_inference_add() {
        let si = ShapeInference::new();
        let mut node = Node::new(0, "add".to_string(), OperatorType::Add);
        node.inputs.push(make_node_io("input1"));
        node.inputs.push(make_node_io("input2"));
        node.outputs.push(make_node_io("output"));

        let mut input_shapes = HashMap::new();
        input_shapes.insert("input1".to_string(), vec![1, 64, 32, 32]);

        let result = si.infer_shape(&node, &input_shapes);
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0], vec![1, 64, 32, 32]);
    }
}
