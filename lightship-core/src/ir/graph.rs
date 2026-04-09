//! Graph data structure for LightShip IR

use super::operator::OperatorType;
use super::tensor::Tensor;
use super::FusionInfo;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// Node ID type
pub type NodeId = u32;

/// Node input/output reference
#[derive(Debug, Clone)]
pub struct NodeIO {
    /// Tensor name
    pub tensor_name: String,
    /// Data type
    pub data_type: crate::common::DataType,
}

/// Graph input/output definition
#[derive(Debug, Clone)]
pub struct GraphIO {
    /// Name
    pub name: String,
    /// Node IO
    pub io: NodeIO,
    /// Is model input
    pub is_model_input: bool,
    /// Is model output
    pub is_model_output: bool,
}

/// Compute graph node
#[derive(Debug, Clone)]
pub struct Node {
    /// Node ID
    pub id: NodeId,
    /// Node name
    pub name: String,
    /// Operator type
    pub operator_type: OperatorType,
    /// Input references
    pub inputs: Vec<NodeIO>,
    /// Output references
    pub outputs: Vec<NodeIO>,
    /// Fusion information (if this node is fused with others)
    pub fusion: Option<FusionInfo>,
}

impl Node {
    /// Create a new node
    pub fn new(id: NodeId, name: String, operator_type: OperatorType) -> Self {
        Self {
            id,
            name,
            operator_type,
            inputs: Vec::new(),
            outputs: Vec::new(),
            fusion: None,
        }
    }
}

/// Compute graph
#[derive(Debug, Clone)]
pub struct Graph {
    /// Graph name
    pub name: String,
    /// Graph nodes
    pub nodes: Vec<Node>,
    /// Graph inputs
    pub inputs: Vec<GraphIO>,
    /// Graph outputs
    pub outputs: Vec<GraphIO>,
    /// Static variables (weights)
    pub variables: HashMap<String, Arc<Tensor>>,
    /// Node name index
    node_name_index: HashMap<String, NodeId>,
}

impl Graph {
    /// Create a new empty graph
    pub fn new(name: String) -> Self {
        Self {
            name,
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            variables: HashMap::new(),
            node_name_index: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, mut node: Node) -> NodeId {
        let id = self.nodes.len() as NodeId;
        node.id = id;
        self.node_name_index.insert(node.name.clone(), id);
        self.nodes.push(node);
        id
    }

    /// Get a node by name
    pub fn node(&self, name: &str) -> Option<&Node> {
        self.node_name_index
            .get(name)
            .and_then(|&id| self.nodes.get(id as usize))
    }

    /// Get a node by ID
    pub fn node_by_id(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id as usize)
    }

    /// Update node name in the index
    pub fn update_node_name(&mut self, old_name: &str, new_name: String, node_id: NodeId) {
        self.node_name_index.remove(old_name);
        self.node_name_index.insert(new_name, node_id);
    }

    /// Remove a node by name, updating the index
    pub fn remove_node(&mut self, name: &str) -> Option<Node> {
        if let Some(&id) = self.node_name_index.get(name) {
            self.node_name_index.remove(name);
            if (id as usize) < self.nodes.len() {
                let node = self.nodes.remove(id as usize);
                // Reindex nodes after the removed one
                for i in id as usize..self.nodes.len() {
                    if let Some(node_name) = self.nodes.get(i).map(|n| n.name.clone()) {
                        self.node_name_index.insert(node_name, i as NodeId);
                    }
                }
                return Some(node);
            }
        }
        None
    }

    /// Retain only nodes that match the predicate, updating the index
    pub fn retain_nodes<F>(&mut self, mut pred: F)
    where
        F: FnMut(&Node) -> bool,
    {
        let names_to_retain: HashSet<String> = self
            .nodes
            .iter()
            .filter(|n| pred(n))
            .map(|n| n.name.clone())
            .collect();

        self.nodes.retain(|n| names_to_retain.contains(&n.name));

        // Rebuild the index
        self.node_name_index.clear();
        for (i, node) in self.nodes.iter().enumerate() {
            self.node_name_index.insert(node.name.clone(), i as NodeId);
        }
    }

    /// Perform topological sort on the graph
    /// Returns node IDs in execution order
    pub fn topological_sort(&self) -> Vec<NodeId> {
        let n = self.nodes.len();
        if n == 0 {
            return Vec::new();
        }

        // Build adjacency list and compute in-degrees
        let mut in_degree = vec![0u32; n];
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for node in &self.nodes {
            for input in &node.inputs {
                // Find the producer of this tensor
                if let Some(producer_id) = self.find_tensor_producer(&input.tensor_name) {
                    adjacency
                        .entry(producer_id)
                        .or_default()
                        .push(node.id);
                    in_degree[node.id as usize] += 1;
                }
            }
        }

        // Kahn's algorithm
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(i as NodeId);
            }
        }

        let mut result = Vec::with_capacity(n);
        while let Some(node_id) = queue.pop_front() {
            result.push(node_id);
            if let Some(neighbors) = adjacency.get(&node_id) {
                for &n in neighbors {
                    in_degree[n as usize] -= 1;
                    if in_degree[n as usize] == 0 {
                        queue.push_back(n);
                    }
                }
            }
        }

        // If there's a cycle, we still return what we have
        result
    }

    /// Find the producer node for a tensor
    fn find_tensor_producer(&self, tensor_name: &str) -> Option<NodeId> {
        self.nodes
            .iter()
            .find(|n| n.outputs.iter().any(|o| &o.tensor_name == tensor_name))
            .map(|n| n.id)
    }
}
